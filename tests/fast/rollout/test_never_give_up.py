from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest
import torch

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.filter_hub.common_filters import FilterReason
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.never_give_up import (
    NGU_BASELINE_REWARD_SQ_SUM_KEY,
    NGU_BASELINE_REWARD_SUM_KEY,
    NGU_BASELINE_SAMPLE_COUNT_KEY,
    NeverGiveUp,
    chain_rewards_and_std,
    make_retry_group,
    prune_stale_attempts,
)
from miles.utils.arguments import _validate_never_give_up_args
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

GROUP_SIZE = 4


def _args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=False,
        rollout_batch_size=1,
        n_samples_per_prompt=GROUP_SIZE,
        buffer_filter_path=None,
        rollout_seed=0,
        reward_key=None,
        dynamic_sampling_filter_path=f"{__name__}.nonzero_std",
        async_unused_samples_handler="never_give_up",
        ngu_requeue_probability=1.0,
        ngu_solved_reward=1.0,
        async_data_buffer_capacity_factor=1000.0,
        max_weight_staleness=None,
    )
    return Namespace(**{**defaults, **overrides})


def nonzero_std(_args, group, **_kwargs):
    keep = len({sample.reward for sample in group}) > 1
    return FilterOutput(keep=keep, reason=None if keep else "zero_std")


def _group(chain_id: int, rewards: list[float], *, first_index: int = 0, version: int | None = None) -> list[Sample]:
    versions = [] if version is None else [WeightVersionsPerCall(spans=[WeightVersionSpan(str(version), 0, 1)])]
    return [
        Sample(
            group_index=chain_id,
            index=first_index + i,
            prompt="prompt",
            response="response",
            response_length=1,
            reward=reward,
            status=Sample.Status.COMPLETED,
            weight_versions=list(versions),
        )
        for i, reward in enumerate(rewards)
    ]


def _answer(retry: list[Sample], rewards: list[float], version: int | None = None) -> list[Sample]:
    """Stand in for generation: give a requeued retry its rewards."""
    for sample, reward in zip(retry, rewards, strict=True):
        sample.reward, sample.status = reward, Sample.Status.COMPLETED
        if version is not None:
            sample.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(str(version), 0, 1)])]
    return retry


def _make_ngu(**overrides) -> tuple[NeverGiveUp, RolloutDataSourceWithBuffer]:
    args = _args(**overrides)
    source = RolloutDataSourceWithBuffer(args)
    return NeverGiveUp(args, data_source=source), source


def _offer(ngu: NeverGiveUp, group: list[Sample]) -> FilterOutput:
    """What a rollout does with a finished group: filter it, then tell the handler what became of it."""
    output = nonzero_std(ngu._args, group)
    ngu(group, group=group, reason=FilterReason.kept if output.keep else output.reason)
    return output


class TestNeverGiveUp:
    def test_group_with_signal_is_kept_with_its_own_baseline(self):
        ngu, source = _make_ngu()
        group = _group(7, [0.0, 1.0, 0.0, 1.0])

        assert _offer(ngu, group).keep
        assert len(group) == GROUP_SIZE
        assert group[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 2.0
        assert group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == GROUP_SIZE
        assert source.buffer == []

    def test_unsolved_group_without_signal_is_requeued_through_the_buffer(self):
        ngu, source = _make_ngu()

        output = _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))

        assert (output.keep, output.reason) == (False, "zero_std")
        [retry] = source.get_samples(1)
        assert [sample.group_index for sample in retry] == [7] * GROUP_SIZE
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert all(sample.reward is None for sample in retry)

    def test_solved_group_is_dropped_not_requeued(self):
        ngu, source = _make_ngu()

        assert not _offer(ngu, _group(7, [1.0] * GROUP_SIZE)).keep
        assert source.buffer == []

    def test_failed_draw_drops_the_chain(self):
        ngu, source = _make_ngu(ngu_requeue_probability=0.5)
        ngu._rng.random = lambda: 0.99

        assert not _offer(ngu, _group(7, [0.0] * GROUP_SIZE)).keep
        assert source.buffer == [] and ngu._chains == {}

    def test_improved_retry_is_merged_in_place_with_a_chain_wide_baseline(self):
        ngu, source = _make_ngu()
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        _offer(ngu, _answer(source.get_samples(1)[0], [0.0] * GROUP_SIZE))
        retry = _answer(source.get_samples(1)[0], [0.0, 0.0, 0.0, 1.0])

        assert _offer(ngu, retry).keep
        assert len(retry) == 3 * GROUP_SIZE
        assert len({sample.index for sample in retry}) == 3 * GROUP_SIZE
        assert {sample.group_index for sample in retry} == {7}
        assert retry[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 1.0
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 3 * GROUP_SIZE
        assert ngu._chains == {}

    def test_a_retry_the_filter_keeps_is_merged_even_without_beating_the_earlier_best(self):
        ngu, source = _make_ngu(ngu_solved_reward=2.0)
        _offer(ngu, _group(9, [0.5] * GROUP_SIZE, first_index=100))
        retry = _answer(source.get_samples(1)[0], [0.0, 0.0, 0.0, 0.5])

        assert _offer(ngu, retry).keep
        assert len(retry) == 2 * GROUP_SIZE
        assert source.buffer == []

    def test_stale_attempts_leave_the_merge_but_stay_in_the_baseline(self):
        ngu, source = _make_ngu(max_weight_staleness=2)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100, version=0))
        retry = _answer(source.get_samples(1)[0], [0.0, 1.0, 0.0, 0.0], version=5)

        assert _offer(ngu, retry).keep
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE


class TestStaleGroups:
    def test_a_stale_group_keeps_going_and_its_retry_trains_with_its_completions_merged_back(self):
        ngu, source = _make_ngu(ngu_requeue_probability=0.0, ngu_solved_reward=2.0)
        stale = _group(7, [0.0, 0.0, 0.0, 1.0], first_index=100, version=1)
        assert _offer(ngu, stale).keep
        ngu(stale, group=stale, reason=FilterReason.stale)  # requeued even though requeues are refused
        [retry] = source.get_samples(1)
        assert all(NGU_BASELINE_REWARD_SUM_KEY not in sample.metadata for sample in retry)
        retry = _answer(retry, [0.0, 1.0, 0.0, 0.0], version=9)

        assert _offer(ngu, retry).keep

        assert len(retry) == 2 * GROUP_SIZE  # the stale attempt's completions are merged back
        assert retry[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 2.0
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE

    def test_a_stale_merged_group_keeps_its_chain_baseline(self):
        ngu, source = _make_ngu(ngu_solved_reward=2.0)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100, version=1))
        merged = _answer(source.get_samples(1)[0], [0.0, 0.0, 0.0, 1.0], version=2)
        assert _offer(ngu, merged).keep and len(merged) == 2 * GROUP_SIZE
        merged[:] = merged[GROUP_SIZE:]  # the fully-async buffer pruned the older attempt on consume

        ngu(merged, group=merged, reason=FilterReason.stale)
        retry = _answer(source.get_samples(1)[0], [0.0, 1.0, 0.0, 0.0], version=9)

        assert _offer(ngu, retry).keep
        assert retry[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 2.0  # 1.0 from the first, 1.0 from the retry
        assert retry[0].metadata[NGU_BASELINE_REWARD_SQ_SUM_KEY] == 2.0
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 3 * GROUP_SIZE


class TestAbortedAndMissingReward:
    @pytest.mark.parametrize("reason", [FilterReason.aborted, FilterReason.missing_reward])
    def test_a_failed_retry_is_requeued_as_it_was(self, reason):
        ngu, source = _make_ngu()
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        retry = source.get_samples(1)[0]

        ngu(retry, group=retry, reason=reason)

        [requeued] = source.buffer
        assert requeued[0].group_index == 7
        assert [sample.index for sample in requeued] != [sample.index for sample in retry]
        assert len(ngu._chains[7].samples) == GROUP_SIZE  # nothing was learned from it

    @pytest.mark.parametrize("reason", [FilterReason.aborted, FilterReason.missing_reward])
    def test_a_failed_first_attempt_is_dropped(self, reason):
        ngu, source = _make_ngu()
        group = _group(7, [0.0] * GROUP_SIZE)

        ngu(group, group=group, reason=reason)

        assert source.buffer == [] and ngu._chains == {}


class TestHelpers:
    def test_make_retry_group_clears_generated_outputs(self):
        [retry] = make_retry_group(_group(3, [1.0])[0], sample_indices=[9])

        assert (retry.response, retry.reward, retry.index, retry.group_index) == ("", None, 9, 3)

    def test_prune_keeps_fresh_attempts_and_always_the_kept_one(self):
        stale, fresh, kept = (
            _group(7, [0.0] * 2, version=1),
            _group(7, [0.0] * 2, first_index=2, version=4),
            _group(7, [1.0, 0.0], first_index=4, version=1),
        )
        pruned = prune_stale_attempts([*stale, *fresh, *kept], attempt_size=2, current_version=5, max_staleness=2)

        assert [sample.index for sample in pruned] == [2, 3, 4, 5]

    def test_chain_shift_moves_only_the_negatives_to_the_chain_baseline(self):
        group = _group(7, [1.0, 0.0, 0.0, 0.0])
        group[0].metadata.update(
            ngu_baseline_reward_sum=1.0, ngu_baseline_reward_sq_sum=1.0, ngu_baseline_sample_count=8
        )

        rewards, std = chain_rewards_and_std(group, torch.tensor([1.0, 0.0, 0.0, 0.0]))

        assert rewards.tolist() == pytest.approx([1.0, -1 / 6, -1 / 6, -1 / 6])
        assert float(rewards.mean()) == pytest.approx(0.125)
        assert float(std) == pytest.approx(0.125**0.5)

    def test_chain_shift_is_skipped_when_a_negative_would_overtake_the_max(self):
        group = _group(7, [0.6, 0.2, 0.0, 0.0])  # stale [0.9] * 4: b = 0.55 would lift 0.2 to 0.667
        group[0].metadata.update(
            ngu_baseline_reward_sum=4.4, ngu_baseline_reward_sq_sum=3.64, ngu_baseline_sample_count=8
        )
        rewards = torch.tensor([0.6, 0.2, 0.0, 0.0])

        assert torch.equal(chain_rewards_and_std(group, rewards)[0], rewards)

    def test_a_group_without_chain_stats_is_left_alone(self):
        rewards = torch.tensor([1.0, 0.0])
        assert chain_rewards_and_std(_group(7, [1.0, 0.0]), rewards) == (rewards, None)


class TestFullyAsyncDataBufferWithNeverGiveUp:
    @staticmethod
    def _buffer(**overrides):
        args = _args(**overrides)
        source = RolloutDataSourceWithBuffer(args)
        ngu = NeverGiveUp(args, data_source=source)
        return DefaultDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=ngu)), source

    @staticmethod
    async def _put(buffer, group):
        await buffer.put(DataBufferInput(prompt_group=group, group=group))

    def test_a_rejected_group_is_requeued_and_the_metric_names_the_real_reason(self):
        async def run():
            buffer, source = self._buffer()
            await self._put(buffer, _group(0, [0.0] * GROUP_SIZE, first_index=100))
            return source, buffer.get_metrics()

        source, metrics = asyncio.run(run())

        assert [group[0].group_index for group in source.buffer] == [0]
        assert metrics["rollout/dynamic_filter/drop_zero_std"] == 1

    def test_an_aborted_retry_is_requeued_instead_of_losing_the_chain(self):
        async def run():
            buffer, source = self._buffer()
            await self._put(buffer, _group(0, [0.0] * GROUP_SIZE, first_index=100))
            retry = source.get_samples(1)[0]
            for sample in retry:
                sample.status = Sample.Status.ABORTED
            await self._put(buffer, retry)
            return source

        assert [group[0].group_index for group in asyncio.run(run()).buffer] == [0]

    def test_consuming_a_merged_group_prunes_stale_attempts_instead_of_dropping_it(self):
        async def run():
            buffer, source = self._buffer(max_weight_staleness=2)
            await self._put(buffer, _group(0, [0.0] * GROUP_SIZE, first_index=100, version=1))
            retry = _answer(source.get_samples(1)[0], [1.0, 0.0, 0.0, 0.0], version=4)
            await self._put(buffer, retry)
            return await buffer.get(current_version=5), buffer.get_metrics()

        entry, metrics = asyncio.run(run())

        assert [sample.index for sample in entry.group] == [0, 1, 2, 3]  # only the retry
        assert entry.group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE
        assert metrics["rollout/fully_async/stale_groups_filtered"] == 0

    def test_a_group_whose_kept_attempt_is_stale_becomes_a_failure_that_keeps_going(self):
        async def run():
            buffer, source = self._buffer(max_weight_staleness=2)
            await self._put(buffer, _group(0, [0.0, 0.0, 0.0, 1.0], first_index=100, version=1))
            buffer._current_version = 9
            fresh = _group(1, [1.0, 0.0, 0.0, 0.0], first_index=200, version=9)
            await self._put(buffer, fresh)
            return await buffer.get(current_version=9), source, buffer.get_metrics()

        entry, source, metrics = asyncio.run(run())

        assert entry.group[0].group_index == 1  # the fresh group trains
        assert metrics["rollout/fully_async/stale_groups_filtered"] == 1
        [retry] = source.buffer
        assert retry[0].group_index == 0


class TestValidateNeverGiveUpArgs:
    @staticmethod
    def _args(**overrides) -> Namespace:
        defaults = dict(
            async_unused_samples_handler="never_give_up",
            ngu_requeue_probability=0.5,
            dynamic_sampling_filter_path="miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
            fully_async=True,
            partial_rollout=False,
            use_dynamic_global_batch_size=True,
            custom_async_data_buffer_path=None,
        )
        return Namespace(**{**defaults, **overrides})

    def test_supported_setups_pass(self):
        _validate_never_give_up_args(self._args())

    def test_other_handlers_skip_every_other_check(self):
        _validate_never_give_up_args(
            self._args(async_unused_samples_handler="drop", dynamic_sampling_filter_path=None)
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(ngu_requeue_probability=1.5),
            dict(dynamic_sampling_filter_path=None),
            dict(fully_async=False),
            dict(custom_async_data_buffer_path="my.Buffer"),
            dict(partial_rollout=True),
            dict(use_dynamic_global_batch_size=False),
        ],
    )
    def test_unsupported_setups_are_rejected(self, overrides):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(**overrides))
