from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest
import torch

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.never_give_up import (
    NGU_ATTEMPT_COUNT_KEY,
    NGU_BASELINE_REWARD_SUM_KEY,
    NGU_BASELINE_SAMPLE_COUNT_KEY,
    NeverGiveUpFilter,
    anchor_positive_advantages,
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
        never_give_up=1.0,
        ngu_solved_reward=1.0,
        async_data_buffer_capacity_factor=1000.0,
        max_weight_staleness=None,
    )
    return Namespace(**{**defaults, **overrides})


def nonzero_std(_args, group, **_kwargs):
    return len({sample.reward for sample in group}) > 1


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


def _make_filter(**overrides) -> tuple[NeverGiveUpFilter, RolloutDataSourceWithBuffer]:
    args = _args(**overrides)
    source = RolloutDataSourceWithBuffer(args)
    return NeverGiveUpFilter(args, dynamic_filter=nonzero_std, data_source=source), source


class TestNeverGiveUpFilter:
    def test_group_with_signal_is_kept_with_its_own_baseline(self):
        ngu, source = _make_filter()
        group = _group(7, [0.0, 1.0, 0.0, 1.0])

        assert ngu(ngu._args, group).keep
        assert len(group) == GROUP_SIZE
        assert group[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 2.0
        assert group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == GROUP_SIZE
        assert source.buffer == []

    def test_unsolved_group_without_signal_is_requeued_through_the_buffer(self):
        ngu, source = _make_filter()

        output = ngu(ngu._args, _group(7, [0.0] * GROUP_SIZE, first_index=100))

        assert (output.keep, output.reason) == (False, "never_give_up_requeued")
        [retry] = source.get_samples(1)
        assert [sample.group_index for sample in retry] == [7] * GROUP_SIZE
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert all(sample.status == Sample.Status.PENDING and sample.reward is None for sample in retry)

    def test_solved_group_is_dropped_not_requeued(self):
        ngu, source = _make_filter()

        assert not ngu(ngu._args, _group(7, [1.0] * GROUP_SIZE)).keep
        assert source.buffer == []

    def test_failed_coin_flip_drops_the_chain(self):
        ngu, source = _make_filter(never_give_up=0.5)
        ngu._rng.random = lambda: 0.99

        assert not ngu(ngu._args, _group(7, [0.0] * GROUP_SIZE)).keep
        assert source.buffer == [] and ngu._chains == {}

    def test_improved_retry_is_merged_in_place_with_a_chain_wide_baseline(self):
        ngu, source = _make_filter()
        ngu(ngu._args, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        ngu(ngu._args, _answer(source.get_samples(1)[0], [0.0] * GROUP_SIZE))
        retry = _answer(source.get_samples(1)[0], [0.0, 0.0, 0.0, 1.0])

        assert ngu(ngu._args, retry).keep
        assert len(retry) == 3 * GROUP_SIZE
        assert len({sample.index for sample in retry}) == 3 * GROUP_SIZE
        assert {sample.group_index for sample in retry} == {7}
        assert retry[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 1.0
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 3 * GROUP_SIZE
        assert retry[-1].metadata[NGU_ATTEMPT_COUNT_KEY] == 3
        assert ngu._chains == {}

    def test_a_retry_that_only_ties_the_best_reward_is_requeued_again(self):
        ngu, source = _make_filter(ngu_solved_reward=2.0)
        ngu(ngu._args, _group(7, [0.0, 0.0, 0.0, 1.0]))  # kept: has signal
        ngu(ngu._args, _group(8, [0.0] * GROUP_SIZE))
        tie = _answer(source.get_samples(1)[0], [0.0] * GROUP_SIZE)

        assert not ngu(ngu._args, tie).keep
        assert len(source.buffer) == 1

    def test_stale_attempts_leave_the_merge_but_stay_in_the_baseline(self):
        ngu, source = _make_filter(max_weight_staleness=2)
        ngu(ngu._args, _group(7, [0.0] * GROUP_SIZE, first_index=100, version=0))
        retry = _answer(source.get_samples(1)[0], [0.0, 1.0, 0.0, 0.0], version=5)

        assert ngu(ngu._args, retry).keep
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert retry[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE

    def test_lost_retries_are_requeued(self):
        ngu, source = _make_filter()
        ngu(ngu._args, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        source.get_samples(1)  # the retry is drawn, then lost (e.g. aborted at the end of a rollout)

        ngu.requeue_lost_retries()

        assert [group[0].group_index for group in source.buffer] == [7]
        ngu.requeue_lost_retries()  # already buffered, so not requeued twice
        assert len(source.buffer) == 1


class TestHelpers:
    def test_make_retry_group_clears_generated_outputs(self):
        [retry] = make_retry_group(_group(3, [1.0])[0], sample_indices=[9])

        assert (retry.response, retry.reward, retry.index, retry.group_index) == ("", None, 9, 3)

    def test_prune_keeps_fresh_attempts_and_always_the_kept_one(self):
        stale, fresh, kept = (
            _group(7, [0.0] * 2),
            _group(7, [0.0] * 2, first_index=2),
            _group(7, [1.0, 0.0], first_index=4),
        )
        pruned = prune_stale_attempts(
            [*stale, *fresh, *kept], attempt_size=2, is_stale=lambda attempt: attempt[0] in (stale[0], kept[0])
        )

        assert [sample.index for sample in pruned] == [2, 3, 4, 5]

    def test_anchor_keeps_positives_and_sums_to_zero(self):
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        out = anchor_positive_advantages(rewards - 0.125, rewards)

        assert out[3] == pytest.approx(0.875)
        assert float(out.sum()) == pytest.approx(0.0, abs=1e-6)
        assert torch.allclose(out[:3], torch.full((3,), -0.875 / 3))

    def test_anchor_leaves_equal_rewards_alone(self):
        advantages = torch.tensor([0.5, 0.5])
        assert torch.equal(anchor_positive_advantages(advantages, torch.tensor([1.0, 1.0])), advantages)


class TestGenerateRolloutWithNeverGiveUp:
    def test_retried_prompt_trains_as_one_merged_group(self, monkeypatch):
        """Chain 0 has no signal on its first attempt and solves one sample on its retry."""
        args = _args(
            rollout_global_dataset=True,
            rollout_batch_size=2,
            over_sampling_batch_size=1,
            rollout_submission_granularity=None,
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            partial_rollout=False,
        )
        source = RolloutDataSourceWithBuffer(_args())
        ngu = NeverGiveUpFilter(args, dynamic_filter=nonzero_std, data_source=source)
        attempts_of_chain: dict[int, int] = {}

        async def generate(group):
            attempt = attempts_of_chain.get(group[0].group_index, 0)
            attempts_of_chain[group[0].group_index] = attempt + 1
            no_signal = group[0].group_index == 0 and attempt == 0
            return _answer(group, [0.0] * GROUP_SIZE if no_signal else [1.0, 0.0, 0.0, 0.0])

        async def fake_abort(_state, pendings, _rollout_id):
            for task in pendings:
                task.cancel()
            await asyncio.gather(*pendings, return_exceptions=True)
            return []

        async def noop(*_args, **_kwargs):
            return None

        state = Namespace(args=args, sampling_params={}, aborted=False, reset=lambda: None)
        monkeypatch.setattr(
            train,
            "submit_generate_tasks",
            lambda _state, samples, sample_done_callback=None: [asyncio.ensure_future(generate(g)) for g in samples],
        )
        monkeypatch.setattr(train, "abort", fake_abort)
        monkeypatch.setattr(train, "load_function", lambda path: None)
        monkeypatch.setattr(train.dumper_utils, "configure_sglang", noop)
        monkeypatch.setattr(train, "recompute_samples_rollout_logprobs_via_prefill", noop)

        output, _ = asyncio.run(train.generate_rollout_async(state, 0, source.get_samples, dynamic_filter=ngu))

        groups = {group[0].group_index: group for group in output.samples}
        assert len(groups[0]) == 2 * GROUP_SIZE
        assert groups[0][0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE
        assert output.metrics["rollout/dynamic_filter/drop_never_give_up_requeued"] == 1


class TestFullyAsyncDataBufferWithNeverGiveUp:
    @staticmethod
    def _buffer(**overrides):
        args = _args(**overrides)
        source = RolloutDataSourceWithBuffer(args)
        buffer = DefaultDataBuffer(
            DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None, data_source=source)
        )
        return buffer, source

    @staticmethod
    async def _put(buffer, group):
        await buffer.put(DataBufferInput(prompt_group=group, group=group))

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

    def test_fresh_merged_group_is_trained_whole(self):
        async def run():
            buffer, source = self._buffer(max_weight_staleness=2)
            await self._put(buffer, _group(0, [0.0] * GROUP_SIZE, first_index=100, version=4))
            await self._put(buffer, _answer(source.get_samples(1)[0], [1.0, 0.0, 0.0, 0.0], version=4))
            return await buffer.get(current_version=5)

        assert len(asyncio.run(run()).group) == 2 * GROUP_SIZE


class TestValidateNeverGiveUpArgs:
    @staticmethod
    def _args(**overrides) -> Namespace:
        defaults = dict(
            never_give_up=0.5,
            dynamic_sampling_filter_path="miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
            rollout_function_path=None,
            fully_async=False,
            partial_rollout=False,
            use_dynamic_global_batch_size=True,
            custom_async_data_buffer_path=None,
        )
        return Namespace(**{**defaults, **overrides})

    def test_supported_setups_pass(self):
        _validate_never_give_up_args(self._args())
        _validate_never_give_up_args(self._args(fully_async=True))

    def test_disabled_skips_every_other_check(self):
        _validate_never_give_up_args(self._args(never_give_up=0.0, dynamic_sampling_filter_path=None))

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(never_give_up=1.5),
            dict(dynamic_sampling_filter_path=None),
            dict(fully_async=True, custom_async_data_buffer_path="my.Buffer"),
            dict(partial_rollout=True),
            dict(use_dynamic_global_batch_size=False),
        ],
    )
    def test_unsupported_setups_are_rejected(self, overrides):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(**overrides))
