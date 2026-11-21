import asyncio
from argparse import Namespace

import pytest
import torch
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group, train_input

from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.never_give_up import NeverGiveUp, group_size_metrics, make_retry_group, prune_stale_attempts
from miles.rollout.resampling import REWARD_BASELINE_KEY, GroupOutcome, baseline_rewards_and_std
from miles.utils.arguments import _validate_never_give_up_args
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

GROUP_SIZE = 4


def _args(**overrides) -> Namespace:
    defaults = dict(
        rollout_batch_size=1,
        n_samples_per_prompt=GROUP_SIZE,
        rollout_seed=0,
        reward_key=None,
        dynamic_sampling_filter_path=f"{__name__}.nonzero_std",
        async_unused_samples_handler="never_give_up",
        ngu_max_budget=None,
        ngu_solved_reward=1.0,
        async_data_buffer_capacity_factor=1000.0,
        async_data_buffer_selection="fifo",
        async_submit_within_staleness=False,
        enable_sample_ownership_checker=False,
        use_output_store=False,
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


class _Requeued(list):
    """The rollout function's retry queue, plus the data source's sample index counter."""

    def __init__(self):
        super().__init__()
        self.next_index = 0

    def reserve(self, num_samples: int) -> list[int]:
        indices = list(range(self.next_index, self.next_index + num_samples))
        self.next_index += num_samples
        return indices


def _make_ngu(admission_changed=lambda: None, **overrides) -> tuple[NeverGiveUp, _Requeued]:
    requeued = _Requeued()
    ngu = NeverGiveUp(
        _args(**overrides),
        admission_changed=admission_changed,
        requeue=requeued.append,
        reserve_sample_indices=requeued.reserve,
    )
    return ngu, requeued


def _entry(group: list[Sample]) -> DataBufferInput:
    return DataBufferInput(prompt_group=group, group=group)


def _offer(ngu: NeverGiveUp, group: list[Sample]) -> FilterOutput:
    """What the buffer does with a finished group: filter it, then tell the policy what became of it."""
    output = nonzero_std(ngu._args, group)
    ngu(_entry(group), GroupOutcome.KEPT if output.keep else GroupOutcome.FILTERED)
    return output


class TestNeverGiveUp:
    def test_metrics_include_exhausted_chains_and_reset_without_losing_pending_state(self):
        ngu, requeued = _make_ngu(ngu_max_budget=4)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE))
        first = ngu.collect_metrics()
        assert first["rollout/ngu/filter_retries"] == 1
        assert first["rollout/ngu/pending_chains"] == 1
        assert ngu.collect_metrics()["rollout/ngu/filter_retries"] == 0
        for _ in range(3):
            _offer(ngu, _answer(requeued.pop(0), [0.0] * GROUP_SIZE))
        final = ngu.collect_metrics()
        assert final["rollout/ngu/filter_retries"] == 2
        assert final["rollout/ngu/filtered_attempts"] == 3
        assert final["rollout/ngu/budget_exhausted_chains"] == 1
        assert final["rollout/ngu/pending_chains"] == 0

    def test_accepted_retry_metrics_count_pruned_history_and_separate_recovery(self):
        ngu, requeued = _make_ngu(max_weight_staleness=1)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, version=1))
        ngu(_entry(requeued.pop(0)), GroupOutcome.ABORTED)
        _offer(ngu, _answer(requeued.pop(0), [0.0, 0.0, 0.0, 1.0], version=4))
        _offer(ngu, _group(8, [0.0, 0.0, 0.0, 1.0], version=4))
        _offer(ngu, _group(9, [1.0] * GROUP_SIZE))
        metrics = ngu.collect_metrics()
        assert metrics["rollout/ngu/accepted_chains"] == 2
        assert metrics["rollout/ngu/accepted_retried_fraction"] == 0.5
        assert metrics["rollout/ngu/accepted_attempts"] == 3
        assert metrics["rollout/ngu/recovery_retries"] == 1
        assert metrics["rollout/ngu/solved_filtered_chains"] == 1

    def test_group_with_signal_is_kept_with_its_own_baseline(self):
        ngu, requeued = _make_ngu()
        group = _group(7, [0.0, 1.0, 0.0, 1.0])

        assert _offer(ngu, group).keep
        assert len(group) == GROUP_SIZE
        assert group[0].metadata[REWARD_BASELINE_KEY]["sum"] == 2.0
        assert group[0].metadata[REWARD_BASELINE_KEY]["count"] == GROUP_SIZE
        assert requeued == []

    def test_unsolved_group_without_signal_is_requeued(self):
        ngu, requeued = _make_ngu()

        output = _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))

        assert (output.keep, output.reason) == (False, "zero_std")
        [retry] = requeued
        assert [sample.group_index for sample in retry] == [7] * GROUP_SIZE
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert all(sample.reward is None for sample in retry)

    def test_solved_group_is_dropped_not_requeued(self):
        ngu, requeued = _make_ngu()

        assert not _offer(ngu, _group(7, [1.0] * GROUP_SIZE)).keep
        assert requeued == []

    def test_max_budget_requeues_until_the_chain_has_had_that_many_attempts(self):
        ngu, requeued = _make_ngu(ngu_max_budget=3)

        assert not _offer(ngu, _group(7, [0.0] * GROUP_SIZE)).keep
        assert not _offer(ngu, _answer(requeued.pop(0), [0.0] * GROUP_SIZE)).keep
        retry = requeued.pop(0)
        assert ngu._chains[7].rejected_attempts == 2

        assert not _offer(ngu, _answer(retry, [0.0] * GROUP_SIZE)).keep  # the third attempt spends the budget
        assert requeued == [] and ngu._chains == {}

    def test_a_budget_of_one_never_requeues(self):
        ngu, requeued = _make_ngu(ngu_max_budget=1)

        assert not _offer(ngu, _group(7, [0.0] * GROUP_SIZE)).keep
        assert requeued == [] and ngu._chains == {}

    def test_a_retry_within_budget_that_the_filter_keeps_merges_up_to_budget_times_the_group(self):
        ngu, requeued = _make_ngu(ngu_max_budget=2)

        assert not _offer(ngu, _group(7, [0.0] * GROUP_SIZE)).keep
        kept = _answer(requeued.pop(0), [0.0, 1.0, 0.0, 0.0])

        assert _offer(ngu, kept).keep
        assert len(kept) == 2 * GROUP_SIZE
        assert kept[0].metadata[REWARD_BASELINE_KEY]["count"] == 2 * GROUP_SIZE
        assert ngu._chains == {}

    def test_unset_budget_keeps_requeuing_until_solved(self):
        ngu, requeued = _make_ngu()

        assert not _offer(ngu, _group(7, [0.0] * GROUP_SIZE)).keep
        for attempt in range(2, 6):
            assert not _offer(ngu, _answer(requeued.pop(0), [0.0] * GROUP_SIZE)).keep
            assert ngu._chains[7].rejected_attempts == attempt
        assert len(requeued) == 1

    def test_improved_retry_is_merged_in_place_with_a_chain_wide_baseline(self):
        ngu, requeued = _make_ngu()
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        _offer(ngu, _answer(requeued.pop(0), [0.0] * GROUP_SIZE))
        retry = _answer(requeued.pop(0), [0.0, 0.0, 0.0, 1.0])

        assert _offer(ngu, retry).keep
        assert len(retry) == 3 * GROUP_SIZE
        assert len({sample.index for sample in retry}) == 3 * GROUP_SIZE
        assert {sample.group_index for sample in retry} == {7}
        assert retry[0].metadata[REWARD_BASELINE_KEY]["sum"] == 1.0
        assert retry[0].metadata[REWARD_BASELINE_KEY]["count"] == 3 * GROUP_SIZE
        assert ngu._chains == {}

    def test_a_retry_the_filter_keeps_is_merged_even_without_beating_the_earlier_best(self):
        ngu, requeued = _make_ngu(ngu_solved_reward=2.0)
        _offer(ngu, _group(9, [0.5] * GROUP_SIZE, first_index=100))
        retry = _answer(requeued.pop(0), [0.0, 0.0, 0.0, 0.5])

        assert _offer(ngu, retry).keep
        assert len(retry) == 2 * GROUP_SIZE
        assert requeued == []

    def test_stale_attempts_leave_the_merge_but_stay_in_the_baseline(self):
        ngu, requeued = _make_ngu(max_weight_staleness=2)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100, version=0))
        retry = _answer(requeued.pop(0), [0.0, 1.0, 0.0, 0.0], version=5)

        assert _offer(ngu, retry).keep
        assert [sample.index for sample in retry] == [0, 1, 2, 3]
        assert retry[0].metadata[REWARD_BASELINE_KEY]["count"] == 2 * GROUP_SIZE

    def test_chains_survive_a_checkpoint_round_trip(self):
        ngu, requeued = _make_ngu()
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        restored, _ = _make_ngu()
        restored.load_state_dict(ngu.state_dict())

        retry = _answer(requeued.pop(0), [0.0, 1.0, 0.0, 0.0])
        assert _offer(restored, retry).keep
        assert len(retry) == 2 * GROUP_SIZE


class TestStaleGroups:
    def test_a_stale_group_keeps_going_and_its_retry_trains_with_its_completions_merged_back(self):
        ngu, requeued = _make_ngu(ngu_max_budget=1, ngu_solved_reward=2.0)
        stale = _group(7, [0.0, 0.0, 0.0, 1.0], first_index=100, version=1)
        assert _offer(ngu, stale).keep
        ngu(_entry(stale), GroupOutcome.STALE)  # requeued even though a budget of 1 refuses filter retries
        [retry] = requeued
        assert all(REWARD_BASELINE_KEY not in sample.metadata for sample in retry)
        retry = _answer(retry, [0.0, 1.0, 0.0, 0.0], version=9)

        assert _offer(ngu, retry).keep

        assert len(retry) == 2 * GROUP_SIZE  # the stale attempt's completions are merged back
        assert retry[0].metadata[REWARD_BASELINE_KEY]["sum"] == 2.0
        assert retry[0].metadata[REWARD_BASELINE_KEY]["count"] == 2 * GROUP_SIZE

    def test_a_stale_merged_group_keeps_its_chain_baseline(self):
        ngu, requeued = _make_ngu(ngu_solved_reward=2.0)
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100, version=1))
        merged = _answer(requeued.pop(0), [0.0, 0.0, 0.0, 1.0], version=2)
        assert _offer(ngu, merged).keep and len(merged) == 2 * GROUP_SIZE
        merged[:] = merged[GROUP_SIZE:]  # the fully-async buffer pruned the older attempt on consume

        ngu(_entry(merged), GroupOutcome.STALE)
        retry = _answer(requeued.pop(0), [0.0, 1.0, 0.0, 0.0], version=9)

        assert _offer(ngu, retry).keep
        assert retry[0].metadata[REWARD_BASELINE_KEY]["sum"] == 2.0  # 1.0 from the first, 1.0 from the retry
        assert retry[0].metadata[REWARD_BASELINE_KEY]["sq_sum"] == 2.0
        assert retry[0].metadata[REWARD_BASELINE_KEY]["count"] == 3 * GROUP_SIZE


class TestAbortedAndMissingReward:
    REASONS = [GroupOutcome.ABORTED, GroupOutcome.MISSING_REWARD, GroupOutcome.RESTORED]

    @pytest.mark.parametrize("reason", REASONS)
    def test_a_failed_retry_is_requeued_as_it_was(self, reason):
        ngu, requeued = _make_ngu()
        _offer(ngu, _group(7, [0.0] * GROUP_SIZE, first_index=100))
        retry = requeued.pop(0)

        ngu(_entry(retry), reason)

        [again] = requeued
        assert again[0].group_index == 7
        assert [sample.index for sample in again] != [sample.index for sample in retry]
        assert len(ngu._chains[7].samples) == GROUP_SIZE  # nothing was learned from it

    @pytest.mark.parametrize("reason", REASONS)
    def test_a_failed_first_attempt_is_dropped(self, reason):
        ngu, requeued = _make_ngu()

        ngu(_entry(_group(7, [0.0] * GROUP_SIZE)), reason)

        assert requeued == [] and ngu._chains == {}


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
        group[0].metadata[REWARD_BASELINE_KEY] = {"sum": 1.0, "sq_sum": 1.0, "count": 8}

        rewards, std = baseline_rewards_and_std(group, torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor(0.5))

        assert rewards.tolist() == pytest.approx([1.0, -1 / 6, -1 / 6, -1 / 6])
        assert float(rewards.mean()) == pytest.approx(0.125)
        assert float(std) == pytest.approx(0.125**0.5)

    def test_chain_shift_is_skipped_when_a_negative_would_overtake_the_max(self):
        group = _group(7, [0.6, 0.2, 0.0, 0.0])  # stale [0.9] * 4: b = 0.55 would lift 0.2 to 0.667
        group[0].metadata[REWARD_BASELINE_KEY] = {"sum": 4.4, "sq_sum": 3.64, "count": 8}
        rewards = torch.tensor([0.6, 0.2, 0.0, 0.0])

        assert torch.equal(baseline_rewards_and_std(group, rewards, rewards.std())[0], rewards)

    def test_a_group_without_a_baseline_stamp_is_left_alone(self):
        rewards = torch.tensor([1.0, 0.0])
        std = rewards.std()
        assert baseline_rewards_and_std(_group(7, [1.0, 0.0]), rewards, std) == (rewards, std)

    def test_group_metrics_distinguish_prompt_and_sample_weighting(self):
        groups = [_versioned(0, 1), _versioned(1, 1, 2, 3), _versioned(2, 1, 2)]
        metrics = group_size_metrics(groups, attempt_size=GROUP_SIZE)
        assert metrics["rollout/groups/size_mean"] == 2 * GROUP_SIZE
        assert metrics["rollout/groups/merged_fraction"] == pytest.approx(2 / 3)
        assert metrics["rollout/groups/extra_sample_fraction"] == 0.5
        assert metrics["rollout/groups/attempts_3/count"] == 1
        assert metrics["rollout/groups/attempts_4/count"] == 0
        assert all(value == 0 for value in group_size_metrics([], attempt_size=4).values())

    def test_group_metrics_accept_nested_trajectories(self):
        metrics = group_size_metrics([[[Sample(), Sample()], [Sample()]]], attempt_size=2)
        assert metrics["rollout/groups/groups"] == 1
        assert metrics["rollout/groups/chain_attempts_mean"] == 1


def _versioned(chain_id: int, *versions: int, first_index: int = 0) -> list[Sample]:
    """A merged group: one attempt of GROUP_SIZE samples per version, the kept attempt last."""
    return [
        sample
        for attempt, version in enumerate(versions)
        for sample in _group(
            chain_id, [0.0] * GROUP_SIZE, first_index=first_index + attempt * GROUP_SIZE, version=version
        )
    ]


class TestDefaultDataBufferWithNeverGiveUp:
    @staticmethod
    def _buffer(**overrides) -> tuple[DefaultDataBuffer, _Requeued, list]:
        """``admissions`` counts the failed rounds NGU dropped, which wake the producer."""
        admissions = []
        ngu, requeued = _make_ngu(admission_changed=lambda: admissions.append(True), **overrides)
        buffer = DefaultDataBuffer(DataBufferConstructorInput(args=ngu._args, resampling_policy=ngu))
        return buffer, requeued, admissions

    async def test_a_rejected_group_is_requeued_and_the_metric_names_the_real_reason(self):
        buffer, requeued, _ = self._buffer()
        await buffer.put(_entry(_group(0, [0.0] * GROUP_SIZE, first_index=100)))

        assert [group[0].group_index for group in requeued] == [0]
        metrics = buffer.get_metrics()
        assert metrics["rollout/dynamic_filter/drop_zero_std"] == 1
        assert metrics["rollout/ngu/filter_retries"] == 1

    async def test_an_aborted_retry_is_requeued_instead_of_losing_the_chain(self):
        buffer, requeued, admissions = self._buffer()
        await buffer.put(_entry(_group(0, [0.0] * GROUP_SIZE, first_index=100)))
        retry = requeued.pop(0)
        for sample in retry:
            sample.status = Sample.Status.ABORTED
        await buffer.put(_entry(retry))

        assert [group[0].group_index for group in requeued] == [0]
        assert len(admissions) == 1  # the aborted copy itself is dropped

    async def test_consuming_a_merged_group_prunes_stale_attempts_instead_of_dropping_it(self):
        buffer, requeued, _ = self._buffer(max_weight_staleness=2)
        await buffer.put(_entry(_group(0, [0.0] * GROUP_SIZE, first_index=100, version=1)))
        await buffer.put(_entry(_answer(requeued.pop(0), [1.0, 0.0, 0.0, 0.0], version=4)))

        [entry] = await buffer.get(num_groups=1, current_version=5)

        assert [sample.index for sample in entry.group] == [0, 1, 2, 3]  # only the retry
        assert entry.group[0].metadata[REWARD_BASELINE_KEY]["count"] == 2 * GROUP_SIZE
        assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 0

    async def test_a_group_whose_kept_attempt_is_stale_becomes_a_failure_that_keeps_going(self):
        buffer, requeued, admissions = self._buffer(max_weight_staleness=2)
        await buffer.put(_entry(_group(0, [0.0, 0.0, 0.0, 1.0], first_index=100, version=1)))
        await buffer.put(_entry(_group(1, [1.0, 0.0, 0.0, 0.0], first_index=200, version=9)))

        [entry] = await buffer.get(num_groups=1, current_version=9)

        assert entry.group[0].group_index == 1  # the fresh group trains
        assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1
        assert len(admissions) == 1  # the stale group itself is dropped
        [retry] = requeued
        assert retry[0].group_index == 0


class _RetryingDataSource(FakeDataSource):
    def __init__(self, scripted):
        super().__init__(scripted)
        self.sample_index = 10_000

    def reserve_sample_indices(self, num_samples):
        indices = list(range(self.sample_index, self.sample_index + num_samples))
        self.sample_index += num_samples
        return indices


class TestFullyAsyncRolloutWithNeverGiveUp:
    async def test_a_rejected_prompt_is_retried_and_trains_as_one_merged_group(self, monkeypatch):
        """Group 1 first scores all zeros; its retry scores [1, 0], so both attempts train together."""
        args = make_args(
            rollout_batch_size=1,
            async_unused_samples_handler="never_give_up",
            dynamic_sampling_filter_path=f"{__name__}.nonzero_std",
            ngu_max_budget=None,
            ngu_solved_reward=1.0,
            async_max_concurrent_samples=2,  # one group in flight, so the retry is the next submission
        )
        attempts = []

        async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
            await asyncio.sleep(0)
            attempts.append([sample.index for sample in group])
            rewards = [0.0, 0.0] if len(attempts) == 1 else [1.0, 0.0]
            for sample, reward in zip(group, rewards, strict=True):
                sample.status, sample.reward = Sample.Status.COMPLETED, reward
            return group

        fn = make_fn(monkeypatch, args, _RetryingDataSource([make_group(1, reward=0)]), generate=generate)
        output = await asyncio.wait_for(fn(train_input(rollout_id=0)), timeout=5)
        await fn.dispose()

        [group] = output.samples
        assert {sample.group_index for sample in group} == {1}
        assert [sample.index for sample in group] == [10, 11, 10_000, 10_001]
        assert [sample.reward for sample in group] == [0.0, 0.0, 1.0, 0.0]
        assert output.metrics["rollout/groups/merged_fraction"] == 1.0
        assert output.metrics["rollout/ngu/accepted_retried_chains"] == 1


class TestValidateNeverGiveUpArgs:
    @staticmethod
    def _args(**overrides) -> Namespace:
        defaults = dict(
            async_unused_samples_handler="never_give_up",
            ngu_max_budget=None,
            dynamic_sampling_filter_path="miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
            fully_async=True,
            partial_rollout=False,
            use_dynamic_global_batch_size=True,
            custom_async_data_buffer_path=None,
            use_output_store=False,
            enable_sample_ownership_checker=False,
        )
        return Namespace(**{**defaults, **overrides})

    @pytest.mark.parametrize(
        "overrides",
        [dict(), dict(ngu_max_budget=1), dict(ngu_max_budget=4)],
    )
    def test_supported_setups_pass(self, overrides):
        _validate_never_give_up_args(self._args(**overrides))

    def test_other_handlers_skip_every_other_check(self):
        _validate_never_give_up_args(
            self._args(async_unused_samples_handler="drop", dynamic_sampling_filter_path=None)
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(ngu_max_budget=0),
            dict(dynamic_sampling_filter_path=None),
            dict(fully_async=False),
            dict(custom_async_data_buffer_path="my.Buffer"),
            dict(partial_rollout=True),
            dict(use_dynamic_global_batch_size=False),
            dict(use_output_store=True),
            dict(enable_sample_ownership_checker=True),
        ],
    )
    def test_unsupported_setups_are_rejected(self, overrides):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(**overrides))
