from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.never_give_up import (
    NGU_ATTEMPT_COUNT_KEY,
    NGU_BASELINE_REWARD_SUM_KEY,
    NGU_BASELINE_SAMPLE_COUNT_KEY,
    NeverGiveUpAction,
    NeverGiveUpConfig,
    NeverGiveUpState,
    anchor_positive_advantages,
    decide_never_give_up,
    make_retry_group,
    should_accept_attempt,
)
from miles.utils.arguments import _validate_never_give_up_args
from miles.utils.types import Sample

GROUP_SIZE = 4


def _config(**overrides) -> NeverGiveUpConfig:
    defaults = dict(probability=1.0, max_pending_age=4, keep_pending_completions=True, solved_reward=1.0)
    return NeverGiveUpConfig(**{**defaults, **overrides})


def _group(chain_id: int, rewards: list[float], first_index: int = 0) -> list[Sample]:
    return [
        Sample(
            group_index=chain_id,
            index=first_index + i,
            prompt="prompt",
            response="response",
            response_length=1,
            reward=reward,
            status=Sample.Status.COMPLETED,
        )
        for i, reward in enumerate(rewards)
    ]


def _decide(state, group, *, filter_keep, rollout_id=0, config=None, rng_value=0.0):
    rng = random.Random()
    rng.random = lambda: rng_value
    return decide_never_give_up(
        config=config or _config(),
        state=state,
        group=group,
        rewards=[sample.reward for sample in group],
        filter_keep=filter_keep,
        prompt_template=Sample(group_index=group[0].group_index, prompt="prompt"),
        rollout_id=rollout_id,
        rng=rng,
    )


class TestShouldAcceptAttempt:
    def test_first_attempt_follows_the_dynamic_filter(self):
        assert should_accept_attempt([0.0, 1.0], filter_keep=True, best_reward=None)
        assert not should_accept_attempt([0.0, 0.0], filter_keep=False, best_reward=None)

    def test_later_attempt_must_strictly_beat_the_best_reward(self):
        assert not should_accept_attempt([0.0, 0.5], filter_keep=True, best_reward=0.5)
        assert should_accept_attempt([0.0, 0.6], filter_keep=False, best_reward=0.5)


class TestDecideNeverGiveUp:
    def test_group_with_signal_is_kept_with_its_own_baseline(self):
        state = NeverGiveUpState()
        decision = _decide(state, _group(7, [0.0, 1.0, 0.0, 1.0]), filter_keep=True)

        assert decision.action == NeverGiveUpAction.KEEP
        assert len(decision.group) == GROUP_SIZE
        assert decision.group[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 2.0
        assert decision.group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == GROUP_SIZE
        assert decision.group[0].metadata[NGU_ATTEMPT_COUNT_KEY] == 1
        assert state.chains == {}

    def test_unsolved_group_without_signal_is_requeued_and_buffered(self):
        state = NeverGiveUpState()
        decision = _decide(state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False)

        assert decision.action == NeverGiveUpAction.REQUEUE
        chain = state.chains[7]
        assert chain.best_reward == 0.0
        assert chain.sample_count == GROUP_SIZE
        assert chain.attempt_count == 1
        assert len(chain.attempts) == 1

    def test_solved_group_is_dropped_not_requeued(self):
        state = NeverGiveUpState()
        decision = _decide(state, _group(7, [1.0] * GROUP_SIZE), filter_keep=False)

        assert decision.action == NeverGiveUpAction.DROP
        assert state.chains == {}

    def test_failed_coin_flip_drops_the_chain_and_its_buffer(self):
        state = NeverGiveUpState()
        _decide(state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False)
        decision = _decide(
            state,
            _group(7, [0.0] * GROUP_SIZE, first_index=4),
            filter_keep=False,
            rng_value=0.99,
            config=_config(probability=0.5),
        )

        assert decision.action == NeverGiveUpAction.DROP
        assert len(decision.dropped_attempts) == 1
        assert state.chains == {}

    def test_improved_attempt_merges_the_chain_with_a_chain_wide_baseline(self):
        state = NeverGiveUpState()
        _decide(state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False, rollout_id=0)
        _decide(state, _group(7, [0.0] * GROUP_SIZE, first_index=4), filter_keep=False, rollout_id=1)
        decision = _decide(state, _group(7, [0.0, 0.0, 0.0, 1.0], first_index=8), filter_keep=True, rollout_id=2)

        assert decision.action == NeverGiveUpAction.KEEP
        assert [sample.index for sample in decision.group] == list(range(12))
        assert {sample.group_index for sample in decision.group} == {7}
        assert decision.group[0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 1.0
        assert decision.group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 12
        assert decision.group[-1].metadata[NGU_ATTEMPT_COUNT_KEY] == 3
        assert state.chains == {}

    def test_stale_attempts_leave_the_merge_but_stay_in_the_baseline(self):
        state = NeverGiveUpState()
        _decide(state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False, rollout_id=0)
        decision = _decide(
            state,
            _group(7, [0.0, 1.0, 0.0, 0.0], first_index=4),
            filter_keep=True,
            rollout_id=5,
            config=_config(max_pending_age=2),
        )

        assert [sample.index for sample in decision.group] == [4, 5, 6, 7]
        assert decision.group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 8

    def test_without_pending_completions_only_the_rewards_are_kept(self):
        state = NeverGiveUpState()
        config = _config(keep_pending_completions=False)
        _decide(state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False, config=config)
        decision = _decide(state, _group(7, [1.0, 0.0, 0.0, 0.0], first_index=4), filter_keep=True, config=config)

        assert len(decision.group) == GROUP_SIZE
        assert decision.group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 8


class TestAnchorPositiveAdvantages:
    def test_positives_are_kept_and_the_group_sums_to_zero(self):
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        advantages = rewards - 0.125  # chain-wide baseline below the group mean
        out = anchor_positive_advantages(advantages, rewards)

        assert out[3] == pytest.approx(0.875)
        assert float(out.sum()) == pytest.approx(0.0, abs=1e-6)
        assert torch.allclose(out[:3], torch.full((3,), -0.875 / 3))

    def test_all_equal_rewards_are_left_alone(self):
        advantages = torch.tensor([0.5, 0.5])
        assert torch.equal(anchor_positive_advantages(advantages, torch.tensor([1.0, 1.0])), advantages)


def _data_source_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=False,
        n_samples_per_prompt=GROUP_SIZE,
        buffer_filter_path=None,
        rollout_seed=0,
        never_give_up=1.0,
        save=None,
        load=None,
        rollout_shuffle=False,
    )
    return Namespace(**{**defaults, **overrides})


class TestRequeuePrompt:
    def test_retry_keeps_the_chain_and_gets_fresh_indices_ahead_of_new_data(self):
        source = RolloutDataSourceWithBuffer(_data_source_args())
        [first] = source.get_samples(1)
        template = first[0]

        source.requeue_prompt(template)
        [retry, fresh] = source.get_samples(2)

        assert [sample.group_index for sample in retry] == [template.group_index] * GROUP_SIZE
        assert [sample.index for sample in retry] == [4, 5, 6, 7]
        assert all(sample.status == Sample.Status.PENDING and sample.reward is None for sample in retry)
        assert fresh[0].group_index == template.group_index + 1

    def test_make_retry_group_clears_generated_outputs(self):
        template = _group(3, [1.0])[0]
        [retry] = make_retry_group(template, sample_indices=[9])

        assert (retry.response, retry.reward, retry.index, retry.group_index) == ("", None, 9, 3)


class TestGenerateRolloutWithNeverGiveUp:
    """Drives generate_rollout_async with scripted rewards: chain 0 has no signal on its first
    attempt and solves one sample on its retry; every other prompt has signal right away."""

    @staticmethod
    def _run(monkeypatch, source, rollout_id=0):
        args = Namespace(
            rollout_global_dataset=True,
            rollout_batch_size=2,
            n_samples_per_prompt=GROUP_SIZE,
            over_sampling_batch_size=1,
            rollout_submission_granularity=None,
            dynamic_sampling_filter_path="zero_std",
            reward_key=None,
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            never_give_up=1.0,
            ngu_max_pending_age=4,
            ngu_keep_pending_completions=True,
            ngu_solved_reward=1.0,
            partial_rollout=False,
        )
        attempts_of_chain: dict[int, int] = {}

        async def generate(group):
            chain_id = group[0].group_index
            attempt = attempts_of_chain.get(chain_id, 0)
            attempts_of_chain[chain_id] = attempt + 1
            rewards = [0.0] * GROUP_SIZE if chain_id == 0 and attempt == 0 else [1.0, 0.0, 0.0, 0.0]
            for sample, reward in zip(group, rewards, strict=True):
                sample.reward = reward
                sample.status = Sample.Status.COMPLETED
            return group

        async def fake_abort(_state, pendings, _rollout_id):
            for task in pendings:
                task.cancel()
            await asyncio.gather(*pendings, return_exceptions=True)
            return []

        async def noop(*_args, **_kwargs):
            return None

        def zero_std_filter(_args, group, **_kwargs):
            return len({sample.reward for sample in group}) > 1

        state = Namespace(args=args, sampling_params={}, aborted=False, reset=lambda: None)
        monkeypatch.setattr(
            train,
            "submit_generate_tasks",
            lambda _state, samples, sample_done_callback=None: [asyncio.ensure_future(generate(g)) for g in samples],
        )
        monkeypatch.setattr(train, "abort", fake_abort)
        monkeypatch.setattr(train, "load_function", lambda path: zero_std_filter if path == "zero_std" else None)
        monkeypatch.setattr(train.dumper_utils, "configure_sglang", noop)
        monkeypatch.setattr(train, "recompute_samples_rollout_logprobs_via_prefill", noop)
        return asyncio.run(
            train.generate_rollout_async(state, rollout_id, source.get_samples, never_give_up_source=source)
        )

    def test_retried_prompt_trains_as_one_merged_group(self, monkeypatch):
        source = RolloutDataSourceWithBuffer(_data_source_args())

        output, _ = self._run(monkeypatch, source)

        groups = {group[0].group_index: group for group in output.samples}
        assert len(groups[0]) == 2 * GROUP_SIZE
        assert groups[0][0].metadata[NGU_BASELINE_REWARD_SUM_KEY] == 1.0
        assert groups[0][0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 2 * GROUP_SIZE
        assert len({sample.index for sample in groups[0]}) == 2 * GROUP_SIZE
        assert output.metrics["rollout/never_give_up/requeue"] == 1
        assert source.never_give_up_state.chains == {}

    def test_save_and_load_round_trip_pending_chains(self, tmp_path: Path):
        source = RolloutDataSourceWithBuffer(_data_source_args(rollout_global_dataset=False))
        _decide(source.never_give_up_state, _group(7, [0.0] * GROUP_SIZE), filter_keep=False)
        source.requeue_prompt(Sample(group_index=7, prompt="prompt"))
        source.args.rollout_global_dataset = True
        source.args.save = source.args.load = str(tmp_path)
        source.dataset = None
        source.save(rollout_id=3)

        restored = RolloutDataSourceWithBuffer(_data_source_args())
        restored.args.rollout_global_dataset = True
        restored.args.load = str(tmp_path)
        restored.load(rollout_id=3)

        assert list(restored.never_give_up_state.chains) == [7]
        assert len(restored.buffer) == 1


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
        )
        return Namespace(**{**defaults, **overrides})

    def test_supported_setup_passes(self):
        _validate_never_give_up_args(self._args())

    def test_disabled_skips_every_other_check(self):
        _validate_never_give_up_args(self._args(never_give_up=0.0, dynamic_sampling_filter_path=None))

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(never_give_up=1.5),
            dict(dynamic_sampling_filter_path=None),
            dict(fully_async=True),
            dict(partial_rollout=True),
            dict(use_dynamic_global_batch_size=False),
        ],
    )
    def test_unsupported_setups_are_rejected(self, overrides):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(**overrides))
