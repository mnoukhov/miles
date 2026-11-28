from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["rollout"])

import asyncio
from argparse import Namespace

import pytest
from tests.fast.fixtures.generation_fixtures import generation_env, make_sample
from tests.fast.rollout.test_fully_async_rollout import N_SAMPLES_PER_PROMPT, make_args, make_fn, train_input

from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.never_give_up_tokens import TOKEN_ROUNDS_KEY, TokenNeverGiveUp
from miles.rollout.resampling import GroupOutcome, make_resampling_policy
from miles.utils.arguments import _validate_never_give_up_args
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample

_ = generation_env


@pytest.fixture
def variant():
    """generation_env builds its args for this generate function."""
    return "single_turn"

GROUP_SIZE = 4
PER_ROUND = 8
MAX_LEN = 16


def _args(**overrides) -> Namespace:
    defaults = dict(
        n_samples_per_prompt=GROUP_SIZE,
        reward_key=None,
        async_unused_samples_handler="never_give_up",
        ngu_budget_unit="tokens",
        ngu_response_len_per_round=PER_ROUND,
        ngu_solved_reward=1.0,
        rollout_max_response_len=MAX_LEN,
        enable_sample_ownership_checker=False,
    )
    return Namespace(**{**defaults, **overrides})


def _make(**overrides) -> tuple[TokenNeverGiveUp, list[list[Sample]]]:
    requeued: list[list[Sample]] = []
    return TokenNeverGiveUp(_args(**overrides), admission_changed=lambda: None, requeue=requeued.append), requeued


def _sample(i: int, *, reward: float, length: int, status: Sample.Status) -> Sample:
    return Sample(
        group_index=7,
        index=i,
        prompt="prompt",
        tokens=[1, 2] + [100 + i] * length,
        response="x" * length,
        response_length=length,
        reward=reward,
        status=status,
        max_response_len=PER_ROUND,
    )


def _group(*specs: tuple[float, int, Sample.Status]) -> list[Sample]:
    return [_sample(i, reward=r, length=n, status=s) for i, (r, n, s) in enumerate(specs)]


DONE, CUT = Sample.Status.COMPLETED, Sample.Status.TRUNCATED


def _filtered(policy: TokenNeverGiveUp, group: list[Sample]) -> None:
    policy(DataBufferInput(prompt_group=group, group=group), GroupOutcome.FILTERED)


class TestTokenNeverGiveUp:
    def test_fresh_round_is_capped_at_the_per_round_budget(self):
        policy, _ = _make()
        group = [Sample(index=i, prompt="p", metadata={TOKEN_ROUNDS_KEY: 3}) for i in range(GROUP_SIZE)]
        policy.prepare_group(group)
        assert [s.max_response_len for s in group] == [PER_ROUND] * GROUP_SIZE
        assert all(TOKEN_ROUNDS_KEY not in s.metadata for s in group)

    def test_continuation_keeps_its_caps(self):
        policy, _ = _make()
        sample = _sample(0, reward=0.0, length=PER_ROUND, status=Sample.Status.ABORTED)
        sample.max_response_len = MAX_LEN
        policy.prepare_group([sample])
        assert sample.max_response_len == MAX_LEN

    def test_unsolved_group_continues_only_its_cut_off_responses(self):
        policy, requeued = _make()
        group = _group((0.0, 3, DONE), (0.0, PER_ROUND, CUT), (0.0, 5, DONE), (0.0, PER_ROUND, CUT))
        _filtered(policy, group)

        [continuation] = requeued
        assert continuation is not group and [s.index for s in continuation] == [0, 1, 2, 3]
        for i in (0, 2):  # finished: resubmitted untouched, so generation skips them
            assert continuation[i].status == DONE and continuation[i].reward == 0.0
            assert continuation[i].max_response_len == PER_ROUND
        for i in (1, 3):  # cut off: resumable, with the next round's budget
            sample = continuation[i]
            assert sample.status == Sample.Status.ABORTED and sample.reward is None
            assert sample.max_response_len == MAX_LEN and sample.response_length == PER_ROUND
            assert sample.tokens == group[i].tokens
            assert sample.metadata[TOKEN_ROUNDS_KEY] == 1
        assert group[1].status == CUT  # the filtered round itself is left as it was

        metrics = policy.collect_metrics()
        assert metrics["rollout/ngu_tokens/continued_groups"] == 1
        assert metrics["rollout/ngu_tokens/continued_responses"] == 2

    def test_continuation_never_exceeds_the_total_budget(self):
        policy, requeued = _make(ngu_response_len_per_round=6, rollout_max_response_len=MAX_LEN)
        group = _group((0.0, 12, CUT), (0.0, 3, DONE), (0.0, 3, DONE), (0.0, 3, DONE))
        _filtered(policy, group)
        assert requeued[0][0].max_response_len == MAX_LEN

    @pytest.mark.parametrize(
        "specs,metric",
        [
            # solved: every response correct, nothing to learn
            ([(1.0, 3, DONE)] * GROUP_SIZE, "solved_filtered_groups"),
            # wrong but finished: more tokens would not change the answers
            ([(0.0, 3, DONE)] * GROUP_SIZE, "finished_filtered_groups"),
            # cut off at the whole budget already
            ([(0.0, MAX_LEN, CUT)] + [(0.0, 3, DONE)] * 3, "exhausted_groups"),
        ],
    )
    def test_groups_with_nothing_to_continue_are_dropped(self, specs, metric):
        policy, requeued = _make()
        _filtered(policy, _group(*specs))
        assert requeued == []
        assert policy.collect_metrics()[f"rollout/ngu_tokens/{metric}"] == 1

    @pytest.mark.parametrize("outcome", [GroupOutcome.ABORTED, GroupOutcome.STALE, GroupOutcome.RESTORED])
    def test_failed_round_restarts_from_scratch(self, outcome):
        policy, requeued = _make()
        group = _group((0.0, PER_ROUND, CUT), (1.0, 3, DONE), (0.0, 3, DONE), (0.0, 3, DONE))
        policy(DataBufferInput(prompt_group=group, group=group), outcome)
        [retry] = requeued
        assert all(s.response_length == 0 and s.tokens == [] and s.reward is None for s in retry)
        assert group[0].response_length == PER_ROUND  # the original is not mutated

    def test_kept_groups_count_continued_ones(self):
        policy, _ = _make()
        plain = _group((1.0, 3, DONE), (0.0, 3, DONE), (0.0, 3, DONE), (0.0, 3, DONE))
        continued = _group((1.0, 12, DONE), (0.0, 3, DONE), (0.0, 3, DONE), (0.0, 3, DONE))
        continued[0].metadata[TOKEN_ROUNDS_KEY] = 1
        for group in (plain, continued):
            policy(DataBufferInput(prompt_group=group, group=group), GroupOutcome.KEPT)
        metrics = policy.collect_metrics()
        assert metrics["rollout/ngu_tokens/kept_groups"] == 2
        assert metrics["rollout/ngu_tokens/kept_continued_groups"] == 1
        rollout = policy.rollout_metrics([plain, continued])
        assert rollout["rollout/ngu_tokens/trained_continued_groups"] == 1
        assert rollout["rollout/ngu_tokens/trained_continued_response_len_mean"] == 12

    def test_factory_picks_the_token_policy(self):
        policy = make_resampling_policy(
            _args(), data_source=None, requeue=lambda group: None, admission_changed=lambda: None
        )
        assert isinstance(policy, TokenNeverGiveUp)


class TestResumeThroughGenerate:
    def test_cut_off_response_resumes_from_its_tokens_with_the_next_rounds_budget(self, generation_env):
        """The real generate path: round one stops at the per-round cap, the continuation resends the
        prompt plus the partial response and asks only for the remaining tokens."""
        server = generation_env.mock_server
        state = GenerateState(generation_env.args)
        sampling_params = {"max_new_tokens": MAX_LEN, "temperature": 0.7}
        policy, requeued = _make(ngu_response_len_per_round=2)

        sample = make_sample(prompt="What is 1+7?")
        sample.index = 0
        policy.prepare_group([sample])
        server.process_fn = lambda _: ProcessResult(text="\\boxed", finish_reason="length")
        server.request_log.clear()
        first = run(inference_rollout_common.generate_and_rm(state, sample, dict(sampling_params)))
        assert server.request_log[0]["sampling_params"]["max_new_tokens"] == 2
        assert first.status == Sample.Status.TRUNCATED and first.response_length == 2
        first_tokens, first_log_probs = list(first.tokens), list(first.rollout_log_probs)

        first.reward = 0.0
        others = [_sample(i, reward=0.0, length=3, status=DONE) for i in range(1, GROUP_SIZE)]
        _filtered(policy, [first, *others])
        [continuation] = requeued
        resumed = continuation[0]

        server.process_fn = lambda _: ProcessResult(text="{8}", finish_reason="stop")
        server.request_log.clear()
        done = run(inference_rollout_common.generate_and_rm(state, resumed, dict(sampling_params)))
        [request] = server.request_log
        assert request["input_ids"] == first_tokens
        assert request["sampling_params"]["max_new_tokens"] == 2 + 2 - 2  # cap 4 minus the 2 already generated
        assert done.status == Sample.Status.COMPLETED
        assert done.response == "\\boxed{8}" and done.response_length == 2 + 3
        assert done.tokens[: len(first_tokens)] == first_tokens
        assert done.rollout_log_probs[:2] == first_log_probs and len(done.rollout_log_probs) == 5
        assert len(done.weight_versions) == 2

        # finished responses in the group are returned untouched, without another request
        server.request_log.clear()
        untouched = run(inference_rollout_common.generate_and_rm(state, continuation[1], dict(sampling_params)))
        assert server.request_log == [] and untouched.reward == 0.0


class FreshPromptSource:
    """Serves unanswered prompts, as the real data source does: nothing generated yet, no reward."""

    def __init__(self):
        self.next_group_index = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.next_group_index += 1
        g = self.next_group_index
        return [[Sample(group_index=g, index=g * 10 + i, prompt=f"prompt {g}") for i in range(N_SAMPLES_PER_PROMPT)]]


def _zero_std(_args, group, **_kwargs):
    keep = len({sample.reward for sample in group}) > 1
    return FilterOutput(keep=keep, reason=None if keep else "zero_std")


async def test_fully_async_continues_an_unsolved_group_and_trains_it(monkeypatch):
    """Through the producer and data buffer: the first round of every group runs out of tokens and is
    all wrong; its continuation finishes, one response correct, and the group trains."""
    args = make_args(
        rollout_batch_size=1,
        async_unused_samples_handler="never_give_up",
        ngu_budget_unit="tokens",
        ngu_response_len_per_round=PER_ROUND,
        ngu_solved_reward=1.0,
        rollout_max_response_len=MAX_LEN,
        dynamic_sampling_filter_path=f"{__name__}._zero_std",
    )
    seen_caps: list[list[int | None]] = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        seen_caps.append([s.max_response_len for s in group])
        for i, sample in enumerate(group):
            if sample.status not in (Sample.Status.PENDING, Sample.Status.ABORTED):
                continue
            if sample.response_length == 0:  # round one: out of tokens, wrong
                sample.response_length, sample.status, sample.reward = PER_ROUND, Sample.Status.TRUNCATED, 0.0
            else:  # continuation: finishes, the first response correct
                sample.response_length += 2
                sample.status, sample.reward = Sample.Status.COMPLETED, float(i == 0)
        return group

    fn = make_fn(monkeypatch, args, FreshPromptSource(), generate=generate)
    try:
        output = await fn(train_input(rollout_id=0))
    finally:
        await fn.dispose()

    [group] = output.samples
    assert [s.reward for s in group] == [1.0] + [0.0] * (len(group) - 1)
    assert all(s.response_length == PER_ROUND + 2 for s in group)
    assert seen_caps[0] == [PER_ROUND] * len(group)
    assert [MAX_LEN] * len(group) in seen_caps[1:]


class TestValidateTokenBudget:
    @staticmethod
    def _args(**overrides) -> Namespace:
        defaults = dict(
            async_unused_samples_handler="never_give_up",
            ngu_max_budget=None,
            ngu_budget_unit="tokens",
            ngu_response_len_per_round=8192,
            rollout_max_response_len=16384,
            dynamic_sampling_filter_path="miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
            fully_async=True,
            partial_rollout=False,
            use_dynamic_global_batch_size=False,
            custom_async_data_buffer_path=None,
            custom_generate_function_path=None,
            use_output_store=False,
            enable_sample_ownership_checker=False,
        )
        return Namespace(**{**defaults, **overrides})

    def test_fixed_global_batch_is_fine(self):
        _validate_never_give_up_args(self._args())

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(ngu_response_len_per_round=None),
            dict(ngu_response_len_per_round=16384),
            dict(ngu_response_len_per_round=0),
            dict(ngu_max_budget=4),
            dict(custom_generate_function_path="my.generate"),
            dict(partial_rollout=True),
            dict(use_output_store=True),
        ],
    )
    def test_unsupported_setups_are_rejected(self, overrides):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(**overrides))

    def test_per_round_length_needs_the_token_unit(self):
        with pytest.raises(AssertionError):
            _validate_never_give_up_args(self._args(ngu_budget_unit="samples", use_dynamic_global_batch_size=True))
