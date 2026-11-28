"""Never Give Up on a token budget: continue an unsolved group's cut-off responses instead of resampling.

Selected with ``--async-unused-samples-handler never_give_up --ngu-budget-unit tokens`` under
``--fully-async``. Sample-budget NGU (:mod:`miles.rollout.never_give_up`) answers an unsolved prompt
with another round of fresh responses; here the budget is response tokens. Every group starts with
each response capped at ``--ngu-response-len-per-round`` tokens. When the dynamic sampling filter
rejects a group because no response is correct, its responses that ran out of tokens continue from
where they stopped, another ``--ngu-response-len-per-round`` tokens each, up to
``--rollout-max-response-len``. Responses that finished keep their text and reward, so the group is
resubmitted as is and only the cut-off responses generate. A group that solves the prompt, or whose
responses all finished or used the whole budget, is treated as plain dynamic sampling treats it.

A continued response is the same trajectory, sampled in two or more calls, so the group still holds
``--n-samples-per-prompt`` samples and trains as an ordinary GRPO group. Its tokens may come from
several weight versions; staleness counts from the oldest, as for any response.

Failed rounds (aborted, stale, or restored from a checkpoint) are retried from scratch, as under
``retry``.
"""

import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.resampling import FAILED_OUTCOMES, GroupOutcome, NoResampling
from miles.utils.types import Sample

# Number of continuations a response has had, on ``Sample.metadata``.
TOKEN_ROUNDS_KEY = "ngu_token_rounds"


class TokenNeverGiveUp(NoResampling):
    """The ``never_give_up`` resampling policy with ``--ngu-budget-unit tokens``."""

    def __init__(self, args, *, admission_changed: Callable[[], None], requeue: Callable[[list[Sample]], None]):
        super().__init__(args, admission_changed=admission_changed)
        self._requeue = requeue
        self._per_round = args.ngu_response_len_per_round
        self._max_len = args.rollout_max_response_len
        self._metrics = Counter()

    def prepare_group(self, group: list[Sample]) -> None:
        """A fresh round stops its responses at the per-round budget; a continuation keeps its own caps."""
        for sample in group:
            if sample.response_length == 0:
                sample.max_response_len = self._per_round
                sample.metadata.pop(TOKEN_ROUNDS_KEY, None)

    def __call__(self, entry: DataBufferInput, outcome: GroupOutcome) -> None:
        super().__call__(entry, outcome)  # log failed rounds as dropped
        if outcome == GroupOutcome.KEPT:
            self._metrics["kept_groups"] += 1
            self._metrics["kept_continued_groups"] += _rounds(entry.group) > 0
        elif outcome == GroupOutcome.FILTERED:
            self._continue_or_drop(entry.group)
        elif outcome in FAILED_OUTCOMES:
            self._metrics["failed_retries"] += 1
            group = copy.deepcopy(entry.prompt_group)
            for sample in group:
                sample.reset_for_retry()
            self._requeue(group)

    def collect_metrics(self) -> dict[str, float]:
        counts, self._metrics = self._metrics, Counter()
        names = (
            "kept_groups",
            "kept_continued_groups",
            "continued_groups",
            "continued_responses",
            "solved_filtered_groups",
            "exhausted_groups",
            "finished_filtered_groups",
            "failed_retries",
        )
        metrics = {name: counts[name] for name in names}
        metrics["continued_responses_per_group"] = counts["continued_responses"] / max(1, counts["continued_groups"])
        return {f"rollout/ngu_tokens/{name}": value for name, value in metrics.items()}

    def rollout_metrics(self, groups: list[list[Sample]]) -> dict[str, float]:
        samples = [sample for group in groups for sample in group]
        continued = [sample for sample in samples if sample.metadata.get(TOKEN_ROUNDS_KEY, 0) > 0]
        return {
            "rollout/ngu_tokens/trained_continued_fraction": len(continued) / max(1, len(samples)),
            "rollout/ngu_tokens/trained_continued_groups": sum(_rounds(group) > 0 for group in groups),
            "rollout/ngu_tokens/trained_continued_response_len_mean": (
                sum(sample.response_length for sample in continued) / max(1, len(continued))
            ),
        }

    def _continue_or_drop(self, group: list[Sample]) -> None:
        rewards = [sample.get_reward_value(self._args) for sample in group]
        if max(rewards) >= self._args.ngu_solved_reward:
            self._metrics["solved_filtered_groups"] += 1
            return
        cut_off = [
            i
            for i, sample in enumerate(group)
            if sample.status == Sample.Status.TRUNCATED and sample.response_length < self._max_len
        ]
        if not cut_off:
            truncated = any(sample.status == Sample.Status.TRUNCATED for sample in group)
            self._metrics["exhausted_groups" if truncated else "finished_filtered_groups"] += 1
            return

        continuation = copy.deepcopy(group)
        for i in cut_off:
            sample = continuation[i]
            sample.status = Sample.Status.ABORTED  # resumable: generation continues from sample.tokens
            sample.reward = None
            sample.max_response_len = min(sample.response_length + self._per_round, self._max_len)
            sample.metadata[TOKEN_ROUNDS_KEY] = sample.metadata.get(TOKEN_ROUNDS_KEY, 0) + 1
        self._metrics["continued_groups"] += 1
        self._metrics["continued_responses"] += len(cut_off)
        self._requeue(continuation)

    def state_dict(self) -> Any:
        return None  # continuations waiting to run live in the rollout's retry buffer


def _rounds(group: list[Sample]) -> int:
    return max((sample.metadata.get(TOKEN_ROUNDS_KEY, 0) for sample in group), default=0)
