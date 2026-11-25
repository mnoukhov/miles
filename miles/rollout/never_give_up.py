"""Never Give Up (NGU): keep retrying an unsolved prompt instead of dropping it.

Selected with ``--async-unused-samples-handler never_give_up`` under ``--fully-async``. With
dynamic sampling, a prompt group with no learning signal (for example all rewards equal) is
dropped. NGU instead requeues the *same* prompt until it is solved or, with ``--ngu-max-budget M``,
until the prompt has had M attempts (at most ``M * n_samples_per_prompt`` samples). All attempts at one prompt form a *chain* that
shares the first attempt's ``group_index``.

:class:`NeverGiveUp` is a :class:`~miles.rollout.resampling.ResamplingPolicy`: each attempt is a
round of ``n_samples_per_prompt`` samples, retried sequentially. It is told the outcome of every
finished group:

- a dynamic filter rejection: the attempt is buffered on its chain and the prompt requeued, unless
  the chain already reached ``--ngu-solved-reward`` or has used its ``--ngu-max-budget`` attempts;
- aborted, restored or missing reward: nothing is learned, so a pending chain's retry is requeued
  as it was;
- stale: a failure that keeps going. Its rewards and completions are buffered for the baseline,
  the chain's best reward is reset, and the prompt is always requeued;
- kept: the group is merged in place with the chain's buffered attempts (up to
  ``--max-weight-staleness`` weight versions old), so it holds a multiple of
  ``n_samples_per_prompt`` samples, and the chain ends.

Merged groups make the number of samples per rollout vary, so NGU requires
``--use-dynamic-global-batch-size``: each rollout still drains ``rollout_batch_size`` groups, and
the trainer takes one optimizer step on however many samples they hold.

A merged group trains against its whole chain, stale-pruned attempts included: NGU stamps the chain's
reward totals under :data:`~miles.rollout.resampling.REWARD_BASELINE_KEY`, and before the GRPO step
:func:`~miles.rollout.resampling.baseline_rewards_and_std` shifts the rewards below the group max so
the group's mean is the chain's mean reward, and uses the chain's reward std.

A port of allenai/open-instruct#1861.
"""

import copy
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.filter_hub.common_filters import group_staleness, group_weight_version_stats
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.resampling import REWARD_BASELINE_KEY, GroupOutcome, NoResampling
from miles.utils.types import Sample


@dataclass
class PendingChain:
    """Everything buffered for one prompt while NGU keeps retrying it."""

    retry_template: Sample
    best_reward: float | None = None  # None right after a stale reset
    samples: list[Sample] = field(default_factory=list)  # buffered attempts, n_samples_per_prompt each
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    sample_count: int = 0
    rejected_attempts: int = 0  # filter rejections so far, counted against --ngu-max-budget


class NeverGiveUp(NoResampling):
    """The ``never_give_up`` resampling policy.

    Failed rounds are dropped as under ``drop``, and NGU decides whether their chain gets another
    round. A kept group is replaced in place by the merged chain, so the buffer trains on whatever
    ``entry.group`` holds afterwards. The pending chains live here, so one instance outlives a
    single rollout and is checkpointed with the rollout function.

    ``requeue`` resubmits a prompt group ahead of new prompts; ``reserve_sample_indices`` hands out
    fresh ``Sample.index`` values for the retries (the data source's counter).
    """

    def __init__(
        self,
        args,
        *,
        admission_changed: Callable[[], None],
        requeue: Callable[[list[Sample]], None],
        reserve_sample_indices: Callable[[int], list[int]],
    ):
        super().__init__(args, admission_changed=admission_changed)
        self._requeue = requeue
        self._reserve_sample_indices = reserve_sample_indices
        self._chains: dict[int, PendingChain] = {}
        self._metrics = Counter()

    # --------------------------- resampling policy ---------------------------

    def __call__(self, entry: DataBufferInput, outcome: GroupOutcome) -> None:
        super().__call__(entry, outcome)  # drop and record failed rounds
        if outcome == GroupOutcome.KEPT:
            self._merge_into_kept(entry.group)
        elif outcome == GroupOutcome.FILTERED:
            self._keep_going_or_drop(entry.group)
        elif outcome == GroupOutcome.STALE:
            self._keep_going_after_stale(entry.group)
        else:  # aborted, restored or missing reward: nothing was learned, so retry the chain as it was
            self._retry_pending_chain(entry.group)

    def prune_stale(self, entry: DataBufferInput, *, current_version: int | None) -> None:
        """A merged group goes stale by its earliest round first: drop the rounds that aged out while it
        waited, whose rewards stay in the chain baseline. If the kept round itself is stale, the group
        is still dropped, and :meth:`_keep_going_after_stale` requeues the prompt."""
        entry.group = prune_stale_attempts(
            entry.group,
            attempt_size=self._args.n_samples_per_prompt,
            current_version=current_version,
            max_staleness=self._args.max_weight_staleness,
        )

    def check_trained_group(self, group: list[Sample], *, n_samples_per_prompt: int) -> None:
        """A merged group holds a whole number of attempts at its prompt."""
        assert (
            group and len(group) % n_samples_per_prompt == 0
        ), f"expected a multiple of {n_samples_per_prompt} samples, got {len(group)}"

    def rollout_metrics(self, groups: list[list[Sample]]) -> dict[str, float]:
        return group_size_metrics(groups, attempt_size=self._args.n_samples_per_prompt)

    def collect_metrics(self) -> dict[str, float]:
        """Windowed retry decisions, including chains that never produce training data."""
        counts, self._metrics = self._metrics, Counter()
        names = (
            "accepted_chains",
            "accepted_retried_chains",
            "accepted_attempts",
            "filtered_attempts",
            "filter_retries",
            "stale_retries",
            "recovery_retries",
            "budget_exhausted_chains",
            "solved_filtered_chains",
        )
        metrics = {name: counts[name] for name in names}
        metrics["pending_chains"] = len(self._chains)
        metrics["accepted_retried_fraction"] = counts["accepted_retried_chains"] / max(1, counts["accepted_chains"])
        metrics["filter_retry_fraction"] = counts["filter_retries"] / max(1, counts["filtered_attempts"])
        metrics["accepted_attempts_mean"] = counts["accepted_attempts"] / max(1, counts["accepted_chains"])
        return {f"rollout/ngu/{name}": value for name, value in metrics.items()}

    def state_dict(self) -> Any:
        return {"chains": dict(self._chains)}

    def load_state_dict(self, state: Any) -> None:
        assert not self._chains, "restore never_give_up before any group finished"
        self._chains = dict(state["chains"])

    # ------------------------------- decisions -------------------------------

    def _merge_into_kept(self, group: list[Sample]) -> None:
        """Replace the kept group with the chain's fresh-enough buffered attempts plus itself, each
        sample carrying the chain-wide baseline."""
        pending = self._chains.pop(group[0].group_index, None) or PendingChain(retry_template=group[0])
        merged = prune_stale_attempts(
            pending.samples + group,
            attempt_size=self._args.n_samples_per_prompt,
            current_version=group_weight_version_stats(group).newest_version,
            max_staleness=self._args.max_weight_staleness,
        )
        rewards = _rewards(self._args, group)
        self._metrics["accepted_chains"] += 1
        self._metrics["accepted_retried_chains"] += pending.sample_count > 0
        self._metrics["accepted_attempts"] += (pending.sample_count + len(group)) // self._args.n_samples_per_prompt
        baseline = {
            "sum": pending.reward_sum + sum(rewards),
            "sq_sum": pending.reward_sq_sum + _sq_sum(rewards),
            "count": pending.sample_count + len(group),
        }
        for sample in merged:
            sample.group_index = group[0].group_index
            sample.metadata[REWARD_BASELINE_KEY] = dict(baseline)
        group[:] = merged

    def _retry_pending_chain(self, group: list[Sample]) -> None:
        if (pending := self._chains.get(group[0].group_index)) is not None:
            self._metrics["recovery_retries"] += 1
            self._requeue_chain(pending.retry_template)

    def _keep_going_after_stale(self, group: list[Sample]) -> None:
        """A stale accepted attempt failed to train, but its rewards still shape the baseline."""
        pending = self._chains.get(group[0].group_index) or PendingChain(retry_template=group[0])
        rewards = _rewards(self._args, group)
        baseline = group[0].metadata.get(REWARD_BASELINE_KEY) or {
            "sum": sum(rewards),
            "sq_sum": _sq_sum(rewards),
            "count": len(group),
        }
        pending.samples.extend(group)
        pending.reward_sum += baseline["sum"]
        pending.reward_sq_sum += baseline["sq_sum"]
        pending.sample_count += baseline["count"]
        pending.best_reward = None
        self._chains[group[0].group_index] = pending
        self._metrics["stale_retries"] += 1
        self._requeue_chain(pending.retry_template)

    def _keep_going_or_drop(self, group: list[Sample]) -> None:
        chain_id = group[0].group_index
        rewards = _rewards(self._args, group)
        pending = self._chains.pop(chain_id, None) or PendingChain(retry_template=group[0])
        best_reward = max(rewards) if pending.best_reward is None else max(pending.best_reward, *rewards)
        pending.rejected_attempts += 1
        self._metrics["filtered_attempts"] += 1
        if best_reward >= self._args.ngu_solved_reward:
            self._metrics["solved_filtered_chains"] += 1
            return
        if (max_budget := self._args.ngu_max_budget) is not None and pending.rejected_attempts >= max_budget:
            self._metrics["budget_exhausted_chains"] += 1
            return

        pending.samples.extend(group)
        pending.reward_sum += sum(rewards)
        pending.reward_sq_sum += _sq_sum(rewards)
        pending.sample_count += len(group)
        pending.best_reward = best_reward
        self._chains[chain_id] = pending
        self._metrics["filter_retries"] += 1
        self._requeue_chain(pending.retry_template)

    def _requeue_chain(self, retry_template: Sample) -> None:
        sample_indices = self._reserve_sample_indices(self._args.n_samples_per_prompt)
        self._requeue(make_retry_group(retry_template, sample_indices=sample_indices))


def _rewards(args, group: list[Sample]) -> list[float]:
    return [sample.get_reward_value(args) for sample in group]


def _sq_sum(rewards: list[float]) -> float:
    return sum(reward * reward for reward in rewards)


def _split_attempts(group: list[Sample], *, attempt_size: int) -> list[list[Sample]]:
    return [group[start : start + attempt_size] for start in range(0, len(group), attempt_size)]


def make_retry_group(retry_template: Sample, *, sample_indices: list[int]) -> list[Sample]:
    """Fresh copies of a chain's prompt. They keep its ``group_index`` so the retry stays on the chain."""
    group = []
    for index in sample_indices:
        sample = copy.deepcopy(retry_template)
        sample.reset_for_retry()
        sample.metadata.pop(REWARD_BASELINE_KEY, None)
        sample.index = index
        group.append(sample)
    return group


def prune_stale_attempts(
    group: list[Sample], *, attempt_size: int, current_version: int | None, max_staleness: int | None
) -> list[Sample]:
    """Drop the buffered attempts of a merged group that became more than ``max_staleness`` weight
    versions old while it waited to be trained on. The kept attempt (the last one) always stays,
    and the chain-wide baseline on the samples still counts the dropped attempts' rewards."""
    if max_staleness is None or len(group) <= attempt_size:
        return group

    def is_stale(attempt: list[Sample]) -> bool:
        staleness = group_staleness(attempt, current_version)
        return staleness is not None and staleness > max_staleness

    attempts = _split_attempts(group, attempt_size=attempt_size)
    kept = [attempt for attempt in attempts[:-1] if not is_stale(attempt)]
    return [sample for attempt in [*kept, attempts[-1]] for sample in attempt]


def group_size_metrics(groups, *, attempt_size: int) -> dict[str, float]:
    """Bounded group-size statistics of a rollout's training groups."""
    sizes = [len(group) for group in groups]
    counts = Counter()
    for size in sizes:
        attempts = size / attempt_size
        counts[str(int(attempts)) if attempts in (1, 2, 3, 4) else "other"] += 1
    count, samples = len(sizes), sum(sizes)
    chain_sizes = [
        (next(iter_samples(group)).metadata.get(REWARD_BASELINE_KEY) or {"count": len(group)})["count"]
        for group in groups
        if group
    ]
    metrics = {
        "groups": count,
        "samples": samples,
        "size_mean": samples / max(1, count),
        "size_min": min(sizes, default=0),
        "size_max": max(sizes, default=0),
        "merged_fraction": sum(size > attempt_size for size in sizes) / max(1, count),
        "extra_sample_fraction": sum(max(0, size - attempt_size) for size in sizes) / max(1, samples),
        "chain_retried_fraction": sum(size > attempt_size for size in chain_sizes) / max(1, count),
        "chain_attempts_mean": sum(chain_sizes) / (attempt_size * max(1, count)),
    }
    for label in ("1", "2", "3", "4", "other"):
        metrics[f"attempts_{label}/count"] = counts[label]
        metrics[f"attempts_{label}/fraction"] = counts[label] / max(1, count)
    return {f"rollout/groups/{name}": value for name, value in metrics.items()}
