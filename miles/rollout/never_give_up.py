"""Never Give Up (NGU): keep retrying an unsolved prompt instead of dropping it.

With dynamic sampling, a prompt group with no learning signal (for example all
rewards equal) is dropped. NGU instead requeues the *same* prompt with probability
``--never-give-up``, through the data source's buffer like any other resubmitted
group. All attempts at one prompt form a *chain* that shares the first attempt's
``group_index``.

:class:`NeverGiveUpFilter` wraps ``--dynamic-sampling-filter-path``:

- Before any attempt is buffered, the wrapped filter decides as usual.
- After that, an attempt is kept only when its max reward is strictly higher than
  the best reward the chain has seen so far.

A kept attempt is merged in place with the chain's buffered attempts (up to
``--max-weight-staleness`` weight versions old), so the group holds a multiple of
``n_samples_per_prompt`` samples. Its advantage baseline is then the mean reward
over *every* attempt in the chain, and the max-reward samples are anchored while the
others are rescaled so the group sums to zero (:func:`anchor_positive_advantages`).

A port of allenai/open-instruct#1861.
"""

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from miles.rollout.filter_hub.base_types import FilterOutput, call_dynamic_filter
from miles.rollout.filter_hub.common_filters import group_weight_version_stats
from miles.utils.types import Sample

if TYPE_CHECKING:  # a runtime import would be circular through miles.rollout.base_types
    from miles.rollout.data_source import RolloutDataSourceWithBuffer

NGU_BASELINE_REWARD_SUM_KEY = "ngu_baseline_reward_sum"
NGU_BASELINE_SAMPLE_COUNT_KEY = "ngu_baseline_sample_count"
NGU_ATTEMPT_COUNT_KEY = "ngu_attempt_count"


@dataclass
class PendingChain:
    """Everything buffered for one prompt while NGU keeps retrying it."""

    best_reward: float
    retry_template: Sample
    # Buffered attempts with the weight version each was generated at.
    attempts: list[tuple[int | None, list[Sample]]] = field(default_factory=list)
    sample_count: int = 0
    reward_sum: float = 0.0
    attempt_count: int = 0


class NeverGiveUpFilter:
    """A dynamic sampling filter that retries unsolved prompts instead of dropping them.

    A kept group is replaced in place by the merged chain, so callers train on whatever
    ``group`` holds after the call. The pending chains live here, so one instance has to
    outlive a single rollout.
    """

    def __init__(self, args, *, dynamic_filter, data_source: "RolloutDataSourceWithBuffer"):
        self._args = args
        self._dynamic_filter = dynamic_filter
        self._data_source = data_source
        self._rng = random.Random(args.rollout_seed)
        self._chains: dict[int, PendingChain] = {}

    def __call__(self, args, group: list[Sample], **kwargs) -> FilterOutput:
        output = call_dynamic_filter(self._dynamic_filter, args, group, **kwargs)
        chain_id = group[0].group_index
        rewards = [sample.get_reward_value(args) for sample in group]
        pending = self._chains.pop(chain_id, None)

        if _should_accept(
            rewards, filter_keep=output.keep, best_reward=None if pending is None else pending.best_reward
        ):
            group[:] = self._merge_chain(pending, group=group, rewards=rewards)
            return FilterOutput(keep=True)

        best_reward = max(rewards) if pending is None else max(pending.best_reward, *rewards)
        if best_reward >= self._args.ngu_solved_reward or self._rng.random() >= self._args.never_give_up:
            return FilterOutput(keep=False, reason=output.reason or "never_give_up_not_improved")

        pending = pending or PendingChain(best_reward=best_reward, retry_template=group[0])
        pending.best_reward = best_reward
        pending.attempts.append((_weight_version(group), group))
        pending.sample_count += len(group)
        pending.reward_sum += sum(rewards)
        pending.attempt_count += 1
        self._chains[chain_id] = pending
        self._requeue(pending.retry_template)
        return FilterOutput(keep=False, reason="never_give_up_requeued")

    def requeue_lost_retries(self) -> None:
        """Requeue chains whose retry is neither buffered nor generating, e.g. because it was
        aborted when a rollout ended or failed a filter before reaching NGU."""
        buffered = {group[0].group_index for group in self._data_source.buffer}
        for chain_id, pending in self._chains.items():
            if chain_id not in buffered:
                self._requeue(pending.retry_template)

    def _requeue(self, retry_template: Sample) -> None:
        sample_indices = self._data_source.reserve_sample_indices(self._args.n_samples_per_prompt)
        self._data_source.add_samples([make_retry_group(retry_template, sample_indices=sample_indices)])

    def _merge_chain(self, pending: PendingChain | None, *, group: list[Sample], rewards: list[float]) -> list[Sample]:
        """The chain's fresh-enough buffered attempts plus the kept one, each sample carrying the
        chain-wide baseline."""
        if pending is None:
            pending = PendingChain(best_reward=max(rewards), retry_template=group[0])
        version = _weight_version(group)
        max_staleness = self._args.max_weight_staleness
        merged = [
            sample
            for attempt_version, attempt in pending.attempts
            if max_staleness is None
            or version is None
            or attempt_version is None
            or version - attempt_version <= max_staleness
            for sample in attempt
        ]
        merged.extend(group)
        for sample in merged:
            sample.group_index = group[0].group_index
            sample.metadata[NGU_BASELINE_REWARD_SUM_KEY] = pending.reward_sum + sum(rewards)
            sample.metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] = pending.sample_count + len(group)
            sample.metadata[NGU_ATTEMPT_COUNT_KEY] = pending.attempt_count + 1
        return merged


def _should_accept(rewards: list[float], *, filter_keep: bool, best_reward: float | None) -> bool:
    if best_reward is None:
        return filter_keep
    return max(rewards) > best_reward


def _weight_version(group: list[Sample]) -> int | None:
    return group_weight_version_stats(group).newest_version


def make_retry_group(retry_template: Sample, *, sample_indices: list[int]) -> list[Sample]:
    """Fresh copies of a chain's prompt. They keep its ``group_index`` so the retry stays on the chain."""
    group = []
    for index in sample_indices:
        sample = copy.deepcopy(retry_template)
        sample.reset_for_retry()
        sample.status = Sample.Status.PENDING
        sample.index = index
        group.append(sample)
    return group


def prune_stale_attempts(
    group: list[Sample], *, attempt_size: int, is_stale: Callable[[list[Sample]], bool]
) -> list[Sample]:
    """Drop the buffered attempts of a merged group that became too stale while it waited to be
    trained on. The kept attempt (the last one) always stays, and the chain-wide baseline on the
    samples still counts the dropped attempts' rewards."""
    if len(group) == attempt_size:
        return group
    attempts = [group[start : start + attempt_size] for start in range(0, len(group), attempt_size)]
    kept = [attempt for attempt in attempts[:-1] if not is_stale(attempt)]
    return [sample for attempt in [*kept, attempts[-1]] for sample in attempt]


def ngu_baseline_mean(samples: list[Sample]) -> float | None:
    """The chain-wide mean reward NGU recorded on a group, if any."""
    metadata = samples[0].metadata
    if NGU_BASELINE_SAMPLE_COUNT_KEY not in metadata:
        return None
    return metadata[NGU_BASELINE_REWARD_SUM_KEY] / metadata[NGU_BASELINE_SAMPLE_COUNT_KEY]


def anchor_positive_advantages(advantages: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Keep the max-reward samples' advantages and rescale the others so the group sums to zero.

    With a chain-wide baseline the group no longer sums to zero. The max-reward samples carry
    the signal NGU retried for, so they are kept as is and the rest absorb the difference.
    """
    is_positive = torch.isclose(rewards, rewards.max())
    if bool(is_positive.all()):
        return advantages

    negative_sum = advantages[~is_positive].sum()
    if negative_sum == 0:
        return advantages

    out = advantages.clone()
    out[~is_positive] = advantages[~is_positive] * (-advantages[is_positive].sum() / negative_sum)
    return out
