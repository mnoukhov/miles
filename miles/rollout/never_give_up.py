"""Never Give Up (NGU): keep retrying an unsolved prompt instead of dropping it.

Selected with ``--async-unused-samples-handler never_give_up`` under ``--fully-async``. With
dynamic sampling, a prompt group with no learning signal (for example all rewards equal) is
dropped. NGU instead requeues
the *same* prompt through the data source's buffer like any other resubmitted group, with
probability ``--ngu-requeue-probability``. All attempts at one prompt form a *chain* that shares
the first attempt's ``group_index``.

The :class:`NeverGiveUp` handler is told why a group is or is not trained on:

- a dynamic filter rejection: the attempt is buffered on its chain and the prompt requeued, unless
  the chain already reached ``--ngu-solved-reward`` or the requeue draw fails;
- aborted or missing reward: nothing is learned, so the chain's retry is requeued as it was;
- stale: a failure that keeps going. Its rewards and completions are buffered
  for the baseline, the chain's best reward is reset, and the prompt is always requeued;
- kept: the group is merged in place with the chain's buffered attempts (up to
  ``--max-weight-staleness`` weight versions old), so it holds a multiple of
  ``n_samples_per_prompt`` samples, and the chain ends.

A merged group trains against its whole chain, stale-pruned attempts included: before the GRPO
step, :func:`chain_rewards_and_std` shifts the rewards below the group max so the group's mean is
the chain's mean reward, and gives the chain's reward std.

A port of allenai/open-instruct#1861.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from miles.rollout.filter_hub.base_types import UnusedSamplesHandler
from miles.rollout.filter_hub.common_filters import FilterReason, group_staleness, group_weight_version_stats
from miles.utils.types import Sample

if TYPE_CHECKING:  # a runtime import would be circular through miles.rollout.base_types
    from miles.rollout.data_source import RolloutDataSourceWithBuffer

NGU_BASELINE_REWARD_SUM_KEY = "ngu_baseline_reward_sum"
NGU_BASELINE_REWARD_SQ_SUM_KEY = "ngu_baseline_reward_sq_sum"
NGU_BASELINE_SAMPLE_COUNT_KEY = "ngu_baseline_sample_count"
_NGU_METADATA_KEYS = (NGU_BASELINE_REWARD_SUM_KEY, NGU_BASELINE_REWARD_SQ_SUM_KEY, NGU_BASELINE_SAMPLE_COUNT_KEY)


@dataclass
class PendingChain:
    """Everything buffered for one prompt while NGU keeps retrying it."""

    retry_template: Sample
    best_reward: float | None = None  # None right after a stale reset
    samples: list[Sample] = field(default_factory=list)  # buffered attempts, n_samples_per_prompt each
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    sample_count: int = 0


class NeverGiveUp(UnusedSamplesHandler):
    """The ``never_give_up`` unused-samples handler.

    A kept group is replaced in place by the merged chain, so callers train on whatever ``group``
    holds after the handler. The pending chains live here, so one instance has to outlive a single
    rollout.
    """

    def __init__(self, args, *, data_source: "RolloutDataSourceWithBuffer"):
        self._args = args
        self._data_source = data_source
        self._rng = random.Random(args.rollout_seed)
        self._chains: dict[int, PendingChain] = {}

    def __call__(self, prompt_group: list[Sample], *, group: list[Sample], reason: str | None) -> None:
        if reason == FilterReason.kept:
            self._merge_into_kept(group)
        elif reason in (FilterReason.aborted, FilterReason.missing_reward):
            self._retry_pending_chain(group)
        elif reason == FilterReason.stale:
            self._keep_going_after_stale(group)
        else:
            self._keep_going_or_drop(group)

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
        chain_stats = {
            NGU_BASELINE_REWARD_SUM_KEY: pending.reward_sum + sum(rewards),
            NGU_BASELINE_REWARD_SQ_SUM_KEY: pending.reward_sq_sum + _sq_sum(rewards),
            NGU_BASELINE_SAMPLE_COUNT_KEY: pending.sample_count + len(group),
        }
        for sample in merged:
            sample.group_index = group[0].group_index
            sample.metadata.update(chain_stats)
        group[:] = merged

    def _retry_pending_chain(self, group: list[Sample]) -> None:
        if (pending := self._chains.get(group[0].group_index)) is not None:
            self._requeue(pending.retry_template)

    def _keep_going_after_stale(self, group: list[Sample]) -> None:
        """A stale accepted attempt failed to train, but its rewards still shape the baseline."""
        pending = self._chains.get(group[0].group_index) or PendingChain(retry_template=group[0])
        metadata, rewards = group[0].metadata, _rewards(self._args, group)
        pending.samples.extend(group)
        pending.reward_sum += metadata.get(NGU_BASELINE_REWARD_SUM_KEY, sum(rewards))
        pending.reward_sq_sum += metadata.get(NGU_BASELINE_REWARD_SQ_SUM_KEY, _sq_sum(rewards))
        pending.sample_count += metadata.get(NGU_BASELINE_SAMPLE_COUNT_KEY, len(group))
        pending.best_reward = None
        self._chains[group[0].group_index] = pending
        self._requeue(pending.retry_template)

    def _keep_going_or_drop(self, group: list[Sample]) -> None:
        chain_id = group[0].group_index
        rewards = _rewards(self._args, group)
        pending = self._chains.pop(chain_id, None) or PendingChain(retry_template=group[0])
        best_reward = max(rewards) if pending.best_reward is None else max(pending.best_reward, *rewards)
        if best_reward >= self._args.ngu_solved_reward or self._rng.random() >= self._args.ngu_requeue_probability:
            return

        pending.samples.extend(group)
        pending.reward_sum += sum(rewards)
        pending.reward_sq_sum += _sq_sum(rewards)
        pending.sample_count += len(group)
        pending.best_reward = best_reward
        self._chains[chain_id] = pending
        self._requeue(pending.retry_template)

    def _requeue(self, retry_template: Sample) -> None:
        sample_indices = self._data_source.reserve_sample_indices(self._args.n_samples_per_prompt)
        self._data_source.add_samples([make_retry_group(retry_template, sample_indices=sample_indices)])


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
        for key in _NGU_METADATA_KEYS:
            sample.metadata.pop(key, None)
        sample.index = index
        group.append(sample)
    return group


def prune_stale_attempts(
    group: list[Sample], *, attempt_size: int, current_version: int | None, max_staleness: int | None
) -> list[Sample]:
    """Drop the buffered attempts of a merged group that became more than ``max_staleness`` weight
    versions old while it waited to be trained on. The kept attempt (the last one) always stays,
    and the chain-wide baseline on the samples still counts the dropped attempts' rewards."""
    if max_staleness is None or len(group) == attempt_size:
        return group

    def is_stale(attempt: list[Sample]) -> bool:
        staleness = group_staleness(attempt, current_version)
        return staleness is not None and staleness > max_staleness

    attempts = _split_attempts(group, attempt_size=attempt_size)
    kept = [attempt for attempt in attempts[:-1] if not is_stale(attempt)]
    return [sample for attempt in [*kept, attempts[-1]] for sample in attempt]


def chain_rewards_and_std(
    group_samples: list[Sample], rewards: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rewards and reward std a group's GRPO step should use.

    For a merged never_give_up group the rewards below the group max are shifted by one amount so
    the group's mean becomes the chain baseline ``b`` (mean reward over every attempt in the chain,
    stale-pruned ones included), and the std is the chain's std around ``b``. The shift is skipped
    when it would lift a negative to the max. Any other group keeps its ``rewards`` and ``std``.
    """
    metadata = group_samples[0].metadata
    if NGU_BASELINE_SAMPLE_COUNT_KEY not in metadata:
        return rewards, std

    count = metadata[NGU_BASELINE_SAMPLE_COUNT_KEY]
    baseline = metadata[NGU_BASELINE_REWARD_SUM_KEY] / count
    variance = (metadata[NGU_BASELINE_REWARD_SQ_SUM_KEY] - count * baseline**2) / max(count - 1, 1)
    chain_std = torch.tensor(max(variance, 0.0) ** 0.5)

    is_negative = rewards < rewards.max()
    if not bool(is_negative.any()):
        return rewards, chain_std
    shifted = rewards.clone()
    shifted[is_negative] += (len(rewards) * baseline - rewards.sum()) / is_negative.sum()
    if shifted[is_negative].max() >= rewards.max():  # a negative would overtake the max: skip the shift
        return rewards, chain_std
    return shifted, chain_std
