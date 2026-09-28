"""Never Give Up (NGU): keep retrying an unsolved prompt instead of dropping it.

With dynamic sampling, a prompt group with no learning signal (for example all
rewards equal) is dropped. NGU instead requeues the *same* prompt with probability
``--never-give-up``. All attempts at one prompt form a *chain*, keyed by the
``group_index`` of its first attempt; every retry keeps that ``group_index``.

Earlier attempts are buffered on the data source until an attempt is accepted:

- Before any attempt is buffered, the dynamic sampling filter decides as usual.
- After that, an attempt is accepted only when its max reward is strictly higher
  than the best reward the chain has seen so far.

On acceptance the buffered attempts (up to ``--ngu-max-pending-age`` steps old)
are merged with the accepted one into a single group, so the group holds a multiple
of ``n_samples_per_prompt`` samples. The group's advantage baseline is then the mean
reward over *every* attempt in the chain, and the max-reward samples are anchored
while the others are rescaled so the group sums to zero
(:func:`anchor_positive_advantages`).

A port of allenai/open-instruct#1861.
"""

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from miles.utils.types import Sample

NGU_BASELINE_REWARD_SUM_KEY = "ngu_baseline_reward_sum"
NGU_BASELINE_SAMPLE_COUNT_KEY = "ngu_baseline_sample_count"
NGU_ATTEMPT_COUNT_KEY = "ngu_attempt_count"


@dataclass
class PendingAttempt:
    # Training step the attempt was generated at: the rollout id, or the weight version under fully async.
    step: int
    group: list[Sample]


@dataclass
class PendingChain:
    """Everything buffered for one prompt while NGU keeps retrying it."""

    prompt_template: Sample
    best_reward: float
    attempts: list[PendingAttempt] = field(default_factory=list)
    sample_count: int = 0
    reward_sum: float = 0.0
    attempt_count: int = 0


@dataclass
class NeverGiveUpState:
    """Pending chains keyed by chain id (the ``group_index`` shared by every attempt)."""

    chains: dict[int, PendingChain] = field(default_factory=dict)


@dataclass(frozen=True)
class NeverGiveUpConfig:
    probability: float
    max_pending_age: int
    keep_pending_completions: bool
    solved_reward: float

    @classmethod
    def from_args(cls, args) -> "NeverGiveUpConfig":
        return cls(
            probability=args.never_give_up,
            max_pending_age=args.ngu_max_pending_age,
            keep_pending_completions=args.ngu_keep_pending_completions,
            solved_reward=args.ngu_solved_reward,
        )


class NeverGiveUpAction(Enum):
    KEEP = "keep"
    DROP = "drop"
    REQUEUE = "requeue"


@dataclass(frozen=True)
class NeverGiveUpDecision:
    action: NeverGiveUpAction
    # The (possibly merged) group to train on, set only for KEEP.
    group: list[Sample] | None = None
    # Earlier attempts flushed along with a DROP, for bookkeeping.
    dropped_attempts: list[list[Sample]] = field(default_factory=list)


def chain_id_of(group: list[Sample]) -> int:
    chain_id = group[0].group_index
    assert chain_id is not None, "never_give_up needs every sample to carry a group_index"
    return chain_id


def should_accept_attempt(rewards: list[float], filter_keep: bool, best_reward: float | None) -> bool:
    """Accept the first attempt when the dynamic filter keeps it; later attempts only when they
    strictly beat the chain's best reward so far."""
    if best_reward is None:
        return filter_keep
    return max(rewards) > best_reward


def decide_never_give_up(
    *,
    config: NeverGiveUpConfig,
    state: NeverGiveUpState,
    group: list[Sample],
    rewards: list[float],
    filter_keep: bool,
    prompt_template: Sample,
    step: int,
    rng: random.Random,
) -> NeverGiveUpDecision:
    """Keep, drop or requeue a finished group, updating ``state`` in place.

    ``prompt_template`` is the group's prompt before generation; a REQUEUE stores it
    on the chain so the caller can build the retry with :func:`make_retry_group`.
    """
    chain_id = chain_id_of(group)
    pending = state.chains.pop(chain_id, None)
    best_reward = None if pending is None else pending.best_reward

    if should_accept_attempt(rewards, filter_keep=filter_keep, best_reward=best_reward):
        return NeverGiveUpDecision(
            action=NeverGiveUpAction.KEEP,
            group=_merge_chain(config, pending, group=group, rewards=rewards, step=step),
        )

    best_reward = max(rewards) if best_reward is None else max(best_reward, max(rewards))
    if best_reward >= config.solved_reward or rng.random() >= config.probability:
        dropped = [] if pending is None else [attempt.group for attempt in pending.attempts]
        return NeverGiveUpDecision(action=NeverGiveUpAction.DROP, dropped_attempts=dropped)

    if pending is None:
        pending = PendingChain(prompt_template=prompt_template, best_reward=best_reward)
    pending.best_reward = best_reward
    if config.keep_pending_completions:
        pending.attempts.append(PendingAttempt(step=step, group=group))
    pending.sample_count += len(group)
    pending.reward_sum += sum(rewards)
    pending.attempt_count += 1
    state.chains[chain_id] = pending
    return NeverGiveUpDecision(action=NeverGiveUpAction.REQUEUE)


def _merge_chain(
    config: NeverGiveUpConfig,
    pending: PendingChain | None,
    *,
    group: list[Sample],
    rewards: list[float],
    step: int,
) -> list[Sample]:
    """Merge the chain's fresh-enough buffered attempts into the accepted group and record the
    chain-wide baseline on every sample."""
    attempts = [] if pending is None else pending.attempts
    merged = [
        sample
        for attempt in attempts
        if config.max_pending_age < 0 or step - attempt.step <= config.max_pending_age
        for sample in attempt.group
    ]
    merged.extend(group)

    chain_id = chain_id_of(group)
    baseline_reward_sum = sum(rewards) + (0.0 if pending is None else pending.reward_sum)
    baseline_sample_count = len(group) + (0 if pending is None else pending.sample_count)
    attempt_count = 1 + (0 if pending is None else pending.attempt_count)
    for sample in merged:
        sample.group_index = chain_id
        sample.metadata[NGU_BASELINE_REWARD_SUM_KEY] = baseline_reward_sum
        sample.metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] = baseline_sample_count
        sample.metadata[NGU_ATTEMPT_COUNT_KEY] = attempt_count
    return merged


def prune_stale_attempts(
    group: list[Sample], *, attempt_size: int, is_stale: Callable[[list[Sample]], bool]
) -> list[Sample]:
    """Drop the buffered attempts of a merged group that became too stale while it waited to be
    trained on. The accepted attempt (the last one) is always kept, and the chain-wide baseline
    recorded on the samples still counts the dropped attempts' rewards."""
    assert len(group) % attempt_size == 0, f"a merged group of {len(group)} samples is not whole attempts"
    attempts = [group[start : start + attempt_size] for start in range(0, len(group), attempt_size)]
    kept = [attempt for attempt in attempts[:-1] if not is_stale(attempt)]
    return [sample for attempt in [*kept, attempts[-1]] for sample in attempt]


def make_prompt_template(sample: Sample) -> Sample:
    """Snapshot a not-yet-generated sample so it can be re-sampled later."""
    return copy.deepcopy(sample)


def make_retry_group(prompt_template: Sample, *, sample_indices: list[int]) -> list[Sample]:
    """Fresh copies of the chain's prompt, one per new sample index. They keep the template's
    ``group_index`` so the retry stays on the same chain."""
    group = []
    for index in sample_indices:
        sample = copy.deepcopy(prompt_template)
        sample.reset_for_retry()
        sample.status = Sample.Status.PENDING
        sample.index = index
        group.append(sample)
    return group


def ngu_baseline_mean(samples: list[Sample]) -> float | None:
    """The chain-wide mean reward recorded by :func:`decide_never_give_up`, if any."""
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
