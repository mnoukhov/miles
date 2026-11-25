"""What becomes of each finished prompt group in fully async mode (``--async-unused-samples-handler``).

The data buffer tells a :class:`ResamplingPolicy` the :class:`GroupOutcome` of every finished
group. Each group is one round of ``n_samples_per_prompt`` samples at its prompt, and the policy
decides whether the prompt gets another round and how its rounds combine into the group that
trains:

- ``drop`` (:class:`NoResampling`): one round per prompt; a failed round is discarded.
- ``retry`` (:class:`ResampleOnFailure`): a round lost to an abort or staleness is redone.
- ``never_give_up`` (:class:`~miles.rollout.never_give_up.NeverGiveUp`): a round the dynamic
  sampling filter would discard because every response is wrong is retried sequentially, and
  the group that finally trains merges all of its attempts together.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.types import Sample

if TYPE_CHECKING:  # the data buffer imports this module
    from miles.rollout.fully_async_data_buffer import DataBufferInput


class GroupOutcome(str, Enum):
    # The group trains. The policy may edit ``entry.group`` in place before it is buffered.
    KEPT = "kept"
    # Rejected by --dynamic-sampling-filter-path, or for a missing reward; recorded as dropped.
    FILTERED = "filtered"
    MISSING_REWARD = "missing_reward"
    # Failed rounds, which a policy may redo.
    ABORTED = "aborted"
    STALE = "stale"
    # Restored from a checkpoint with output store refs, whose objects did not survive the restart.
    RESTORED = "restored"


FAILED_OUTCOMES = (GroupOutcome.ABORTED, GroupOutcome.STALE, GroupOutcome.RESTORED)

# A policy whose group should train against rewards it no longer holds (e.g. never_give_up's earlier
# attempts at the prompt) stamps their totals on every sample of the group as
# ``{"sum": ..., "sq_sum": ..., "count": ...}``; see :func:`baseline_rewards_and_std`.
REWARD_BASELINE_KEY = "reward_baseline"


class ResamplingPolicy:
    """Told the outcome of every finished group; decides what happens to its prompt next."""

    def __call__(self, entry: DataBufferInput, outcome: GroupOutcome) -> None:
        raise NotImplementedError

    def prune_stale(self, entry: DataBufferInput, *, current_version: int | None) -> None:
        """Called on a buffered group that went over ``--max-weight-staleness``, before it is dropped.

        A policy may remove samples from ``entry.group`` in place, keeping it nonempty. The buffer then
        checks the group's staleness again: if it is within the limit the group stays buffered,
        otherwise it is dropped and reported as ``STALE``. By default nothing is pruned.
        """

    def check_trained_group(self, group: list[Sample], *, n_samples_per_prompt: int) -> None:
        """Assert that a group about to train has the shape this policy produces: one round by default."""
        assert len(group) == n_samples_per_prompt, f"expected {n_samples_per_prompt} samples, got {len(group)}"

    def collect_metrics(self) -> dict[str, float]:
        """Buffer-side metrics, reported with the data buffer's."""
        return {}

    def rollout_metrics(self, groups: list[list[Sample]]) -> dict[str, float]:
        """Metrics about the groups one rollout trains on."""
        return {}

    def state_dict(self) -> Any:
        return None

    def load_state_dict(self, state: Any) -> None:
        pass


class NoResampling(ResamplingPolicy):
    """``drop``: a failed round is discarded and recorded as dropped."""

    def __init__(self, args, *, admission_changed: Callable[[], None]):
        self._args = args
        self._admission_changed = admission_changed

    def __call__(self, entry: DataBufferInput, outcome: GroupOutcome) -> None:
        if outcome in FAILED_OUTCOMES:
            SampleOwnershipRecorder.log_dropped_samples(
                args=self._args, samples=entry.prompt_group, reason=outcome.value
            )
            self._admission_changed()


class ResampleOnFailure(ResamplingPolicy):
    """``retry``: a failed round's prompt group is reset and submitted again."""

    def __init__(self, *, requeue: Callable[[list[Sample]], None]):
        self._requeue = requeue

    def __call__(self, entry: DataBufferInput, outcome: GroupOutcome) -> None:
        if outcome in FAILED_OUTCOMES:
            for sample in entry.prompt_group:
                sample.reset_for_retry()
            self._requeue(entry.prompt_group)


def make_resampling_policy(
    args,
    *,
    data_source,
    requeue: Callable[[list[Sample]], None],
    admission_changed: Callable[[], None],
) -> ResamplingPolicy:
    """The policy ``--async-unused-samples-handler`` names.

    ``requeue`` submits a prompt group again ahead of new prompts, and ``admission_changed`` tells
    the producer that a group left without training. never_give_up also takes fresh sample indices
    for its retries from ``data_source``.
    """
    name = args.async_unused_samples_handler
    if name == "drop":
        return NoResampling(args, admission_changed=admission_changed)
    if name == "retry":
        return ResampleOnFailure(requeue=requeue)
    assert name == "never_give_up", f"unknown --async-unused-samples-handler {name!r}"
    # Imported here: NeverGiveUp subclasses NoResampling from this module.
    from miles.rollout.never_give_up import NeverGiveUp

    return NeverGiveUp(
        args,
        admission_changed=admission_changed,
        requeue=requeue,
        reserve_sample_indices=data_source.reserve_sample_indices,
    )


def baseline_rewards_and_std(
    group_samples: list[Sample], rewards: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rewards and reward std a group's GRPO step should use.

    A group without :data:`REWARD_BASELINE_KEY` keeps its ``rewards`` and ``std``. Otherwise the
    rewards below the group max are shifted by one amount so the group's mean becomes the stamped
    baseline ``b`` (the mean over every reward the stamp counts), and the std is the stamped rewards'
    std around ``b``. The shift is skipped when it would lift a non-max reward to the max.
    """
    stamp = group_samples[0].metadata.get(REWARD_BASELINE_KEY)
    if stamp is None:
        return rewards, std

    count = stamp["count"]
    baseline = stamp["sum"] / count
    variance = (stamp["sq_sum"] - count * baseline**2) / max(count - 1, 1)
    baseline_std = torch.tensor(max(variance, 0.0) ** 0.5)

    is_negative = rewards < rewards.max()
    if not bool(is_negative.any()):
        return rewards, baseline_std
    shifted = rewards.clone()
    shifted[is_negative] += (len(rewards) * baseline - rewards.sum()) / is_negative.sum()
    if shifted[is_negative].max() >= rewards.max():  # a non-max reward would overtake the max: skip the shift
        return rewards, baseline_std
    return shifted, baseline_std
