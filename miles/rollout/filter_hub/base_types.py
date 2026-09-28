import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.rollout.never_give_up import NGU_ATTEMPT_COUNT_KEY, NeverGiveUpDecision
from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0
        self._never_give_up_action_count = defaultdict(lambda: 0)
        self._never_give_up_kept_attempts = 0
        self._never_give_up_kept_samples = 0

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_never_give_up_decision(self, decision: NeverGiveUpDecision) -> None:
        self._never_give_up_action_count[decision.action.value] += 1
        if decision.group is not None:
            self._never_give_up_kept_attempts += decision.group[0].metadata[NGU_ATTEMPT_COUNT_KEY]
            self._never_give_up_kept_samples += len(decision.group)

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        metrics |= {
            f"rollout/never_give_up/{action}": count for action, count in self._never_give_up_action_count.items()
        }
        if kept_groups := self._never_give_up_action_count.get("keep"):
            metrics["rollout/never_give_up/mean_attempts_per_kept_group"] = (
                self._never_give_up_kept_attempts / kept_groups
            )
            metrics["rollout/never_give_up/mean_samples_per_kept_group"] = (
                self._never_give_up_kept_samples / kept_groups
            )
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
