from collections.abc import Callable

from miles.rollout.resampling import FAILED_OUTCOMES, GroupOutcome, ResamplingPolicy
from miles.utils.types import Sample


class RecordingPolicy(ResamplingPolicy):
    """Hands each failed round's prompt group and outcome to ``record``; ignores kept and filtered groups."""

    def __init__(self, record: Callable[[list[Sample], GroupOutcome], object] = lambda prompt_group, outcome: None):
        self._record = record

    def __call__(self, entry, outcome: GroupOutcome) -> None:
        if outcome in FAILED_OUTCOMES:
            self._record(entry.prompt_group, outcome)
