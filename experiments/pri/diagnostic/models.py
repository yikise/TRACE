"""Data structures for the frozen Bilateral PRI diagnostic.

The diagnostic deliberately keeps the view payload opaque.  A JSONL producer can
store an audio path, an already-loaded waveform, or a model-specific feature
reference; the scoring adapter decides how that payload is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


VIEW_NAMES: tuple[str, ...] = ("x_f", "n_f", "x_s", "n_s")


def normalize_label(value: Any, *, field_name: str = "label") -> str:
    """Return a normalized answer label suitable for answer-only scoring.

    Labels are intentionally represented as strings rather than token ids.  The
    Hugging Face adapter validates that each string maps to exactly one tokenizer
    token at scoring time.  Keeping this check here catches malformed JSONL early
    while still allowing labels such as ``"A"``, ``"B"`` and ``"<yes>"``.
    """

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    label = value.strip()
    if not label:
        raise ValueError(f"{field_name} must not be empty")
    if any(character.isspace() for character in label):
        raise ValueError(
            f"{field_name} must be a single whitespace-free option token: {value!r}"
        )
    return label


@dataclass(frozen=True)
class ViewRecord:
    """One of ``x_f``, ``n_f``, ``x_s`` or ``n_s``.

    ``payload`` is the exact JSON value supplied by the data manifest.  For
    convenience, mapping payloads may contain ``audio``/``audio_path`` and a
    per-view ``question``; neither is interpreted by this class.
    """

    name: str
    payload: Any
    question: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in VIEW_NAMES:
            raise ValueError(f"unknown PRI view {self.name!r}; expected one of {VIEW_NAMES}")
        if not isinstance(self.question, str):
            raise ValueError("view question must be a string")


@dataclass(frozen=True)
class FourViewRecord:
    """A factual/swapped pair and their matched predicate-preserving nulls."""

    record_id: str
    source_tuple_id: str
    event_pair: str
    question_template: str
    question: str
    y_f: str
    y_s: str
    views: Mapping[str, ViewRecord]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must not be empty")
        if not self.source_tuple_id:
            raise ValueError("source_tuple_id must not be empty")
        if not isinstance(self.question, str):
            raise ValueError("question must be a string")
        object.__setattr__(self, "y_f", normalize_label(self.y_f, field_name="y_f"))
        object.__setattr__(self, "y_s", normalize_label(self.y_s, field_name="y_s"))
        if self.y_f == self.y_s:
            raise ValueError("y_f and y_s must be distinct labels")
        missing = [name for name in VIEW_NAMES if name not in self.views]
        if missing:
            raise ValueError(f"missing PRI views: {', '.join(missing)}")
        wrong = [name for name, view in self.views.items() if view.name != name]
        if wrong:
            raise ValueError(f"view mapping/name mismatch for: {wrong}")

    @property
    def cluster_id(self) -> str:
        """The source/stem tuple unit used by all default resampling."""

        return self.source_tuple_id

    @property
    def stratum(self) -> tuple[str, str]:
        """Label-shuffle stratum (event pair, question template)."""

        return self.event_pair, self.question_template

    def labels(self) -> tuple[str, str]:
        return self.y_f, self.y_s

