"""InjectionDetector protocol and the detection record it produces."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class Severity(StrEnum):
    """How strongly a detection indicates an injection attempt."""

    LOW = "low"
    HIGH = "high"


class Detection(BaseModel):
    """A single rule or classifier hit on a piece of text."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    reason: str
    severity: Severity


class InjectionDetector(Protocol):
    """Scans untrusted text for prompt-injection attempts."""

    def scan(self, text: str) -> list[Detection]:
        """Return all detections for ``text``; an empty list means clean."""
        ...
