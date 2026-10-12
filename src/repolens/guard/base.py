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


class CompositeDetector:
    """Runs several detectors and concatenates their detections."""

    def __init__(self, detectors: list[InjectionDetector]) -> None:
        self._detectors = detectors

    def scan(self, text: str) -> list[Detection]:
        """Detections of every detector, in detector order."""
        return [d for detector in self._detectors for d in detector.scan(text)]
