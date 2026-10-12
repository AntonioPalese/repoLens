"""Optional prompt-injection classifier (a Hugging Face text-classification model).

Off by default: it downloads a ~700 MB model and is slow on CPU. Classifiers like
``protectai/deberta-v3-base-prompt-injection-v2`` are trained on natural-language
prompts, not code, so their hits are reported with ``low`` severity: they mark a
chunk ``suspected`` but never block it on their own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from repolens.guard.base import Detection, Severity

logger = logging.getLogger(__name__)

Classify = Callable[..., Any]


class ClassifierDetector:
    """Scores overlapping windows of the text; reports the highest injection score."""

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.5,
        label: str = "INJECTION",
        device: str = "cpu",
        window_chars: int = 1500,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._label = label
        self._device = device
        self._window = window_chars

    def scan(self, text: str) -> list[Detection]:
        """Return one ``low`` detection if any window scores above the threshold."""
        windows = _windows(text, self._window)
        if not windows:
            return []
        classify = _load_classifier(self._model_name, self._device)
        results = classify(windows, truncation=True, top_k=None)
        score = max(self._label_score(r) for r in results)
        if score < self._threshold:
            return []
        return [
            Detection(
                rule_id="classifier.injection",
                reason=f"{self._model_name} scored {score:.2f} for {self._label}",
                severity=Severity.LOW,
            )
        ]

    def _label_score(self, scores: list[dict[str, Any]]) -> float:
        return max((float(s["score"]) for s in scores if s["label"] == self._label), default=0.0)


def _windows(text: str, size: int) -> list[str]:
    """Split into windows of ``size`` characters overlapping by a quarter."""
    stripped = text.strip()
    if not stripped:
        return []
    step = max(1, size * 3 // 4)
    return [stripped[i : i + size] for i in range(0, max(1, len(stripped) - size + step), step)]


@lru_cache(maxsize=2)
def _load_classifier(model_name: str, device: str) -> Classify:
    """Load the text-classification pipeline once per process."""
    from transformers import pipeline

    logger.info("Loading injection classifier %s on %s", model_name, device)
    classify: Classify = pipeline("text-classification", model=model_name, device=device)
    return classify
