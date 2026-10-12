"""Rule-based prompt-injection detector.

Threat model: repository text (comments, docstrings, string literals) written to
manipulate the LLM that reads search results. Rules come in two kinds:

* **character rules** run on the raw text and catch invisible or reordering
  Unicode (zero-width, bidi controls, tag characters);
* **phrase rules** run on a normalised copy (NFKC, invisible characters removed,
  whitespace collapsed), so ``ig​nore previous instructions`` cannot slip
  past a pattern by splitting words.

Severity encodes confidence: ``high`` rules match phrasing that has no business in
source code (instruction overrides, chat-template tokens, requests to print SSH
keys); ``low`` rules match things that are suspicious but have benign uses (an
install one-liner piping ``curl`` into ``sh``, "send ... to https://...").

Heuristics are cheap and explainable, but they are pattern matching: paraphrases,
other languages and encodings (base64, leetspeak) get through. The evaluation in
``evals/datasets/injection.jsonl`` measures exactly that.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from repolens.guard.base import Detection, Severity

_ZERO_WIDTH = "​‌‍⁠᠎"
_BIDI_CONTROLS = "‪‫‬‭‮⁦⁧⁨⁩"
_TAG_CHARS = r"\U000e0000-\U000e007f"
_INVISIBLE_RE = re.compile(f"[{_ZERO_WIDTH}﻿{_TAG_CHARS}]")
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Rule:
    """A named pattern with a human-readable reason."""

    id: str
    reason: str
    severity: Severity
    pattern: re.Pattern[str]
    on_raw_text: bool = False


def _phrase(rule_id: str, reason: str, severity: Severity, pattern: str) -> Rule:
    return Rule(rule_id, reason, severity, re.compile(pattern, re.IGNORECASE | re.MULTILINE))


def _chars(rule_id: str, reason: str, severity: Severity, chars: str) -> Rule:
    return Rule(rule_id, reason, severity, re.compile(f"[{chars}]"), on_raw_text=True)


_TARGET = r"(?:instructions?|prompts?|directions?|directives?|rules|guidelines|context|messages?)"
_SENSITIVE = (
    r"(?:~?/?\.ssh\b|id_rsa|id_ed25519|id_ecdsa|\.env\b|/etc/(?:passwd|shadow)|"
    r"\.aws/credentials|\.netrc|\.npmrc|\.pypirc|private\s+keys?)"
)
_REVEAL = (
    r"(?:print|show|reveal|output|display|dump|cat|read|leak|exfiltrate|send|upload|post|copy)"
)

RULES: tuple[Rule, ...] = (
    _phrase(
        "override.ignore_instructions",
        "tells the reader to ignore or replace its previous instructions",
        Severity.HIGH,
        r"\b(?:ignore|disregard|forget|override|bypass|skip)\s+(?:all\s+|any\s+|the\s+|your\s+|of\s+)*"
        rf"(?:previous|prior|above|earlier|preceding|system|original|existing)\s+{_TARGET}",
    ),
    _phrase(
        "override.mode_switch",
        "tries to switch the reader into a privileged or unrestricted mode",
        Severity.HIGH,
        r"\byou\s+are\s+now\s+(?:in\s+)?(?:an?\s+)?"
        r"(?:developer|dan|jailbreak|jailbroken|unrestricted|unfiltered|god|admin|root|debug)"
        r"\s+mode\b|\b(?:enable|activate|enter)\s+(?:developer|dan|jailbreak|god)\s+mode\b",
    ),
    _phrase(
        "role.chat_template_tokens",
        "contains chat-template control tokens used to fake a system or assistant turn",
        Severity.HIGH,
        r"<\|(?:im_start|im_end|system|assistant|user|begin_of_text|start_header_id|eot_id)\|>"
        r"|\[/?INST\]|<<\s*/?SYS\s*>>",
    ),
    _phrase(
        "role.impersonation",
        "impersonates a system/assistant message that addresses the reader",
        Severity.HIGH,
        r"^[\s#/*\"'>-]*(?:system|assistant|developer)\s*:\s*"
        r"(?:you\b|ignore\b|disregard\b|from\s+now\s+on\b|act\s+as\b|your\b|new\s+instructions\b)",
    ),
    _phrase(
        "exec.command_instruction",
        "instructs the reader to run a shell command",
        Severity.HIGH,
        r"\b(?:run|execute|exec)\s+(?:the\s+following\s+)?(?:command\s*:?\s*)?[`'\"]?\s*"
        r"(?:curl|wget|bash|sh\b|zsh|rm\s+-rf|powershell|iex\b|nc\s|python3?\s+-c)",
    ),
    _phrase(
        "exfil.sensitive_files",
        "asks to reveal or send credential files or private keys",
        Severity.HIGH,
        rf"\b{_REVEAL}\b[\s\S]{{0,80}}?{_SENSITIVE}",
    ),
    _chars(
        "unicode.bidi_controls",
        "contains bidirectional control characters that can reorder displayed text",
        Severity.HIGH,
        _BIDI_CONTROLS,
    ),
    _chars(
        "unicode.tag_characters",
        "contains invisible Unicode tag characters (hidden ASCII smuggling)",
        Severity.HIGH,
        _TAG_CHARS,
    ),
    _chars(
        "unicode.zero_width",
        "contains zero-width characters that can hide or split words",
        Severity.LOW,
        _ZERO_WIDTH,
    ),
    _phrase(
        "exfil.secrets",
        "asks to reveal or send secrets, keys or credentials",
        Severity.LOW,
        r"\b(?:print|reveal|leak|exfiltrate|dump|send|output)\b[\s\S]{0,60}?"
        r"\b(?:api[\s_-]?keys?|secrets?|passwords?|credentials|access\s+tokens?|"
        r"environment\s+variables)\b",
    ),
    _phrase(
        "exfil.send_to_url",
        "asks to send data to an external URL",
        Severity.LOW,
        # Prose only ("send the logs to https://..."): the verb must be followed by a space,
        # which excludes calls such as client.post("https://...") in ordinary HTTP code.
        r"\b(?:send|post|upload|exfiltrate|forward|transmit)\s+[^\n(\"']{0,80}?\bto\s+https?://\S+",
    ),
    _phrase(
        "exec.pipe_to_shell",
        "downloads a script and pipes it straight into a shell",
        Severity.LOW,
        r"\b(?:curl|wget)\b[^\n|]{0,200}\|\s*(?:sudo\s+)?(?:ba|z)?sh\b",
    ),
    _phrase(
        "override.addresses_ai",
        "addresses AI assistants directly with an instruction",
        Severity.LOW,
        r"\b(?:ai\s+(?:assistant|agent|model)s?|language\s+models?|llms?|chatbots?)\b[^\n]{0,40}"
        r"\b(?:must|should|need\s+to|are\s+required\s+to|always|never)\b",
    ),
)


def normalise(text: str) -> str:
    """NFKC-fold, drop invisible characters and collapse runs of spaces/tabs."""
    folded = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", _INVISIBLE_RE.sub("", folded))


class HeuristicDetector:
    """Applies :data:`RULES`; each rule reports at most once per text."""

    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self._rules = rules

    def scan(self, text: str) -> list[Detection]:
        """Return one detection per matching rule, in rule order."""
        normalised = normalise(text)
        return [
            Detection(rule_id=rule.id, reason=rule.reason, severity=rule.severity)
            for rule in self._rules
            if rule.pattern.search(_strip_leading_bom(text) if rule.on_raw_text else normalised)
        ]


def _strip_leading_bom(text: str) -> str:
    return text.removeprefix("﻿")
