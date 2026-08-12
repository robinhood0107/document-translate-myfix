from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TRANSLATE = "translate_inpaint"
PRESERVE = "preserve"
REVIEW = "review"


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    role: str
    action: str
    available: bool = True
    reason: str = ""


def default_detector_decision(region: Mapping[str, object]) -> SemanticDecision:
    proposal = region.get("proposal")
    evidence = proposal if isinstance(proposal, Mapping) else region
    text_class = str(evidence.get("text_class") or "").strip().lower()
    if text_class == "text_bubble":
        return SemanticDecision("dialogue_bubble", TRANSLATE)
    if text_class == "text_free":
        return SemanticDecision("dialogue_free", TRANSLATE)
    if text_class in {"narration", "caption"}:
        return SemanticDecision("narration", TRANSLATE)
    if text_class in {"sfx", "onomatopoeia"}:
        return SemanticDecision("sfx", PRESERVE)
    if text_class in {"decorative", "decoration"}:
        return SemanticDecision("decorative", PRESERVE)
    return SemanticDecision("ambiguous", REVIEW)


def explicit_decision(
    region: Mapping[str, object],
    *,
    role_key: str,
    action_key: str,
) -> SemanticDecision:
    proposal = region.get("proposal")
    evidence = proposal if isinstance(proposal, Mapping) else region
    role = str(evidence.get(role_key) or "").strip().lower()
    action = str(evidence.get(action_key) or "").strip().lower()
    if not role or not action:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            available=False,
            reason=f"missing_{role_key}_or_{action_key}",
        )
    return SemanticDecision(role, action)


def consensus_decision(
    detector: SemanticDecision,
    ocr: SemanticDecision,
) -> SemanticDecision:
    if not detector.available or not ocr.available:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            available=False,
            reason="consensus_provider_missing",
        )
    if detector.role == ocr.role and detector.action == ocr.action:
        return detector
    return SemanticDecision(
        "ambiguous",
        REVIEW,
        reason="explicit_role_disagreement",
    )
