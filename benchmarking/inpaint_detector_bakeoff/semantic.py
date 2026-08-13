from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


TRANSLATE = "translate_inpaint"
PRESERVE = "preserve"
REVIEW = "review"
OCR_EVIDENCE_AVAILABLE = "ocr_evidence_available"


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


def ocr_provenance_decision(region: Mapping[str, object]) -> SemanticDecision:
    """Route only provider-backed OCR text and explicit preserve evidence.

    OCR boxes remain ownership-only.  This decision never creates edit pixels;
    it only admits or rejects a detector claim already owned by the region.
    """

    proposal = region.get("proposal")
    evidence = proposal if isinstance(proposal, Mapping) else region
    if evidence.get(OCR_EVIDENCE_AVAILABLE) is not True:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            available=False,
            reason="ocr_evidence_provider_missing",
        )

    text_class = str(evidence.get("text_class") or "").strip().lower()
    hinted_role = str(evidence.get("semantic_role_hint") or "").strip().lower()
    hinted_action = str(evidence.get("processing_action_hint") or "").strip().lower()
    if hinted_action == PRESERVE or text_class in {
        "sfx",
        "onomatopoeia",
        "decorative",
        "decoration",
    }:
        return SemanticDecision(hinted_role or text_class or "sfx", PRESERVE)

    text = str(evidence.get("ocr_text") or "").strip()
    script = str(evidence.get("ocr_script") or evidence.get("script") or "").strip()
    confidence = evidence.get("ocr_confidence", evidence.get("confidence"))
    confidence_valid = (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
    )
    if not text or not script or not confidence_valid:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            reason="ocr_positive_text_evidence_absent",
        )

    ownership_role = {
        "text_bubble": "dialogue_bubble",
        "text_free": "dialogue_free",
        "narration": "narration",
        "caption": "narration",
    }.get(text_class)
    if ownership_role is None:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            reason="ocr_positive_text_ownership_absent",
        )
    if hinted_action and hinted_action != TRANSLATE:
        return SemanticDecision(
            "ambiguous",
            REVIEW,
            reason="ocr_action_hint_not_translate",
        )
    return SemanticDecision(hinted_role or ownership_role, TRANSLATE)
