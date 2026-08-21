"""A bounded two-engine verifier for difficult crops."""

from __future__ import annotations

from typing import Any

from .corrector import normalize_russian_text
from .recognizer import CyrillicRecognizer


class EnsembleRecognizer:
    """Run a second Cyrillic recognizer only when the primary is uncertain.

    Agreement is evidence, not a guarantee. When the engines disagree the
    result is explicitly marked uncertain so the APK can ask for a rescan.
    """

    def __init__(self, primary: CyrillicRecognizer, secondary: CyrillicRecognizer) -> None:
        self.primary = primary
        self.secondary = secondary

    def recognize_consensus(
        self,
        image,
        *,
        target_confidence: float = 0.90,
        max_passes: int = 4,
    ) -> dict[str, Any]:
        primary = self.primary.recognize_consensus(
            image,
            target_confidence=target_confidence,
            max_passes=max_passes,
        )
        # Fast path: do not spend a second model pass on an already stable crop.
        if primary["certain"] and primary["confidence"] >= target_confidence:
            primary["engine"] = "ppocrv3"
            primary["engines_agree"] = True
            return primary

        secondary = self.secondary.recognize_consensus(
            image,
            target_confidence=target_confidence,
            max_passes=max_passes,
        )
        p_text = normalize_russian_text(primary["text"], allow_dictionary=True)
        s_text = normalize_russian_text(secondary["text"], allow_dictionary=True)
        same = bool(p_text) and p_text.casefold() == s_text.casefold()

        if same:
            score = min(0.999, 0.5 * (float(primary["confidence"]) + float(secondary["confidence"])) + 0.12)
            return {
                "text": p_text,
                "confidence": round(score, 4),
                "certain": bool(score >= target_confidence),
                "passes": max(int(primary["passes"]), int(secondary["passes"])),
                "engine": "ppocrv3+ppocrv5",
                "engines_agree": True,
                "evidence": {"ppocrv3": primary, "ppocrv5": secondary},
            }

        # If outputs disagree, retain the stronger visual candidate but never
        # promote it to certain. This is the important abstention behaviour.
        chosen = primary if float(primary["confidence"]) >= float(secondary["confidence"]) else secondary
        return {
            "text": normalize_russian_text(chosen["text"], allow_dictionary=True),
            "confidence": round(float(chosen["confidence"]) * 0.85, 4),
            "certain": False,
            "passes": max(int(primary["passes"]), int(secondary["passes"])),
            "engine": "ppocrv3+ppocrv5",
            "engines_agree": False,
            "evidence": {"ppocrv3": primary, "ppocrv5": secondary},
        }
