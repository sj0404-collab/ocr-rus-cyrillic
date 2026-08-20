"""Conservative Russian post-processing for OCR output.

The recognizer is still the source of visual evidence. This module only fixes
common Latin/Cyrillic look-alikes and proposes a one-edit Russian dictionary
correction when the morphology dictionary strongly supports it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# Only visually plausible confusions are mapped. Unknown Latin is not silently
# converted to a random Russian letter.
CONFUSABLES = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "k": "к", "m": "м", "o": "о",
    "p": "р", "t": "т", "x": "х", "y": "у", "i": "и", "j": "й",
    "u": "и", "b": "в",
})

RU_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
ALLOWED_PUNCTUATION = " .,!?;:-()[]{}\"'«»„“”%№+/=…—–"
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")

try:  # Optional on Android; the visual-only mode remains usable without it.
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    MorphAnalyzer = None  # type: ignore[assignment,misc]

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None  # type: ignore[assignment]

_MORPH = MorphAnalyzer() if MorphAnalyzer is not None else None


def map_confusables(text: str) -> str:
    """Convert common Latin look-alikes to their Cyrillic counterparts."""
    return text.translate(CONFUSABLES)


def _is_russian_word(word: str) -> bool:
    if not word or not any(ch in RU_LETTERS for ch in word.lower()):
        return False
    if _MORPH is None:
        return True
    parsed = _MORPH.parse(word.lower())
    return bool(parsed and parsed[0].tag.POS is not None)


@lru_cache(maxsize=4096)
def _morph_score(word: str) -> float:
    if _MORPH is None:
        return 0.0
    parsed = _MORPH.parse(word.lower())
    if not parsed or parsed[0].tag.POS is None:
        return 0.0
    return float(parsed[0].score)


def _edits(word: str):
    """Generate candidates at Levenshtein distance one."""
    lower = word.lower()
    alphabet = RU_LETTERS
    seen: set[str] = set()

    for i in range(len(lower)):
        candidate = lower[:i] + lower[i + 1 :]
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for i in range(len(lower) + 1):
        for ch in alphabet:
            candidate = lower[:i] + ch + lower[i:]
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    for i, old in enumerate(lower):
        for ch in alphabet:
            if ch == old:
                continue
            candidate = lower[:i] + ch + lower[i + 1 :]
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    for i in range(len(lower) - 1):
        if lower[i] == lower[i + 1]:
            continue
        candidate = lower[:i] + lower[i + 1] + lower[i] + lower[i + 2 :]
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _restore_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def correct_word(word: str, *, allow_dictionary: bool = True) -> str:
    """Return a sanitized Russian spelling, conservatively.

    The correction is deliberately limited to one edit. A separate OCR agent
    should be used when the visual candidate remains ambiguous.
    """
    if not word:
        return word
    mapped = map_confusables(word)
    cyr = "".join(ch for ch in mapped if ch in RU_LETTERS.upper() or ch in RU_LETTERS)
    if not cyr:
        return "".join(ch for ch in mapped if ch.isdigit() or ch in ALLOWED_PUNCTUATION)

    source_case = cyr
    lower = cyr.lower()
    if not allow_dictionary or _MORPH is None:
        return _restore_case(source_case, lower)

    base_score = _morph_score(lower)
    if base_score >= 0.20:
        return _restore_case(source_case, lower)

    best: Optional[tuple[float, str]] = None
    # The edit search is used only for unknown/low-confidence tokens.
    for candidate in _edits(lower):
        score = _morph_score(candidate)
        if score <= 0.0:
            continue
        frequency = 0.0
        if zipf_frequency is not None:
            frequency = min(1.0, max(0.0, zipf_frequency(candidate, "ru") / 6.0))
        # Morphology rejects nonsense; frequency breaks ties between valid but
        # rare forms (for example, "проверка" versus a rare technical noun).
        rank = 0.62 * score + 0.38 * frequency
        rank -= 0.015 * abs(len(candidate) - len(lower))
        if best is None or rank > best[0]:
            best = (rank, candidate)

    if best is not None and best[0] >= max(0.35, base_score + 0.15):
        return _restore_case(source_case, best[1])
    return _restore_case(source_case, lower)


def normalize_russian_text(text: str, *, allow_dictionary: bool = True) -> str:
    """Normalize an OCR string without inventing unobserved content."""
    text = map_confusables(text)
    parts: list[str] = []
    pos = 0
    for match in _WORD_RE.finditer(text):
        parts.append(text[pos : match.start()])
        parts.append(correct_word(match.group(0), allow_dictionary=allow_dictionary))
        pos = match.end()
    parts.append(text[pos:])
    result = "".join(parts)

    # Remove unsupported Latin and control characters while keeping digits and
    # punctuation useful in Russian documents.
    result = "".join(
        ch for ch in result
        if ch in RU_LETTERS.upper() or ch in RU_LETTERS or ch.isdigit()
        or ch.isspace() or ch in ALLOWED_PUNCTUATION
    )
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\s+([,.;:!?%])", r"\1", result)
    return result.strip()
