"""Conservative Russian post-processing for OCR output.

The recognizer is still the source of visual evidence. This module only fixes
common Latin/Cyrillic look-alikes and proposes a one-edit Russian dictionary
correction when the morphology dictionary strongly supports it.
"""

from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path
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
# Digits occasionally replace a visually similar Cyrillic glyph in low-quality
# manga scans. They are mapped only inside a token that also contains letters.
DIGIT_CONFUSABLES = str.maketrans({"0": "о", "3": "з", "6": "б", "7": "т", "8": "в", "9": "а"})
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

try:  # Optional on Android; the visual-only mode remains usable without it.
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    MorphAnalyzer = None  # type: ignore[assignment,misc]

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None  # type: ignore[assignment]

_MORPH = MorphAnalyzer() if MorphAnalyzer is not None else None


def _load_opencorpora_frequency() -> dict[str, float]:
    path = Path(__file__).resolve().parents[2] / "models" / "dicts" / "opencorpora_freqrnc2011.tsv"
    if not path.exists():
        return {}
    result: dict[str, float] = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                word = (row.get("Lemma") or "").strip().lower()
                if not word:
                    continue
                try:
                    result[word] = max(result.get(word, 0.0), float(row.get("Freq(ipm") or row.get("Freq(ipm)") or 0.0))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return result


_LOCAL_FREQUENCY = _load_opencorpora_frequency()


def _frequency_score(word: str) -> float:
    scores: list[float] = []
    if zipf_frequency is not None:
        scores.append(min(1.0, max(0.0, zipf_frequency(word, "ru") / 6.0)))
    ipm = _LOCAL_FREQUENCY.get(word.lower(), 0.0)
    if ipm > 0:
        scores.append(min(1.0, math.log1p(ipm) / 9.0))
    return max(scores, default=0.0)


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
    if any(ch in RU_LETTERS.upper() or ch in RU_LETTERS for ch in mapped):
        mapped = mapped.translate(DIGIT_CONFUSABLES)
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
        frequency = _frequency_score(candidate)
        # Morphology rejects nonsense; frequency breaks ties between valid but
        # rare forms (for example, "проверка" versus a rare technical noun).
        rank = 0.62 * score + 0.38 * frequency
        rank -= 0.015 * abs(len(candidate) - len(lower))
        if best is None or rank > best[0]:
            best = (rank, candidate)

    if best is not None and best[0] >= max(0.35, base_score + 0.15):
        return _restore_case(source_case, best[1])
    return _restore_case(source_case, lower)


@lru_cache(maxsize=4096)
def _segment_concatenated_word(word: str) -> str:
    """Split an OCR-fused token when a strong Russian word path exists."""
    if _MORPH is None or (zipf_frequency is None and not _LOCAL_FREQUENCY) or len(word) < 9 or not word.isalpha():
        return word
    lower = word.lower()
    # Never split a token that the Russian dictionary already recognizes as a
    # plausible standalone word. This protects long words such as
    # "покровителем" and "сериализация" from false segmentation.
    if _morph_score(lower) >= 0.20 and _frequency_score(lower) >= 0.16:
        return word
    one_letter = {"а", "в", "и", "к", "о", "с", "у", "я"}
    short_words = {"а", "в", "и", "к", "о", "с", "у", "я", "же", "не", "но", "на", "по", "из", "за", "от", "до", "ко", "со"}

    def score(candidate: str) -> float:
        if len(candidate) == 1:
            return 0.10 if candidate in one_letter else -1.0
        morph = _morph_score(candidate)
        freq = _frequency_score(candidate)
        if len(candidate) < 4:
            if candidate not in short_words or freq < 0.50:
                return -1.0
        if morph <= 0.0 and freq < 0.50:
            return -1.0
        return 0.55 * min(1.0, morph / 0.5) + 0.45 * freq

    best: list[tuple[float, list[str]]] = [(-10**9, []) for _ in range(len(lower) + 1)]
    best[0] = (0.0, [])
    for start in range(len(lower)):
        if best[start][0] < -10**8:
            continue
        for end in range(start + 1, min(len(lower), start + 20) + 1):
            part = lower[start:end]
            part_score = score(part)
            if part_score < 0:
                continue
            boundary_penalty = (1.35 if len(part) == 1 else 0.95) if start else 0.0
            candidate_score = best[start][0] + part_score - boundary_penalty
            if candidate_score > best[end][0]:
                best[end] = (candidate_score, best[start][1] + [part])

    base = score(lower)
    total, pieces = best[-1]
    if len(pieces) <= 1 or total < base + 0.25:
        return word
    if word.isupper():
        pieces = [piece.upper() for piece in pieces]
    elif word[:1].isupper():
        pieces = [pieces[0].capitalize(), *pieces[1:]]
    return " ".join(pieces)


def normalize_russian_text(text: str, *, allow_dictionary: bool = True) -> str:
    """Normalize an OCR string without inventing unobserved content."""
    text = map_confusables(text)
    parts: list[str] = []
    pos = 0
    for match in _WORD_RE.finditer(text):
        parts.append(text[pos : match.start()])
        corrected = correct_word(match.group(0), allow_dictionary=allow_dictionary)
        if allow_dictionary:
            corrected = _segment_concatenated_word(corrected)
        parts.append(corrected)
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
