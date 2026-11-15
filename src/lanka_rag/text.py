"""Text handling for Sinhala, Tamil and English.

Sinhala and Tamil text has a few properties that silently break naive pipelines:

* Vowel signs are combining marks (Unicode category Mn/Mc). Python's ``\\w`` and
  ``str.isalnum()`` do not treat them as word characters, so a regex tokenizer
  splits words in the middle (for example ``ලංකා`` -> ``ල``, ``ක``).
* The zero width joiner (U+200D) is part of Sinhala spelling: yansaya (``්‍ය``),
  rakaransaya (``්‍ර``) and touching letters. The same word is routinely typed with
  and without it (``ශ්‍රී`` vs ``ශ්රී``), which makes exact matching unreliable.
* Composite vowel signs have two encodings (``ො`` U+0DDC vs ``ෙ`` + ``ා``).
  NFC folds these together.
* Many Sri Lankan government PDFs use pre-Unicode "legacy" fonts (FM Abhaya,
  DL-Manel, ...). Extracted text then looks like Latin garbage, not Sinhala.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

ZWJ = chr(0x200D)
ZWNJ = chr(0x200C)
ZWSP = chr(0x200B)
BOM = chr(0xFEFF)
KUNDDALIYA = chr(0x0DF4)  # ෴, the traditional Sinhala full stop

SINHALA_RANGE = (0x0D80, 0x0DFF)
TAMIL_RANGE = (0x0B80, 0x0BFF)

_WS_RE = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_STRAY_ZWJ_RE = re.compile("(?<!\\u0dca)\\u200d")  # ZWJ not preceded by the Sinhala virama


def _in(ch: str, rng: tuple[int, int]) -> bool:
    return rng[0] <= ord(ch) <= rng[1]


def normalize(text: str, *, strip_zwj: bool = False) -> str:
    """Normalize text for storage and embedding.

    Always: NFC, drop BOM / zero width space / ZWNJ, collapse horizontal
    whitespace, trim runs of blank lines.

    ``strip_zwj`` removes U+200D as well. That loses the visual conjunct form
    (``ශ්‍රී`` renders as ``ශ්රී``) but makes both spellings identical, which is what
    lexical matching wants. Keep it off for text you show to people.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(BOM, "").replace(ZWSP, "").replace(ZWNJ, "")
    # A ZWJ only means something right after a virama (al-lakuna); strays are noise.
    text = text.replace(ZWJ, "") if strip_zwj else _STRAY_ZWJ_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _is_token_char(ch: str) -> bool:
    # Letters, combining marks (Sinhala/Tamil vowel signs, virama) and digits.
    return unicodedata.category(ch)[0] in "LMN"


def tokenize(text: str) -> list[str]:
    """Split into lexical tokens without breaking Sinhala/Tamil words apart.

    Lowercases Latin script, removes ZWJ so both spellings of a conjunct match,
    and treats every punctuation or symbol character as a separator.
    """
    text = normalize(text, strip_zwj=True).lower()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_token_char(ch):
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    # A token made only of combining marks is an artefact of bad text extraction.
    return [t for t in tokens if any(unicodedata.category(c)[0] in "LN" for c in t)]


def char_ngrams(text: str, n_values: tuple[int, ...] = (3, 4)) -> list[str]:
    """Character n-grams per word, with word boundary markers.

    Sinhala and Tamil are heavily suffixing (case markers, postpositions), so
    ``ලංකාවේ`` and ``ලංකාව`` share no whole-word token but share most n-grams.
    """
    grams: list[str] = []
    for tok in tokenize(text):
        padded = f"<{tok}>"
        for n in n_values:
            if len(padded) < n:
                continue
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


@dataclass(frozen=True)
class ScriptStats:
    sinhala: int
    tamil: int
    latin: int
    other: int

    @property
    def total(self) -> int:
        return self.sinhala + self.tamil + self.latin + self.other

    @property
    def dominant(self) -> str:
        """``si``, ``ta``, ``en`` (any Latin script, including Singlish) or ``unknown``."""
        if self.total == 0:
            return "unknown"
        counts = {"si": self.sinhala, "ta": self.tamil, "en": self.latin}
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] / self.total >= 0.5 else "unknown"


def script_stats(text: str) -> ScriptStats:
    si = ta = la = other = 0
    for ch in text:
        if not ch.isalpha() and unicodedata.category(ch)[0] != "M":
            continue
        if _in(ch, SINHALA_RANGE):
            si += 1
        elif _in(ch, TAMIL_RANGE):
            ta += 1
        elif ch.isascii() or unicodedata.name(ch, "").startswith("LATIN"):
            la += 1
        else:
            other += 1
    return ScriptStats(si, ta, la, other)


def detect_language(text: str) -> str:
    return script_stats(text).dominant


LANGUAGE_NAMES = {"si": "Sinhala", "ta": "Tamil", "en": "English"}

# Characters that legacy Sinhala fonts map to letters but that almost never
# occur *inside* a word in real English text.
_LEGACY_MARKERS = set("%&;{}|][\\^~`")


def legacy_font_score(text: str) -> float:
    """Heuristic 0..1 score that ``text`` is Sinhala typed in a legacy font.

    Text extracted from an FM Abhaya / DL-Manel PDF contains no Sinhala code
    points, but lots of punctuation glued to letters (``Y%S ,xld`` is how
    "ශ්‍රී ලංකා" comes out). Treat anything above ~0.3 as needing conversion or OCR
    before it can be indexed.
    """
    stats = script_stats(text)
    if stats.sinhala or stats.tamil or stats.latin < 40:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    glued = sum(
        1 for w in words if any(c in _LEGACY_MARKERS for c in w) and any(c.isalpha() for c in w)
    )
    # Legacy text also has unusual letter distributions: very few vowels.
    letters = [c.lower() for c in text if c.isascii() and c.isalpha()]
    vowel_ratio = sum(c in "aeiou" for c in letters) / max(len(letters), 1)
    glued_ratio = glued / len(words)
    score = min(1.0, glued_ratio * 3)
    if vowel_ratio < 0.25:
        score = min(1.0, score + 0.3)
    return round(score, 3)


def top_tokens(texts: list[str], k: int = 20) -> list[tuple[str, int]]:
    """Most frequent tokens, handy for eyeballing a corpus."""
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter.most_common(k)
