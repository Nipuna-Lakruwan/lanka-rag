import re

from lanka_rag.text import (
    ZWJ,
    char_ngrams,
    detect_language,
    legacy_font_score,
    normalize,
    tokenize,
)

SRI_LANKA_ZWJ = "ශ්‍රී ලංකාව"
SRI_LANKA_NO_ZWJ = "ශ්රී ලංකාව"


def test_regex_word_class_breaks_sinhala():
    # Documents the pitfall the tokenizer exists to avoid.
    assert re.findall(r"\w+", "ජනාධිපතිවරයා") != ["ජනාධිපතිවරයා"]


def test_tokenize_keeps_sinhala_and_tamil_words_whole():
    assert tokenize("ශ්‍රී ලංකාවේ ජනාධිපතිවරයා, කොළඹදී!") == [
        "ශ්රී",
        "ලංකාවේ",
        "ජනාධිපතිවරයා",
        "කොළඹදී",
    ]
    assert tokenize("இலங்கையின் தலைநகரம் கொழும்பு.") == ["இலங்கையின்", "தலைநகரம்", "கொழும்பு"]


def test_zwj_and_no_zwj_spellings_match_lexically():
    assert tokenize(SRI_LANKA_ZWJ) == tokenize(SRI_LANKA_NO_ZWJ)


def test_normalize_keeps_meaningful_zwj_and_drops_stray_ones():
    assert ZWJ in normalize(SRI_LANKA_ZWJ)
    assert ZWJ not in normalize(f"ලංකා{ZWJ}ව")  # not after a virama
    assert ZWJ not in normalize(SRI_LANKA_ZWJ, strip_zwj=True)


def test_normalize_composes_split_vowel_signs():
    # කො written as ක + ෙ + ා must equal the precomposed ො (U+0DDC)
    assert normalize("කො") == "කො"


def test_normalize_whitespace():
    assert normalize("  a  b\n\n\n\nc​ ") == "a b\n\nc"


def test_char_ngrams_share_stem_across_suffixes():
    a, b = set(char_ngrams("ලංකාවේ")), set(char_ngrams("ලංකාව"))
    assert len(a & b) >= 4


def test_detect_language():
    assert detect_language("ශ්‍රී ලංකාව") == "si"
    assert detect_language("இலங்கை") == "ta"
    assert detect_language("mama gedara yanawa") == "en"  # Singlish is Latin script
    assert detect_language("123 !!") == "unknown"


def test_legacy_font_detection():
    legacy = (
        "Y%S ,xld m%cd;dka;%sl iudcjd§ ckrcfha .eiÜ m;%h wxl 2345$12 - 2023 cqks ui 05 jeks "
        "i÷od - 2023'"
    )
    english = (
        "The Gazette of the Democratic Socialist Republic of Sri Lanka, published by authority, "
        "notifies the appointment of officers."
    )
    assert legacy_font_score(legacy) >= 0.3
    assert legacy_font_score(english) < 0.3
    assert legacy_font_score("ශ්‍රී ලංකා ප්‍රජාතාන්ත්‍රික සමාජවාදී ජනරජයේ ගැසට් පත්‍රය") == 0.0
