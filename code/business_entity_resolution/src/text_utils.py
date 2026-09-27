"""Text cleaning, string similarity, transliteration and core-name helpers.

All functions are pure and dependency-light so they can be reused by blocking,
feature engineering and inference.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import CORE_SUFFIXES, NAME_STOPWORDS

NUMBER_RE = re.compile(r"\d+")

try:  # indic-transliteration is optional; fall back to pass-through.
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    HAS_TRANSLIT = True
except Exception:  # pragma: no cover - depends on environment
    HAS_TRANSLIT = False

SCRIPT_MAP = {}
if HAS_TRANSLIT:
    SCRIPT_MAP = {
        "DEVANAGARI": sanscript.DEVANAGARI,
        "BENGALI": sanscript.BENGALI,
        "GURMUKHI": sanscript.GURMUKHI,
        "GUJARATI": sanscript.GUJARATI,
        "ORIYA": sanscript.ORIYA,
        "TAMIL": sanscript.TAMIL,
        "TELUGU": sanscript.TELUGU,
        "KANNADA": sanscript.KANNADA,
        "MALAYALAM": sanscript.MALAYALAM,
    }


# ============================================================
# basic text
# ============================================================

def clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def char_similarity(a, b):
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return np.nan
    return fuzz.ratio(a, b) / 100.0


def token_jaccard(a, b):
    a_tokens = set(clean_text(a).split())
    b_tokens = set(clean_text(b).split())
    if not a_tokens or not b_tokens:
        return np.nan
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def ngram_jaccard(a, b, n=3):
    a = clean_text(a).replace(" ", "")
    b = clean_text(b).replace(" ", "")
    if not a or not b:
        return np.nan
    a_grams = {a[i:i + n] for i in range(max(1, len(a) - n + 1))}
    b_grams = {b[i:i + n] for i in range(max(1, len(b) - n + 1))}
    if not a_grams or not b_grams:
        return np.nan
    return len(a_grams & b_grams) / len(a_grams | b_grams)


def jaccard_sets(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def numbers_of(text):
    return set(NUMBER_RE.findall(clean_text(text)))


def length_ratio(a, b):
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def token_count(text):
    return len(clean_text(text).split())


# ============================================================
# transliteration (single pass)
# ============================================================

def detect_script(text):
    if pd.isna(text):
        return None
    for ch in str(text):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for script_name in SCRIPT_MAP:
            if script_name in name:
                return script_name
    return None


def transliterate_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if not HAS_TRANSLIT:
        return text.lower()
    script = detect_script(text)
    if script is None:
        return text.lower()
    try:
        return transliterate(text, SCRIPT_MAP[script], sanscript.ITRANS)
    except Exception:
        return text.lower()


def normalize_transliterated(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_transliteration(text):
    """detect script -> transliterate once -> normalise (never repeats)."""
    return normalize_transliterated(transliterate_text(text))


_TRANSLIT_CACHE = {}


def get_translit(name):
    name = clean_text(name)
    if name in _TRANSLIT_CACHE:
        return _TRANSLIT_CACHE[name]
    value = normalize_transliteration(name)
    _TRANSLIT_CACHE[name] = value
    return value


# ============================================================
# tokenisation and phonetic keys (for the improved blockers)
# ============================================================

def meaningful_tokens(name):
    """Tokens of a name with stop-words and single characters removed."""
    if name is None or pd.isna(name) or not str(name).strip():
        return []
    return [
        token
        for token in str(name).split()
        if token not in NAME_STOPWORDS and len(token) >= 2
    ]


# Consonant classes collapse spellings that transliteration or typos vary
# (e.g. g/j, c/k/q, s/z, v/w). Vowels and 'y' are dropped so that
# "green" -> "jrn" and its Indic transliteration "grina" -> "jrn".
_PHONETIC_MAP = {
    "b": "b", "p": "p",
    "c": "k", "k": "k", "q": "k", "x": "k",
    "g": "j", "j": "j",
    "s": "s", "z": "s",
    "v": "v", "w": "v",
    "f": "f", "h": "h",
    "d": "d", "t": "t",
    "n": "n", "m": "m",
    "r": "r", "l": "l",
}


def consonant_skeleton(text):
    """Ordered consonant skeleton used as a phonetic blocking key."""
    letters = re.sub(r"[^a-z]", "", clean_text(text).lower())
    out = []
    prev = ""
    for ch in letters:
        if ch in "aeiouy":
            prev = ""
            continue
        mapped = _PHONETIC_MAP.get(ch, "")
        if not mapped or mapped == prev:
            continue
        out.append(mapped)
        prev = mapped
    return "".join(out)


def char_ngrams_set(text, n=3):
    text = clean_text(text).replace(" ", "")
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def first_meaningful_token(name):
    tokens = meaningful_tokens(name)
    if tokens:
        return tokens[0]
    tokens = clean_text(name).split()
    return tokens[0] if tokens else ""


# ============================================================
# core name (legal suffix stripped)
# ============================================================

# Matches one or more trailing legal-suffix words:
# "green logistics pvt ltd" -> "green logistics".
CORE_SUFFIX_PATTERN = (
    r"(\s+\b("
    + "|".join(
        sorted((re.escape(s) for s in CORE_SUFFIXES), key=len, reverse=True)
    )
    + r")\b)+$"
)


def core_of(name):
    result = clean_text(name)
    result = re.sub(CORE_SUFFIX_PATTERN, "", result)
    return re.sub(r"\s+", " ", result).strip()


def create_name_core(name_series):
    result = pd.Series(name_series, dtype="string").fillna("").str.strip()
    result = result.str.replace(CORE_SUFFIX_PATTERN, "", regex=True)
    return result.str.replace(r"\s+", " ", regex=True).str.strip()
