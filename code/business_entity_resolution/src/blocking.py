"""Blocking / candidate generation.

Blockers (all unioned):

    exact_name     exact (country_norm, name_core)
    name_token     name token (frequency filtered)
    address_token  address token (frequency filtered)
    char3          shared character n-grams, >= CHAR3_MIN_SHARED
    phonetic       country-keyed consonant skeleton of the full name and of the
                   first meaningful token (bridges transliteration / typos)

``FROZEN_BLOCKERS`` reproduces the original candidate set. ``ADVANCED_BLOCKERS``
adds ``char3`` and ``phonetic`` to raise the recall ceiling.

Country is only ever used as an exact normalised key, never as a hard-coded
list, so unseen test countries work unchanged.
"""

import gc
from collections import Counter, defaultdict

import pandas as pd

from .config import (
    CHUNK_SIZE,
    CHAR3_MAX_FREQ,
    CHAR3_MIN_FREQ,
    CHAR3_MIN_SHARED,
    CHAR3_N,
    FROZEN_BLOCKERS,
    MAX_PHONETIC_BLOCK,
    MAX_TOKEN_FREQ,
    MIN_TOKEN_FREQ,
)
from .text_utils import (
    char_ngrams_set,
    consonant_skeleton,
    first_meaningful_token,
    meaningful_tokens,
)

__all__ = [
    "meaningful_tokens",
    "allowed_from_counter",
    "token_counter",
    "build_name_index",
    "build_token_index",
    "build_blocker",
    "candidates_by_blocker",
    "candidates_for_record",
    "iter_candidates",
]


def allowed_from_counter(counter, min_freq=MIN_TOKEN_FREQ, max_freq=MAX_TOKEN_FREQ):
    return {tok for tok, count in counter.items() if min_freq <= count <= max_freq}


def _iter_column(path, columns, chunk_size):
    for chunk in pd.read_csv(
        path, sep="\t", usecols=columns, chunksize=chunk_size, dtype="string"
    ):
        yield chunk
        del chunk
        gc.collect()


def token_counter(path, column, tokenizer, chunk_size=CHUNK_SIZE):
    counter = Counter()
    for chunk in _iter_column(path, ["entity_id", column], chunk_size):
        for value in chunk[column].dropna():
            counter.update(set(tokenizer(value)))
    return counter


def build_name_index(path, chunk_size=CHUNK_SIZE):
    index = defaultdict(list)
    for chunk in _iter_column(path, ["entity_id", "country_norm", "name_core"], chunk_size):
        for eid, country, name in zip(
            chunk["entity_id"], chunk["country_norm"], chunk["name_core"]
        ):
            if pd.isna(name) or str(name).strip() == "":
                continue
            index[(country, name)].append(eid)
    return index


def build_token_index(path, column, allowed, tokenizer, chunk_size=CHUNK_SIZE):
    index = defaultdict(list)
    for chunk in _iter_column(path, ["entity_id", column], chunk_size):
        for eid, value in zip(chunk["entity_id"], chunk[column]):
            if pd.isna(value):
                continue
            for token in set(tokenizer(value)):
                if token in allowed:
                    index[token].append(eid)
    return index


def build_char3_index(path, allowed, chunk_size=CHUNK_SIZE):
    index = defaultdict(list)
    for chunk in _iter_column(path, ["entity_id", "name_core"], chunk_size):
        for eid, name in zip(chunk["entity_id"], chunk["name_core"]):
            if pd.isna(name):
                continue
            for gram in char_ngrams_set(name, CHAR3_N):
                if gram in allowed:
                    index[gram].append(eid)
    return index


def build_skeleton_name_index(path, chunk_size=CHUNK_SIZE):
    """(country_norm, full-name consonant skeleton) -> ids."""
    index = defaultdict(list)
    for chunk in _iter_column(path, ["entity_id", "country_norm", "name_core"], chunk_size):
        for eid, country, name in zip(
            chunk["entity_id"], chunk["country_norm"], chunk["name_core"]
        ):
            skeleton = consonant_skeleton(name)
            if skeleton:
                index[(country, skeleton)].append(eid)
    return index


def build_first_token_skeleton_index(path, chunk_size=CHUNK_SIZE):
    """(country_norm, first-token consonant skeleton) -> ids."""
    index = defaultdict(list)
    for chunk in _iter_column(path, ["entity_id", "country_norm", "name_core"], chunk_size):
        for eid, country, name in zip(
            chunk["entity_id"], chunk["country_norm"], chunk["name_core"]
        ):
            skeleton = consonant_skeleton(first_meaningful_token(name))
            if skeleton:
                index[(country, skeleton)].append(eid)
    return index


def build_blocker(path, source_tag, enabled=None, chunk_size=CHUNK_SIZE):
    """Build every index needed to block one source (S2 or S3)."""
    enabled = list(enabled) if enabled is not None else list(FROZEN_BLOCKERS)
    print(f"building blocker for {source_tag} from {path} | blockers={enabled}")

    blocker = {"source": source_tag, "enabled": enabled}

    if "exact_name" in enabled:
        blocker["exact_name_index"] = build_name_index(path, chunk_size)

    if "name_token" in enabled:
        counter = token_counter(path, "name_core", meaningful_tokens, chunk_size)
        blocker["allowed_name_tokens"] = allowed_from_counter(counter)
        blocker["name_token_index"] = build_token_index(
            path, "name_core", blocker["allowed_name_tokens"], meaningful_tokens, chunk_size
        )

    if "address_token" in enabled:
        counter = token_counter(path, "address_norm", lambda a: str(a).split(), chunk_size)
        blocker["allowed_address_tokens"] = allowed_from_counter(counter)
        blocker["address_token_index"] = build_token_index(
            path, "address_norm", blocker["allowed_address_tokens"],
            lambda a: str(a).split(), chunk_size,
        )

    if "char3" in enabled:
        counter = token_counter(path, "name_core", lambda n: char_ngrams_set(n, CHAR3_N), chunk_size)
        blocker["allowed_char3"] = allowed_from_counter(
            counter, CHAR3_MIN_FREQ, CHAR3_MAX_FREQ
        )
        blocker["char3_index"] = build_char3_index(path, blocker["allowed_char3"], chunk_size)
        blocker["char3_min_shared"] = CHAR3_MIN_SHARED

    if "phonetic" in enabled:
        blocker["skeleton_name_index"] = build_skeleton_name_index(path, chunk_size)
        blocker["first_token_skeleton_index"] = build_first_token_skeleton_index(path, chunk_size)

    print(
        f"  {source_tag}: "
        + ", ".join(
            f"{k}={len(blocker[k])}"
            for k in (
                "exact_name_index",
                "name_token_index",
                "address_token_index",
                "char3_index",
                "skeleton_name_index",
                "first_token_skeleton_index",
            )
            if k in blocker
        )
    )
    return blocker


def candidates_by_blocker(country, name, address, blocker):
    """Return {blocker_name: set(candidate ids)} for one S1 record."""
    results = {}
    has_name = name is not None and not pd.isna(name) and str(name).strip() != ""
    has_addr = address is not None and not pd.isna(address) and str(address).strip() != ""

    if "exact_name_index" in blocker and has_name:
        results["exact_name"] = set(blocker["exact_name_index"].get((country, name), []))

    if "name_token_index" in blocker and has_name:
        found = set()
        for token in set(meaningful_tokens(name)):
            if token in blocker["allowed_name_tokens"]:
                found.update(blocker["name_token_index"].get(token, []))
        results["name_token"] = found

    if "address_token_index" in blocker and has_addr:
        found = set()
        for token in set(str(address).split()):
            if token in blocker["allowed_address_tokens"]:
                found.update(blocker["address_token_index"].get(token, []))
        results["address_token"] = found

    if "char3_index" in blocker and has_name:
        counts = Counter()
        for gram in char_ngrams_set(name, CHAR3_N):
            if gram in blocker["allowed_char3"]:
                for cid in blocker["char3_index"].get(gram, []):
                    counts[cid] += 1
        min_shared = blocker.get("char3_min_shared", CHAR3_MIN_SHARED)
        results["char3"] = {cid for cid, n in counts.items() if n >= min_shared}

    if "skeleton_name_index" in blocker and has_name:
        found = set()
        found.update(blocker["skeleton_name_index"].get((country, consonant_skeleton(name)), []))
        fskel = consonant_skeleton(first_meaningful_token(name))
        if fskel:
            same_key = blocker["first_token_skeleton_index"].get((country, fskel), [])
            if len(same_key) <= MAX_PHONETIC_BLOCK:
                found.update(same_key)
        results["phonetic"] = found

    return results


def candidates_for_record(country, name, address, blocker):
    """Union of every enabled blocker for one S1 record."""
    found = set()
    for value in candidates_by_blocker(country, name, address, blocker).values():
        found.update(value)
    return found


def iter_candidates(s1_path, blockers, chunk_size=CHUNK_SIZE):
    """Yield (s1_entity_id, candidate_entity_id, source)."""
    for chunk in _iter_column(
        s1_path, ["entity_id", "country_norm", "name_core", "address_norm"], chunk_size
    ):
        for eid, country, name, address in zip(
            chunk["entity_id"], chunk["country_norm"],
            chunk["name_core"], chunk["address_norm"],
        ):
            for source_tag, blocker in blockers.items():
                for cand_id in candidates_for_record(country, name, address, blocker):
                    yield eid, cand_id, source_tag
