"""Memory-bounded ("reverse") candidate generation.

The original design indexes the large noisy sources (S2/S3) and keeps their full
text in RAM. On Colab that is the memory that blows up: the source posting lists
plus the source text lookup together run to several GB per source.

This module inverts the join. Only the **reference side (S1)** is indexed and
held in memory; S2/S3 are streamed chunk by chunk. For every source record the
blocking keys are computed once and looked up against the S1 indices, so:

* no source posting lists
* no source text lookup
* peak memory is driven by S1 alone (the deduplicated reference)

The blockers are symmetric, so the candidate pairs are identical to the
source-indexed version.
"""

import gc
import os
from collections import Counter

import pandas as pd

from .blocking import (
    allowed_from_counter,
    build_char3_index,
    build_first_token_skeleton_index,
    build_name_index,
    build_skeleton_name_index,
    build_token_index,
    candidates_by_blocker,
    candidates_for_record,
    token_counter,
)
from .config import (
    CHAR3_MAX_FREQ,
    CHAR3_MIN_FREQ,
    CHAR3_MIN_SHARED,
    CHAR3_N,
    CHUNK_SIZE,
    MAX_TOKEN_FREQ,
    MIN_TOKEN_FREQ,
)
from .features import build_text_lookup
from .text_utils import char_ngrams_set, meaningful_tokens

SOURCE_COLUMNS = [
    "entity_id",
    "country_norm",
    "name_norm",
    "name_core",
    "address_norm",
]


def build_allowed_sets(source_paths, chunk_size=CHUNK_SIZE):
    """Frequency-filtered vocabulary, learned once from the source corpora.

    Only counters are held (no ids), so this pass is cheap.
    """
    name_counter = Counter()
    addr_counter = Counter()
    char3_counter = Counter()

    for path in source_paths.values():
        if not os.path.exists(path):
            continue
        name_counter.update(
            token_counter(path, "name_core", meaningful_tokens, chunk_size)
        )
        addr_counter.update(
            token_counter(path, "address_norm", lambda a: str(a).split(), chunk_size)
        )
        char3_counter.update(
            token_counter(
                path, "name_core", lambda n: char_ngrams_set(n, CHAR3_N), chunk_size
            )
        )

    allowed_name = allowed_from_counter(name_counter, MIN_TOKEN_FREQ, MAX_TOKEN_FREQ)
    allowed_addr = allowed_from_counter(addr_counter, MIN_TOKEN_FREQ, MAX_TOKEN_FREQ)
    allowed_char3 = allowed_from_counter(char3_counter, CHAR3_MIN_FREQ, CHAR3_MAX_FREQ)

    print(
        f"allowed vocabulary: name_tokens={len(allowed_name):,} "
        f"address_tokens={len(allowed_addr):,} char3={len(allowed_char3):,}"
    )
    return allowed_name, allowed_addr, allowed_char3


def build_s1_blocker(
    s1_path,
    allowed_name,
    allowed_addr,
    allowed_char3,
    enabled,
    chunk_size=CHUNK_SIZE,
):
    """Index the reference side (S1) and keep its text in memory."""
    enabled = list(enabled)
    print(f"building S1 blocker | blockers={enabled}")

    blocker = {
        "source": "S1",
        "enabled": enabled,
        # S1 is the small side, so its full text fits comfortably.
        "text": build_text_lookup(s1_path, chunk_size=chunk_size),
    }

    if "exact_name" in enabled:
        blocker["exact_name_index"] = build_name_index(s1_path, chunk_size)

    if "name_token" in enabled:
        blocker["allowed_name_tokens"] = allowed_name
        blocker["name_token_index"] = build_token_index(
            s1_path, "name_core", allowed_name, meaningful_tokens, chunk_size
        )

    if "address_token" in enabled:
        blocker["allowed_address_tokens"] = allowed_addr
        blocker["address_token_index"] = build_token_index(
            s1_path, "address_norm", allowed_addr, lambda a: str(a).split(), chunk_size
        )

    if "char3" in enabled:
        blocker["allowed_char3"] = allowed_char3
        blocker["char3_index"] = build_char3_index(s1_path, allowed_char3, chunk_size)
        blocker["char3_min_shared"] = CHAR3_MIN_SHARED

    if "phonetic" in enabled:
        blocker["skeleton_name_index"] = build_skeleton_name_index(s1_path, chunk_size)
        blocker["first_token_skeleton_index"] = build_first_token_skeleton_index(
            s1_path, chunk_size
        )

    print("  S1 records:", len(blocker["text"]))
    return blocker


def iter_pairs(source_paths, s1_blocker, chunk_size=CHUNK_SIZE):
    """Stream the sources and yield (s1_id, cand_id, s1_tuple, cand_tuple, tag).

    The source row is already in hand, so no source lookup is ever needed.
    """
    text = s1_blocker["text"]

    for tag, path in source_paths.items():
        if not os.path.exists(path):
            continue

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=SOURCE_COLUMNS,
            chunksize=chunk_size,
            dtype="string",
        ):
            for eid, country, name_norm, name_core, address in zip(
                chunk["entity_id"],
                chunk["country_norm"],
                chunk["name_norm"],
                chunk["name_core"],
                chunk["address_norm"],
            ):
                cand_ids = candidates_for_record(country, name_core, address, s1_blocker)
                if not cand_ids:
                    continue

                cand_tuple = (country, name_norm, name_core, address)
                for s1_id in cand_ids:
                    s1_tuple = text.get(s1_id)
                    if s1_tuple is None:
                        continue
                    yield s1_id, eid, s1_tuple, cand_tuple, tag

            del chunk
            gc.collect()


def iter_pairs_by_blocker(source_paths, s1_blocker, chunk_size=CHUNK_SIZE):
    """Like :func:`iter_pairs` but also returns the per-blocker candidate sets.

    Used by the evaluation to attribute recall to individual blockers.
    """
    for tag, path in source_paths.items():
        if not os.path.exists(path):
            continue

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=SOURCE_COLUMNS,
            chunksize=chunk_size,
            dtype="string",
        ):
            for eid, country, name_norm, name_core, address in zip(
                chunk["entity_id"],
                chunk["country_norm"],
                chunk["name_norm"],
                chunk["name_core"],
                chunk["address_norm"],
            ):
                by = candidates_by_blocker(country, name_core, address, s1_blocker)
                if not by:
                    continue
                cand_tuple = (country, name_norm, name_core, address)
                yield eid, tag, cand_tuple, by

            del chunk
            gc.collect()
