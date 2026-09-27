"""Pairwise feature construction.

``compute_pair_features`` is the single entry point used by training and by
inference so there is no train/serve skew.
"""

import gc

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .text_utils import (
    char_similarity,
    clean_text,
    core_of,
    get_translit,
    jaccard_sets,
    length_ratio,
    ngram_jaccard,
    numbers_of,
    token_count,
    token_jaccard,
)


def compute_pair_features(s1, cand, source):
    """Features for one (S1, candidate) pair.

    ``s1`` and ``cand`` are tuples of
    ``(country_norm, name_norm, name_core, address_norm)``.
    """
    s1_country, s1_name, s1_core, s1_addr = s1
    c_country, c_name, c_core, c_addr = cand

    s1_name = clean_text(s1_name)
    c_name = clean_text(c_name)
    s1_addr = clean_text(s1_addr)
    c_addr = clean_text(c_addr)

    s1_translit = get_translit(s1_name)
    c_translit = get_translit(c_name)
    s1_translit_core = core_of(s1_translit)
    c_translit_core = core_of(c_translit)

    s1_tokens = s1_name.split()
    c_tokens = c_name.split()

    s1_nums = numbers_of(s1_addr)
    c_nums = numbers_of(c_addr)

    source = (source or "").upper()

    return {
        "name_char_similarity": char_similarity(s1_name, c_name),
        "name_3gram_jaccard": ngram_jaccard(s1_name, c_name, 3),
        "name_token_jaccard": token_jaccard(s1_name, c_name),
        "address_char_similarity": char_similarity(s1_addr, c_addr),
        "address_token_jaccard": token_jaccard(s1_addr, c_addr),
        "country_exact": int(clean_text(s1_country) == clean_text(c_country)),
        "name_exact": int(s1_name != "" and s1_name == c_name),
        "name_core_exact": int(
            clean_text(s1_core) != "" and clean_text(s1_core) == clean_text(c_core)
        ),
        "name_length_ratio": length_ratio(s1_name, c_name),
        "name_token_count_diff": abs(token_count(s1_name) - token_count(c_name)),
        "name_first_token_match": int(
            bool(s1_tokens) and bool(c_tokens) and s1_tokens[0] == c_tokens[0]
        ),
        "name_last_token_match": int(
            bool(s1_tokens) and bool(c_tokens) and s1_tokens[-1] == c_tokens[-1]
        ),
        "name_contains": int(
            s1_name != ""
            and c_name != ""
            and s1_name != c_name
            and (s1_name in c_name or c_name in s1_name)
        ),
        "address_exact": int(s1_addr != "" and s1_addr == c_addr),
        "address_length_ratio": length_ratio(s1_addr, c_addr),
        "address_number_match": int(bool(s1_nums & c_nums)),
        "address_number_jaccard": jaccard_sets(s1_nums, c_nums),
        "name_translit_exact": int(s1_translit != "" and s1_translit == c_translit),
        "name_translit_core_exact": int(
            s1_translit_core != "" and s1_translit_core == c_translit_core
        ),
        "name_translit_char_similarity": char_similarity(s1_translit, c_translit),
        "name_translit_3gram_jaccard": ngram_jaccard(s1_translit, c_translit, 3),
        "name_translit_token_jaccard": token_jaccard(s1_translit, c_translit),
        "source_is_s2": int(source == "S2"),
        "source_is_s3": int(source == "S3"),
    }


def features_to_vector(feats):
    return [feats[c] for c in FEATURE_COLUMNS]


def finalize_matrix(rows):
    """Convert a list of feature vectors to a clean float32 matrix."""
    if not rows:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype="float32")
    matrix = np.asarray(rows, dtype="float32")
    return np.nan_to_num(matrix, nan=0.0)


def build_text_lookup(path, id_col="entity_id", chunk_size=200_000):
    """Map entity_id -> (country_norm, name_norm, name_core, address_norm)."""
    lookup = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[id_col, "country_norm", "name_norm", "name_core", "address_norm"],
        chunksize=chunk_size,
        dtype="string",
    ):
        for row in chunk.itertuples(index=False):
            lookup[getattr(row, id_col)] = (
                row.country_norm,
                row.name_norm,
                row.name_core,
                row.address_norm,
            )
        del chunk
        gc.collect()
    return lookup


def load_ground_truth(path):
    """Map S1 entity id -> set of matched entity ids."""
    gt = pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
    )
    lookup = {}
    for row in gt.itertuples(index=False):
        matched = row.matched_entity_ids
        if pd.isna(matched) or matched == "":
            lookup[row.source1_entity_id] = set()
        else:
            lookup[row.source1_entity_id] = {
                x.strip() for x in str(matched).split(",") if x.strip()
            }
    del gt
    gc.collect()
    return lookup
