"""Test inference and submission writing.

Produces the two required tab-separated files:

    output/candidate_pairs.tsv  -> source1_entity_id, candidate_entity_ids
    output/matching_results.tsv -> source1_entity_id, matched_entity_ids

One row per test S1 entity, comma separated id lists, no duplicates, only
S2-/S3- ids, and matches are always a subset of the candidates.
"""

import gc
import os

import joblib
import numpy as np
import pandas as pd

from .blocking import build_blocker, iter_candidates
from .config import CHUNK_SIZE, FEATURE_BATCH, FROZEN_BLOCKERS
from .decision import decide_entity, load_decision
from .features import (
    build_text_lookup,
    compute_pair_features,
    features_to_vector,
    finalize_matrix,
)


def score_candidates(s1_path, source_paths, model, chunk_size=CHUNK_SIZE,
                     enabled_blockers=None):
    """Return a DataFrame of (s1, candidate, source, probability)."""
    enabled_blockers = enabled_blockers or FROZEN_BLOCKERS
    s1_text = build_text_lookup(s1_path, chunk_size=chunk_size)
    print("test S1 records:", len(s1_text))

    blockers = {}
    for tag, path in source_paths.items():
        if os.path.exists(path):
            blockers[tag] = build_blocker(
                path, tag, enabled=enabled_blockers, chunk_size=chunk_size
            )

    frames = []
    for tag, path in source_paths.items():
        if tag not in blockers:
            continue
        source_text = build_text_lookup(path, chunk_size=chunk_size)
        print(f"{tag} text records:", len(source_text))

        batch_ids, batch_cand, batch_src, batch_X = [], [], [], []

        def flush():
            if not batch_X:
                return
            matrix = finalize_matrix(batch_X)
            probs = model.predict_proba(matrix)[:, 1]
            frames.append(
                pd.DataFrame(
                    {
                        "s1_entity_id": list(batch_ids),
                        "candidate_entity_id": list(batch_cand),
                        "source": list(batch_src),
                        "match_probability": probs,
                    }
                )
            )
            batch_ids.clear()
            batch_cand.clear()
            batch_src.clear()
            batch_X.clear()

        for s1_id, cand_id, source_tag in iter_candidates(s1_path, {tag: blockers[tag]}, chunk_size):
            s1 = s1_text.get(s1_id)
            cand = source_text.get(cand_id)
            if s1 is None or cand is None:
                continue
            feats = compute_pair_features(s1, cand, source_tag)
            batch_ids.append(s1_id)
            batch_cand.append(cand_id)
            batch_src.append(source_tag)
            batch_X.append(features_to_vector(feats))
            if len(batch_X) >= FEATURE_BATCH:
                flush()
        flush()

        del source_text
        gc.collect()

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        columns=["s1_entity_id", "candidate_entity_id", "source", "match_probability"]
    )


def write_submission(candidate_pairs, s1_ids, threshold, relative, output_dir):
    """Write both TSVs, one row per S1, preserving ``s1_ids`` order."""
    os.makedirs(output_dir, exist_ok=True)

    groups = {k: v for k, v in candidate_pairs.groupby("s1_entity_id", sort=False)}

    candidate_rows = []
    matched_rows = []

    for s1_id in s1_ids:
        group = groups.get(s1_id)

        if group is None or len(group) == 0:
            candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ""})
            matched_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
            continue

        group = group.sort_values("match_probability", ascending=False)
        cand_ids = list(dict.fromkeys(group["candidate_entity_id"].tolist()))
        probs = group["match_probability"].to_numpy()

        matched = list(
            dict.fromkeys(decide_entity(probs, group["candidate_entity_id"].tolist(), threshold, relative))
        )

        candidate_rows.append(
            {"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cand_ids)}
        )
        matched_rows.append(
            {"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matched)}
        )

    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matches_path = os.path.join(output_dir, "matching_results.tsv")

    pd.DataFrame(candidate_rows).to_csv(candidate_path, sep="\t", index=False)
    pd.DataFrame(matched_rows).to_csv(matches_path, sep="\t", index=False)

    print("wrote", candidate_path)
    print("wrote", matches_path)

    return candidate_path, matches_path


def run_inference(cleaned_dir, model_dir, output_dir, chunk_size=CHUNK_SIZE,
                  enabled_blockers=None):
    model = joblib.load(os.path.join(model_dir, "pairwise_model.joblib"))
    decision = load_decision(os.path.join(model_dir, "decision_config.json"))
    threshold = float(decision["threshold"])
    relative = float(decision.get("relative", 0.0))
    print("decision:", threshold, relative)

    s1_path = os.path.join(cleaned_dir, "test_s1_cleaned.tsv")
    source_paths = {
        "S2": os.path.join(cleaned_dir, "test_s2_cleaned.tsv"),
        "S3": os.path.join(cleaned_dir, "test_s3_cleaned.tsv"),
    }

    candidate_pairs = score_candidates(
        s1_path, source_paths, model, chunk_size, enabled_blockers
    )
    print("candidate pairs scored:", len(candidate_pairs))

    s1_ids = pd.read_csv(
        s1_path, sep="\t", usecols=["entity_id"], dtype="string"
    )["entity_id"].tolist()

    return write_submission(candidate_pairs, s1_ids, threshold, relative, output_dir)
