"""Training of the pairwise matcher.

Builds the training candidate set with the frozen blockers, labels it from the
ground truth, computes features, splits by S1 entity and fits a
HistGradientBoostingClassifier.
"""

import gc
import hashlib
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .blocking import build_blocker, iter_candidates
from .config import (
    CHUNK_SIZE,
    FEATURE_COLUMNS,
    FROZEN_BLOCKERS,
    MAX_TRAIN_NEG,
    MAX_TRAIN_POS,
    MODEL_PARAMS,
    RANDOM_SEED,
    TRAIN_NEG_RATE,
    VAL_HASH_MOD,
    VAL_HASH_REM,
)
from .features import (
    build_text_lookup,
    compute_pair_features,
    features_to_vector,
    finalize_matrix,
    load_ground_truth,
)


def stable_bucket(text, mod):
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def is_validation_entity(s1_id):
    return stable_bucket(s1_id, VAL_HASH_MOD) < VAL_HASH_REM


def build_training_matrix(
    s1_path, s2_path, s3_path, gt_path, chunk_size=CHUNK_SIZE,
    enabled_blockers=None,
):
    """Return the training feature matrix plus validation ids and labels."""
    enabled_blockers = enabled_blockers or FROZEN_BLOCKERS
    gt_lookup = load_ground_truth(gt_path)
    print("ground-truth S1 entities:", len(gt_lookup))

    s1_text = build_text_lookup(s1_path, chunk_size=chunk_size)
    print("S1 records:", len(s1_text))

    source_paths = {"S2": s2_path, "S3": s3_path}
    blockers = {}
    source_text = {}
    for tag, path in source_paths.items():
        if not os.path.exists(path):
            continue
        blockers[tag] = build_blocker(
            path, tag, enabled=enabled_blockers, chunk_size=chunk_size
        )
        source_text[tag] = build_text_lookup(path, chunk_size=chunk_size)
        print(f"{tag} text records:", len(source_text[tag]))

    rng = random.Random(RANDOM_SEED)

    ids, cands, sources, labels, is_val_flags, rows = [], [], [], [], [], []
    stats = {"pos": 0, "neg": 0, "val_rows": 0}

    for s1_id, cand_id, source_tag in iter_candidates(s1_path, blockers, chunk_size):
        s1 = s1_text.get(s1_id)
        cand = source_text.get(source_tag, {}).get(cand_id)
        if s1 is None or cand is None:
            continue

        label = int(cand_id in gt_lookup.get(s1_id, set()))
        is_val = is_validation_entity(s1_id)

        if label == 1:
            if stats["pos"] >= MAX_TRAIN_POS and not is_val:
                continue
        else:
            if not is_val:
                if stats["neg"] >= MAX_TRAIN_NEG:
                    continue
                if rng.random() > TRAIN_NEG_RATE:
                    continue

        feats = compute_pair_features(s1, cand, source_tag)

        ids.append(s1_id)
        cands.append(cand_id)
        sources.append(source_tag)
        labels.append(label)
        is_val_flags.append(is_val)
        rows.append(features_to_vector(feats))

        stats["pos" if label == 1 else "neg"] += 1
        if is_val:
            stats["val_rows"] += 1

    del s1_text, source_text, blockers
    gc.collect()

    print("kept rows:", len(rows), stats)

    X = finalize_matrix(rows)
    y = np.asarray(labels, dtype="int8")
    val_mask = np.asarray(is_val_flags, dtype=bool)

    return X, y, val_mask, ids, cands, sources


def run_training(cleaned_dir, model_dir, gt_path, chunk_size=CHUNK_SIZE,
                 enabled_blockers=None):
    os.makedirs(model_dir, exist_ok=True)

    s1_path = os.path.join(cleaned_dir, "train_s1_cleaned.tsv")
    s2_path = os.path.join(cleaned_dir, "train_s2_cleaned.tsv")
    s3_path = os.path.join(cleaned_dir, "train_s3_cleaned.tsv")

    X, y, val_mask, ids, cands, sources = build_training_matrix(
        s1_path, s2_path, s3_path, gt_path, chunk_size, enabled_blockers
    )

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("train:", X_train.shape, "positives:", int(y_train.sum()))
    print("val  :", X_val.shape, "positives:", int(y_val.sum()))

    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "pairwise_model.joblib")
    features_path = os.path.join(model_dir, "feature_columns.json")
    val_pred_path = os.path.join(model_dir, "val_predictions.tsv")

    joblib.dump(model, model_path)
    with open(features_path, "w", encoding="utf-8") as handle:
        json.dump(FEATURE_COLUMNS, handle, indent=2)

    if len(X_val):
        val_prob = model.predict_proba(X_val)[:, 1]
        val_ids = [ids[i] for i in range(len(ids)) if val_mask[i]]
        val_cands = [cands[i] for i in range(len(cands)) if val_mask[i]]
        val_sources = [sources[i] for i in range(len(sources)) if val_mask[i]]
        pd.DataFrame(
            {
                "s1_entity_id": val_ids,
                "candidate_entity_id": val_cands,
                "source": val_sources,
                "label": y_val,
                "match_probability": val_prob,
            }
        ).to_csv(val_pred_path, sep="\t", index=False)

    print("saved model:", model_path)
    print("saved features:", features_path)
    print("saved validation predictions:", val_pred_path)

    return model_path, val_pred_path
