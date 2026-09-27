"""Training of the pairwise matcher.

Split into two cheap-to-reuse steps:

1. ``build_features`` - the slow part: stream S2/S3, generate candidates,
   compute the 24 features and cache the matrix to ``train_matrix.npz``.
2. ``run_training`` - the fast part: load the cached matrix, fit the model and
   write validation predictions.

That way the expensive streaming pass runs once and you can re-fit / re-tune the
model and decision layer in seconds.
"""

import gc
import hashlib
import json
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import (
    CHUNK_SIZE,
    FEATURE_BATCH,
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
from .features import compute_pair_features, features_to_vector, load_ground_truth
from .reverse import build_allowed_sets, build_s1_blocker, iter_pairs

MATRIX_NAME = "train_matrix.npz"
META_NAME = "train_matrix.meta.json"


def stable_bucket(text, mod):
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def is_validation_entity(s1_id):
    return stable_bucket(s1_id, VAL_HASH_MOD) < VAL_HASH_REM


def save_matrix(path, X, y, val_mask, val_ids, val_cands, val_sources):
    np.savez_compressed(
        path,
        X=X,
        y=y,
        val_mask=val_mask,
        val_ids=np.asarray(val_ids, dtype=object),
        val_cands=np.asarray(val_cands, dtype=object),
        val_sources=np.asarray(val_sources, dtype=object),
    )


def load_matrix(path):
    data = np.load(path, allow_pickle=True)
    return (
        data["X"],
        data["y"],
        data["val_mask"],
        list(data["val_ids"]),
        list(data["val_cands"]),
        list(data["val_sources"]),
    )


def build_training_matrix(
    s1_path,
    s2_path,
    s3_path,
    gt_path,
    chunk_size=CHUNK_SIZE,
    enabled_blockers=None,
    s1_rate=1.0,
):
    """Stream the sources and return (X, y, val_mask, val_ids, val_cands, val_sources).

    ``s1_rate`` subsamples S1 entities (deterministic hash) for quick
    experiments; validation entities are always kept.
    """
    enabled_blockers = enabled_blockers or FROZEN_BLOCKERS

    gt = load_ground_truth(gt_path)
    print("ground-truth S1 entities:", len(gt))

    source_paths = {"S2": s2_path, "S3": s3_path}
    allowed_name, allowed_addr, allowed_char3 = build_allowed_sets(
        source_paths, chunk_size
    )
    s1_blocker = build_s1_blocker(
        s1_path, allowed_name, allowed_addr, allowed_char3,
        enabled_blockers, chunk_size,
    )

    rng = random.Random(RANDOM_SEED)
    sample_cut = int(round(s1_rate * 1000))

    X_parts, y_parts, val_parts = [], [], []
    val_ids, val_cands, val_sources = [], [], []

    batch_X, batch_y, batch_val = [], [], []
    stats = {"pos": 0, "neg": 0, "val_rows": 0, "skipped_s1": 0}

    def flush():
        if not batch_X:
            return
        X_parts.append(np.asarray(batch_X, dtype="float32"))
        y_parts.append(np.asarray(batch_y, dtype="int8"))
        val_parts.append(np.asarray(batch_val, dtype=bool))
        batch_X.clear()
        batch_y.clear()
        batch_val.clear()

    for s1_id, cand_id, s1_tuple, cand_tuple, tag in iter_pairs(
        source_paths, s1_blocker, chunk_size
    ):
        is_val = is_validation_entity(s1_id)

        # Fast-iteration subsample: keep validation entities, sample the rest.
        if not is_val and sample_cut < 1000:
            if stable_bucket(s1_id, 1000) >= sample_cut:
                stats["skipped_s1"] += 1
                continue

        true_ids = gt.get(s1_id)
        label = int(true_ids is not None and cand_id in true_ids)

        if label == 1:
            if stats["pos"] >= MAX_TRAIN_POS and not is_val:
                continue
        else:
            if not is_val:
                if stats["neg"] >= MAX_TRAIN_NEG:
                    continue
                if rng.random() > TRAIN_NEG_RATE:
                    continue

        feats = compute_pair_features(s1_tuple, cand_tuple, tag)
        batch_X.append(features_to_vector(feats))
        batch_y.append(label)
        batch_val.append(is_val)

        if is_val:
            val_ids.append(s1_id)
            val_cands.append(cand_id)
            val_sources.append(tag)
            stats["val_rows"] += 1

        stats["pos" if label == 1 else "neg"] += 1

        if len(batch_X) >= FEATURE_BATCH:
            flush()

    flush()

    del s1_blocker
    gc.collect()

    n_features = len(FEATURE_COLUMNS)
    if X_parts:
        X = np.nan_to_num(np.concatenate(X_parts, axis=0), nan=0.0)
        y = np.concatenate(y_parts, axis=0)
        val_mask = np.concatenate(val_parts, axis=0)
    else:
        X = np.zeros((0, n_features), dtype="float32")
        y = np.zeros((0,), dtype="int8")
        val_mask = np.zeros((0,), dtype=bool)

    print("kept rows:", len(X), stats)
    del X_parts, y_parts, val_parts
    gc.collect()

    return X, y, val_mask, val_ids, val_cands, val_sources


def build_features(
    cleaned_dir,
    model_dir,
    gt_path,
    chunk_size=CHUNK_SIZE,
    enabled_blockers=None,
    s1_rate=1.0,
):
    """Run the slow streaming feature pass and cache the matrix."""
    os.makedirs(model_dir, exist_ok=True)
    started = time.time()

    X, y, val_mask, val_ids, val_cands, val_sources = build_training_matrix(
        os.path.join(cleaned_dir, "train_s1_cleaned.tsv"),
        os.path.join(cleaned_dir, "train_s2_cleaned.tsv"),
        os.path.join(cleaned_dir, "train_s3_cleaned.tsv"),
        gt_path,
        chunk_size,
        enabled_blockers,
        s1_rate,
    )

    path = os.path.join(model_dir, MATRIX_NAME)
    save_matrix(path, X, y, val_mask, val_ids, val_cands, val_sources)

    # The matrix is only valid for the exact configuration that produced it.
    with open(os.path.join(model_dir, META_NAME), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "enabled_blockers": list(enabled_blockers or FROZEN_BLOCKERS),
                "s1_rate": float(s1_rate),
                "rows": int(X.shape[0]),
            },
            handle,
            indent=2,
        )

    print(f"saved training matrix: {path} ({X.shape}) in {time.time() - started:.1f}s")
    return path


def run_training(
    cleaned_dir,
    model_dir,
    gt_path,
    chunk_size=CHUNK_SIZE,
    enabled_blockers=None,
    s1_rate=1.0,
    reuse_features=True,
):
    os.makedirs(model_dir, exist_ok=True)
    started = time.time()

    enabled = list(enabled_blockers or FROZEN_BLOCKERS)
    path = os.path.join(model_dir, MATRIX_NAME)
    meta_path = os.path.join(model_dir, META_NAME)

    reusable = reuse_features and os.path.exists(path)
    if reusable and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("enabled_blockers") != enabled:
            print("cache uses different blockers -> rebuilding features")
            reusable = False
        elif abs(float(meta.get("s1_rate", 1.0)) - float(s1_rate)) > 1e-9:
            print("cache uses a different --train-s1-rate -> rebuilding features")
            reusable = False

    if reusable:
        X, y, val_mask, val_ids, val_cands, val_sources = load_matrix(path)
        print("loaded cached training matrix:", path, X.shape)
    else:
        path = build_features(
            cleaned_dir, model_dir, gt_path,
            chunk_size, enabled_blockers, s1_rate,
        )
        X, y, val_mask, val_ids, val_cands, val_sources = load_matrix(path)

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_train) == 0:
        raise SystemExit(
            "No training rows were produced. Check --cleaned-dir / --gt paths, the "
            "blocker set, and raise --train-s1-rate (it may be sampling out every "
            "S1 entity)."
        )

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
        pd.DataFrame(
            {
                "s1_entity_id": val_ids,
                "candidate_entity_id": val_cands,
                "source": val_sources,
                "label": y_val,
                "match_probability": val_prob,
            }
        ).to_csv(val_pred_path, sep="\t", index=False)

    print(f"model trained in {time.time() - started:.1f}s")
    print("saved model:", model_path)
    print("saved features:", features_path)
    print("saved validation predictions:", val_pred_path)

    return model_path, val_pred_path
