"""S1-level decision layer and the challenge metric.

The pairwise model returns P(match). The challenge is scored per S1 entity
(macro F0.5), so probabilities are converted into a per-S1 match set:

    cut       = max(threshold, relative * max_probability_within_S1)
    predicted = {candidate : P(match) >= cut}
"""

import json
import os

import numpy as np

from .config import DEFAULT_RELATIVE, DEFAULT_THRESHOLD


def entity_f05(true_ids, pred_ids):
    """F0.5 for a single S1 entity."""
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0

    beta2 = 0.25  # beta = 0.5
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def decide_entity(probs, cand_ids, threshold, relative):
    if len(probs) == 0:
        return []
    cut = max(threshold, relative * float(np.max(probs)))
    return [c for c, p in zip(cand_ids, probs) if p >= cut]


def macro_f05(pred_df, threshold, relative=0.0):
    scores = []
    for _, group in pred_df.groupby("s1_entity_id", sort=False):
        probs = group["match_probability"].to_numpy()
        cand_ids = group["candidate_entity_id"].tolist()
        true_ids = set(group.loc[group["label"] == 1, "candidate_entity_id"])
        scores.append(
            entity_f05(true_ids, set(decide_entity(probs, cand_ids, threshold, relative)))
        )
    return float(np.mean(scores)) if scores else 0.0


def tune(validation_path, decision_path):
    """Sweep thresholds and persist the best decision configuration."""
    import pandas as pd

    val = pd.read_csv(
        validation_path,
        sep="\t",
        dtype={
            "s1_entity_id": "string",
            "candidate_entity_id": "string",
            "source": "string",
        },
    )

    results = []
    for relative in [0.0, 0.90, 0.95, 0.98]:
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]:
            results.append(
                {
                    "threshold": threshold,
                    "relative": relative,
                    "macro_f05": macro_f05(val, threshold, relative),
                }
            )

    results.sort(key=lambda r: r["macro_f05"], reverse=True)
    best = results[0]

    config = {
        "threshold": float(best["threshold"]),
        "relative": float(best["relative"]),
        "note": "cut = max(threshold, relative * max_probability_within_s1)",
    }
    os.makedirs(os.path.dirname(decision_path), exist_ok=True)
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("best decision config:", config)
    print("top 10:")
    for row in results[:10]:
        print(
            f"  thr={row['threshold']:.3f} rel={row['relative']:.2f} "
            f"macro_F0.5={row['macro_f05']:.4f}"
        )

    return config


def load_decision(path):
    if not os.path.exists(path):
        return {"threshold": DEFAULT_THRESHOLD, "relative": DEFAULT_RELATIVE}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
