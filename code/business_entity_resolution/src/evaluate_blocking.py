"""Blocking evaluation.

Measures, on the training data:

* recall of each blocker on its own
* recall of the union of the frozen three (the 87.24% baseline)
* recall of the union including the advanced blockers (the new ceiling)
* candidate volume per S1 (mean / median / p95 / p99 / max / zero-candidate)

Run:

    python -m src.evaluate_blocking --cleaned work/cleaned --gt dataset/train/train_ground_truth.tsv
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .blocking import build_blocker, candidates_by_blocker
from .config import ADVANCED_BLOCKERS, CHUNK_SIZE, FROZEN_BLOCKERS
from .features import load_ground_truth


def evaluate(s1_path, source_paths, gt_path, enabled, chunk_size=CHUNK_SIZE):
    gt = load_ground_truth(gt_path)
    print("ground-truth S1 entities:", len(gt))

    blockers = {}
    for tag, path in source_paths.items():
        if os.path.exists(path):
            blockers[tag] = build_blocker(path, tag, enabled=enabled, chunk_size=chunk_size)

    total = 0
    captured_union = 0
    captured_base = 0
    per_blocker = defaultdict(int)
    cand_counts = []
    s1_with_matches = 0
    hard_misses = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=["entity_id", "country_norm", "name_core", "address_norm"],
        chunksize=chunk_size,
        dtype="string",
    ):
        for eid, country, name, address in zip(
            chunk["entity_id"], chunk["country_norm"],
            chunk["name_core"], chunk["address_norm"],
        ):
            true_ids = gt.get(eid, set())
            if not true_ids:
                continue
            s1_with_matches += 1
            per_s1_candidates = 0
            any_captured = False

            for tag, blocker in blockers.items():
                prefix = f"{tag}-"
                true_tag = {x for x in true_ids if x.startswith(prefix)}

                by = candidates_by_blocker(country, name, address, blocker)
                if by:
                    union_tag = set().union(*by.values())
                    base_tag = set().union(
                        *[by[b] for b in FROZEN_BLOCKERS if b in by]
                    ) if any(b in by for b in FROZEN_BLOCKERS) else set()
                else:
                    union_tag, base_tag = set(), set()

                total += len(true_tag)
                captured_union += len(true_tag & union_tag)
                captured_base += len(true_tag & base_tag)
                any_captured = any_captured or bool(true_tag & union_tag)

                for bname, ids in by.items():
                    per_blocker[bname] += len(true_tag & ids)

                per_s1_candidates += len(union_tag)

            cand_counts.append(per_s1_candidates)
            if true_ids and not any_captured:
                hard_misses += 1

        del chunk
        gc.collect()

    counts = np.asarray(cand_counts) if cand_counts else np.zeros(1)

    result = {
        "enabled": list(enabled),
        "total_true_matches": int(total),
        "captured_union": int(captured_union),
        "recall_union": captured_union / total if total else 0.0,
        "captured_frozen": int(captured_base),
        "recall_frozen": captured_base / total if total else 0.0,
        "incremental_captured": int(captured_union - captured_base),
        "incremental_recall": (captured_union - captured_base) / total if total else 0.0,
        "per_blocker_captured": {k: int(v) for k, v in per_blocker.items()},
        "per_blocker_recall": {
            k: (v / total if total else 0.0) for k, v in per_blocker.items()
        },
        "s1_with_matches": int(s1_with_matches),
        "s1_hard_misses": int(hard_misses),
        "candidate_stats": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "max": float(counts.max()),
            "zero_s1": int((counts == 0).sum()),
        },
    }
    return result


def report(result):
    print("\n=== blocking evaluation ===")
    print("blockers           :", result["enabled"])
    print("total true matches :", f"{result['total_true_matches']:,}")
    print(f"frozen union recall: {result['recall_frozen']:.4%}")
    print(f"union recall       : {result['recall_union']:.4%}")
    print(
        f"incremental        : +{result['incremental_captured']:,} matches "
        f"(+{result['incremental_recall']:.4%})"
    )
    print("per-blocker recall :")
    for name, value in sorted(
        result["per_blocker_recall"].items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {name:<14} {value:.4%}")
    stats = result["candidate_stats"]
    print(
        "candidates / S1    : "
        f"mean={stats['mean']:.1f} median={stats['median']:.0f} "
        f"p95={stats['p95']:.0f} p99={stats['p99']:.0f} max={stats['max']:.0f} "
        f"zero={stats['zero_s1']}"
    )
    print("S1 hard misses     :", result["s1_hard_misses"], "/", result["s1_with_matches"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate blocking recall.")
    parser.add_argument("--cleaned", required=True, help="folder with *_cleaned.tsv")
    parser.add_argument("--gt", required=True, help="train_ground_truth.tsv")
    parser.add_argument("--s1", default=None, help="defaults to <cleaned>/train_s1_cleaned.tsv")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--advanced", action="store_true", help="include advanced blockers")
    parser.add_argument("--frozen", action="store_true", help="only the frozen three")
    args = parser.parse_args(argv)

    enabled = ADVANCED_BLOCKERS if args.advanced else FROZEN_BLOCKERS
    s1_path = args.s1 or os.path.join(args.cleaned, "train_s1_cleaned.tsv")
    source_paths = {
        "S2": os.path.join(args.cleaned, "train_s2_cleaned.tsv"),
        "S3": os.path.join(args.cleaned, "train_s3_cleaned.tsv"),
    }

    result = evaluate(s1_path, source_paths, args.gt, enabled)
    report(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("wrote", args.out)

    return result


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
