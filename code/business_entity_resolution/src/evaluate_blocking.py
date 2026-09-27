"""Blocking evaluation (memory-bounded).

Source-driven: S2/S3 are streamed against the S1 index, so no source index or
source text is held in RAM.

Measures:

* recall of each blocker on its own
* recall of the frozen union (the 87.24% baseline)
* recall of the advanced union (the new ceiling)
* candidate volume per S1

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

from .config import ADVANCED_BLOCKERS, CHUNK_SIZE, FROZEN_BLOCKERS
from .features import load_ground_truth
from .reverse import build_allowed_sets, build_s1_blocker, iter_pairs_by_blocker


def evaluate(s1_path, source_paths, gt_path, enabled, chunk_size=CHUNK_SIZE):
    gt = load_ground_truth(gt_path)
    total = sum(len(v) for v in gt.values())
    print("ground-truth S1 entities:", len(gt), "| true matches:", f"{total:,}")

    # Reverse ground truth: candidate id -> set of S1 ids it truly matches.
    gt_rev = {}
    for s1_id, cand_ids in gt.items():
        for cand_id in cand_ids:
            gt_rev.setdefault(cand_id, set()).add(s1_id)

    allowed_name, allowed_addr, allowed_char3 = build_allowed_sets(
        source_paths, chunk_size
    )
    blocker = build_s1_blocker(
        s1_path, allowed_name, allowed_addr, allowed_char3, enabled, chunk_size
    )
    s1_universe = len(blocker["text"])

    captured = 0
    captured_frozen = 0
    per_blocker = defaultdict(int)
    cand_counts = defaultdict(int)
    captured_per_s1 = defaultdict(int)

    for cand_id, tag, cand_tuple, by in iter_pairs_by_blocker(
        source_paths, blocker, chunk_size
    ):
        union = set().union(*by.values()) if by else set()
        for s1_id in union:
            cand_counts[s1_id] += 1

        true_s1 = gt_rev.get(cand_id)
        if not true_s1:
            continue

        hit = true_s1 & union
        if hit:
            captured += len(hit)
            for s1_id in hit:
                captured_per_s1[s1_id] += 1

        frozen_sets = [by[b] for b in FROZEN_BLOCKERS if b in by]
        if frozen_sets:
            captured_frozen += len(true_s1 & set().union(*frozen_sets))

        for bname, ids in by.items():
            per_blocker[bname] += len(true_s1 & ids)

    del blocker, gt_rev
    gc.collect()

    counts = np.asarray(list(cand_counts.values()), dtype="float64")
    if counts.size == 0:
        counts = np.zeros(1)

    s1_with_matches = sum(1 for v in gt.values() if v)
    hard_misses = sum(
        1 for s1_id, v in gt.items() if v and captured_per_s1.get(s1_id, 0) == 0
    )

    result = {
        "enabled": list(enabled),
        "total_true_matches": int(total),
        "captured_union": int(captured),
        "recall_union": captured / total if total else 0.0,
        "captured_frozen": int(captured_frozen),
        "recall_frozen": captured_frozen / total if total else 0.0,
        "incremental_captured": int(captured - captured_frozen),
        "incremental_recall": (captured - captured_frozen) / total if total else 0.0,
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
            "zero_s1": int(max(0, s1_universe - len(cand_counts))),
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
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--advanced", action="store_true", help="include advanced blockers")
    args = parser.parse_args(argv)

    enabled = ADVANCED_BLOCKERS if args.advanced else FROZEN_BLOCKERS
    s1_path = args.s1 or os.path.join(args.cleaned, "train_s1_cleaned.tsv")
    source_paths = {
        "S2": os.path.join(args.cleaned, "train_s2_cleaned.tsv"),
        "S3": os.path.join(args.cleaned, "train_s3_cleaned.tsv"),
    }

    result = evaluate(s1_path, source_paths, args.gt, enabled, args.chunk_size)
    report(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("wrote", args.out)

    return result


if __name__ == "__main__":
    main()
    sys.exit(0)
