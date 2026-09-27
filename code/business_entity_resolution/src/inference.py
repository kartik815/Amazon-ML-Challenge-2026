"""Test inference and submission writing.

Memory-bounded:

* reverse blocking -> no source index / source text in RAM
* features are computed and scored in batches
* scored pairs are spilled into buckets keyed by the S1's position, so outputs
  are produced one bounded slice at a time (never all pairs at once)

Produces the two required tab-separated files:

    output/matching_results.tsv   source1_entity_id, matched_entity_ids
    output/candidate_pairs.tsv    source1_entity_id, candidate_entity_ids

One row per test S1, comma separated id lists, no duplicates, only S2-/S3- ids,
and matches are always a subset of the candidates.
"""

import gc
import os
import shutil

import joblib
import numpy as np
import pandas as pd

from .config import (
    BUCKET_BUFFER_ROWS,
    CHUNK_SIZE,
    FEATURE_BATCH,
    FROZEN_BLOCKERS,
    INFERENCE_BUCKETS,
)
from .decision import decide_entity, load_decision
from .features import compute_pair_features, features_to_vector, finalize_matrix
from .reverse import build_allowed_sets, build_s1_blocker, iter_pairs

_CAND_HEADER = "source1_entity_id\tcandidate_entity_ids\n"
_MATCH_HEADER = "source1_entity_id\tmatched_entity_ids\n"


class _BucketWriter:
    """Appends scored pairs to per-S1 buckets, flushing in bounded batches."""

    def __init__(self, base_dir, count):
        os.makedirs(base_dir, exist_ok=True)
        self.count = count
        self.base_dir = base_dir
        self.paths = [os.path.join(base_dir, f"bucket_{b:04d}.tsv") for b in range(count)]
        self._buf = {}
        self._buffered = 0
        self._written = set()

    def add(self, buckets, s1_ids, cand_ids, probs):
        for b, s, c, p in zip(buckets, s1_ids, cand_ids, probs):
            entry = self._buf.get(b)
            if entry is None:
                entry = ([], [], [])
                self._buf[b] = entry
            entry[0].append(s)
            entry[1].append(c)
            entry[2].append(float(p))
        self._buffered += len(s1_ids)
        if self._buffered >= BUCKET_BUFFER_ROWS:
            self.flush()

    def flush(self):
        for b, (s1s, cands, probs) in self._buf.items():
            frame = pd.DataFrame(
                {
                    "s1_entity_id": s1s,
                    "candidate_entity_id": cands,
                    "match_probability": probs,
                }
            )
            first = b not in self._written
            frame.to_csv(
                self.paths[b], sep="\t", index=False,
                mode="w" if first else "a", header=first,
            )
            self._written.add(b)
        self._buf.clear()
        self._buffered = 0

    def cleanup(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)


def _score_to_buckets(s1_ids, s1_blocker, source_paths, model, writer, span, chunk_size):
    idx_of = {sid: i for i, sid in enumerate(s1_ids)}

    batch_X, batch_bucket, batch_s1, batch_cand = [], [], [], []
    stats = {"scored": 0, "skipped": 0}

    def flush():
        if not batch_X:
            return
        matrix = finalize_matrix(batch_X)
        probs = model.predict_proba(matrix)[:, 1]
        writer.add(batch_bucket, batch_s1, batch_cand, probs)
        stats["scored"] += len(batch_X)
        batch_X.clear()
        batch_bucket.clear()
        batch_s1.clear()
        batch_cand.clear()

    for s1_id, cand_id, s1_tuple, cand_tuple, tag in iter_pairs(
        source_paths, s1_blocker, chunk_size
    ):
        idx = idx_of.get(s1_id)
        if idx is None:
            stats["skipped"] += 1
            continue
        feats = compute_pair_features(s1_tuple, cand_tuple, tag)
        batch_X.append(features_to_vector(feats))
        batch_bucket.append(idx // span)
        batch_s1.append(s1_id)
        batch_cand.append(cand_id)
        if len(batch_X) >= FEATURE_BATCH:
            flush()

    flush()
    writer.flush()
    return stats


def _write_outputs(s1_ids, writer, span, threshold, relative, output_dir):
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matches_path = os.path.join(output_dir, "matching_results.tsv")

    with open(candidate_path, "w", encoding="utf-8", newline="") as cf, open(
        matches_path, "w", encoding="utf-8", newline=""
    ) as mf:
        cf.write(_CAND_HEADER)
        mf.write(_MATCH_HEADER)

        for b in range(writer.count):
            lo = b * span
            hi = min((b + 1) * span, len(s1_ids))
            if lo >= hi:
                continue

            by_s1 = {}
            path = writer.paths[b]
            if os.path.exists(path):
                for chunk in pd.read_csv(
                    path,
                    sep="\t",
                    chunksize=500_000,
                    dtype={
                        "s1_entity_id": "string",
                        "candidate_entity_id": "string",
                        "match_probability": "float32",
                    },
                ):
                    for s, c, p in zip(
                        chunk["s1_entity_id"],
                        chunk["candidate_entity_id"],
                        chunk["match_probability"],
                    ):
                        by_s1.setdefault(s, []).append((c, float(p)))
                    del chunk
                    gc.collect()

            for i in range(lo, hi):
                sid = s1_ids[i]
                entries = by_s1.get(sid)
                if not entries:
                    cf.write(f"{sid}\t\n")
                    mf.write(f"{sid}\t\n")
                    continue

                entries.sort(key=lambda item: item[1], reverse=True)
                cand_ids = list(dict.fromkeys(c for c, _ in entries))
                cut = max(threshold, relative * entries[0][1])
                matched = list(dict.fromkeys(c for c, p in entries if p >= cut))
                cand_set = set(cand_ids)
                matched = [c for c in matched if c in cand_set]

                cf.write(f"{sid}\t{','.join(cand_ids)}\n")
                mf.write(f"{sid}\t{','.join(matched)}\n")

            del by_s1
            gc.collect()

    print("wrote", candidate_path)
    print("wrote", matches_path)
    return candidate_path, matches_path


def run_inference(
    cleaned_dir,
    model_dir,
    output_dir,
    chunk_size=CHUNK_SIZE,
    enabled_blockers=None,
    num_buckets=INFERENCE_BUCKETS,
):
    enabled_blockers = enabled_blockers or FROZEN_BLOCKERS
    os.makedirs(output_dir, exist_ok=True)

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

    s1_ids = pd.read_csv(
        s1_path, sep="\t", usecols=["entity_id"], dtype="string"
    )["entity_id"].tolist()
    print("test S1 entities:", len(s1_ids))

    allowed_name, allowed_addr, allowed_char3 = build_allowed_sets(source_paths, chunk_size)
    s1_blocker = build_s1_blocker(
        s1_path, allowed_name, allowed_addr, allowed_char3,
        enabled_blockers, chunk_size,
    )

    span = max(1, -(-len(s1_ids) // max(1, num_buckets)))
    writer = _BucketWriter(os.path.join(output_dir, "_buckets"), num_buckets)

    stats = _score_to_buckets(
        s1_ids, s1_blocker, source_paths, model, writer, span, chunk_size
    )
    print("scored pairs:", stats["scored"], "| skipped:", stats["skipped"])

    del s1_blocker
    gc.collect()

    candidate_path, matches_path = _write_outputs(
        s1_ids, writer, span, threshold, relative, output_dir
    )
    writer.cleanup()

    return candidate_path, matches_path
