"""End-to-end pipeline entry point.

Reproduce the submission from the raw dataset:

    python run_pipeline.py all --data-dir dataset --output-dir output

Stages (``all`` runs normalize -> build-features -> train -> tune -> predict ->
validate)::

    normalize          raw TSVs -> cleaned TSVs
    evaluate-blocking  candidate recall ceiling (frozen vs advanced)
    build-features     the slow streaming feature pass -> train_matrix.npz
    train              fit the model from the cached matrix
    tune               sweep the decision rule for macro F0.5
    predict            test inference -> the two submission TSVs
    validate           format / coverage checks

Speed tips:

* reuse cleaned data with ``--cleaned-dir`` (skips normalize)
* the feature matrix is cached, so ``train``/``tune`` are fast on re-runs
* use ``--train-s1-rate 0.1`` for a quick smoke run
* use ``--chunk-size 20000 --buckets 1024`` on a small Colab runtime
"""

import argparse
import os
import sys
import time

from src import (
    decision,
    evaluate_blocking,
    inference,
    normalization,
    training,
    validate_submission,
)
from src.config import (
    ADVANCED_BLOCKERS,
    CHUNK_SIZE,
    FROZEN_BLOCKERS,
    INFERENCE_BUCKETS,
)


def _blockers(args):
    return ADVANCED_BLOCKERS if args.advanced_blocking else FROZEN_BLOCKERS


def _cleaned_dir(args):
    return args.cleaned_dir or os.path.join(args.work_dir, "cleaned")


def _gt_path(args):
    return args.gt or os.path.join(args.data_dir, "train", "train_ground_truth.tsv")


def run_normalize(args):
    if args.cleaned_dir:
        print(f"using existing cleaned data: {args.cleaned_dir} (skipping normalize)")
        return
    return normalization.normalize_all(args.data_dir, args.work_dir, args.chunk_size)


def run_build_features(args):
    return training.build_features(
        _cleaned_dir(args), args.model_dir, _gt_path(args),
        chunk_size=args.chunk_size,
        enabled_blockers=_blockers(args),
        s1_rate=args.train_s1_rate,
    )


def run_train(args):
    return training.run_training(
        _cleaned_dir(args), args.model_dir, _gt_path(args),
        chunk_size=args.chunk_size,
        enabled_blockers=_blockers(args),
        s1_rate=args.train_s1_rate,
        reuse_features=not args.no_reuse_features,
    )


def run_tune(args):
    val_path = os.path.join(args.model_dir, "val_predictions.tsv")
    decision_path = os.path.join(args.model_dir, "decision_config.json")
    return decision.tune(val_path, decision_path)


def run_predict(args):
    return inference.run_inference(
        _cleaned_dir(args), args.model_dir, args.output_dir,
        chunk_size=args.chunk_size,
        enabled_blockers=_blockers(args),
        num_buckets=args.buckets,
    )


def run_evaluate_blocking(args):
    cleaned = _cleaned_dir(args)
    source_paths = {
        "S2": os.path.join(cleaned, "train_s2_cleaned.tsv"),
        "S3": os.path.join(cleaned, "train_s3_cleaned.tsv"),
    }
    result = evaluate_blocking.evaluate(
        os.path.join(cleaned, "train_s1_cleaned.tsv"),
        source_paths,
        _gt_path(args),
        _blockers(args),
        args.chunk_size,
    )
    evaluate_blocking.report(result)

    out = os.path.join(args.model_dir, "blocking_eval.json")
    import json

    os.makedirs(args.model_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", out)
    return result


def run_validate(args):
    issues = validate_submission.validate(
        os.path.join(args.output_dir, "matching_results.tsv"),
        os.path.join(args.output_dir, "candidate_pairs.tsv"),
        os.path.join(args.data_dir, "test"),
    )
    if issues:
        print("FAIL")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1
    print("PASS")
    return 0


STAGES = {
    "normalize": run_normalize,
    "evaluate-blocking": run_evaluate_blocking,
    "build-features": run_build_features,
    "train": run_train,
    "tune": run_tune,
    "predict": run_predict,
    "validate": run_validate,
}

ALL_ORDER = ["normalize", "build-features", "train", "tune", "predict", "validate"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument(
        "--stage", default="all", choices=["all", *STAGES.keys()],
        help="which stage to run",
    )
    parser.add_argument("--data-dir", default="dataset", help="folder with train/ and test/")
    parser.add_argument("--work-dir", default="work", help="intermediate artefacts")
    parser.add_argument(
        "--cleaned-dir", default=None,
        help="reuse an existing cleaned folder (e.g. 03_Experiments/Cleaned_Data)",
    )
    parser.add_argument("--gt", default=None, help="train_ground_truth.tsv path")
    parser.add_argument("--model-dir", default=None, help="defaults to work/model")
    parser.add_argument("--output-dir", default="output", help="submission output folder")
    parser.add_argument(
        "--advanced-blocking", action="store_true",
        help="use the advanced blocker set (char3 + phonetic)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help="rows per read chunk; lower it on a small Colab runtime",
    )
    parser.add_argument(
        "--buckets", type=int, default=INFERENCE_BUCKETS,
        help="number of inference spill buckets (higher = less RAM)",
    )
    parser.add_argument(
        "--train-s1-rate", type=float, default=1.0,
        help="subsample S1 entities for quick experiments (e.g. 0.1)",
    )
    parser.add_argument(
        "--no-reuse-features", action="store_true",
        help="force rebuilding the cached training matrix",
    )
    args = parser.parse_args(argv)

    if args.model_dir is None:
        args.model_dir = os.path.join(args.work_dir, "model")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    to_run = ALL_ORDER if args.stage == "all" else [args.stage]

    exit_code = 0
    for stage in to_run:
        print(f"\n===== stage: {stage} =====")
        started = time.time()
        result = STAGES[stage](args)
        print(f"--- {stage} finished in {time.time() - started:.1f}s ---")
        if isinstance(result, int):
            exit_code = result

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
