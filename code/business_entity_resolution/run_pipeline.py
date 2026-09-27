"""End-to-end pipeline entry point.

Reproduces the submission from the raw dataset:

    python run_pipeline.py all --data-dir dataset --output-dir output

Stages can also be run individually with --stage.

    normalize -> train -> tune -> predict -> validate
"""

import argparse
import os
import sys

from src import decision, evaluate_blocking, inference, normalization, training, validate_submission
from src.config import (
    ADVANCED_BLOCKERS,
    CHUNK_SIZE,
    FROZEN_BLOCKERS,
    INFERENCE_BUCKETS,
)


def _blockers(args):
    return ADVANCED_BLOCKERS if args.advanced_blocking else FROZEN_BLOCKERS


def run_normalize(args):
    return normalization.normalize_all(args.data_dir, args.work_dir)


def run_train(args):
    cleaned_dir = os.path.join(args.work_dir, "cleaned")
    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    return training.run_training(
        cleaned_dir, args.model_dir, gt_path,
        chunk_size=args.chunk_size, enabled_blockers=_blockers(args),
    )


def run_evaluate_blocking(args):
    cleaned_dir = os.path.join(args.work_dir, "cleaned")
    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(cleaned_dir, "train_s1_cleaned.tsv")
    source_paths = {
        "S2": os.path.join(cleaned_dir, "train_s2_cleaned.tsv"),
        "S3": os.path.join(cleaned_dir, "train_s3_cleaned.tsv"),
    }
    result = evaluate_blocking.evaluate(
        s1_path, source_paths, gt_path, _blockers(args), args.chunk_size
    )
    evaluate_blocking.report(result)
    out = os.path.join(args.model_dir, "blocking_eval.json")
    import json

    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", out)
    return result


def run_tune(args):
    val_path = os.path.join(args.model_dir, "val_predictions.tsv")
    decision_path = os.path.join(args.model_dir, "decision_config.json")
    return decision.tune(val_path, decision_path)


def run_predict(args):
    cleaned_dir = os.path.join(args.work_dir, "cleaned")
    return inference.run_inference(
        cleaned_dir, args.model_dir, args.output_dir,
        chunk_size=args.chunk_size, enabled_blockers=_blockers(args),
        num_buckets=args.buckets,
    )


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
    "train": run_train,
    "tune": run_tune,
    "predict": run_predict,
    "validate": run_validate,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", *STAGES.keys()],
        help="which stage to run",
    )
    parser.add_argument("--data-dir", default="dataset", help="folder with train/ and test/")
    parser.add_argument("--work-dir", default="work", help="intermediate artefacts")
    parser.add_argument("--model-dir", default=None, help="defaults to work/model")
    parser.add_argument("--output-dir", default="output", help="submission output folder")
    parser.add_argument(
        "--advanced-blocking",
        action="store_true",
        help="use the advanced blocker set (char3 + phonetic)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="rows per read chunk; lower it on a small Colab runtime",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=INFERENCE_BUCKETS,
        help="number of inference spill buckets (higher = less RAM)",
    )
    args = parser.parse_args(argv)

    if args.model_dir is None:
        args.model_dir = os.path.join(args.work_dir, "model")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    order = ["normalize", "train", "tune", "predict", "validate"]
    to_run = order if args.stage == "all" else [args.stage]

    exit_code = 0
    for stage in to_run:
        print(f"\n===== stage: {stage} =====")
        result = STAGES[stage](args)
        if isinstance(result, int):
            exit_code = result

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
