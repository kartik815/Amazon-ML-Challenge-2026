# Business Entity Resolution — runnable pipeline

Self-contained code that regenerates both submission files from the raw
`train/` and `test/` TSVs. No network access, no external data lookups.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model: `sklearn.ensemble.HistGradientBoostingClassifier` (scikit-learn, BSD-3).
The pipeline uses only MIT/BSD/Apache-licensed libraries and no pretrained
weights, so it is comfortably within the licence and parameter limits.

## Inputs

Place the dataset so the layout is:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run everything

```bash
python run_pipeline.py all --data-dir dataset --output-dir output
```

Stages, in order:

| Stage | What it does | Output |
| ----- | ------------ | ------ |
| `normalize` | NFKC, casing, abbreviation/suffix normalisation, `name_core`, token counts | `work/cleaned/*_cleaned.tsv` |
| `evaluate-blocking` | recall of each blocker and the union + candidate volume (optional) | `work/model/blocking_eval.json` |
| `train` | frozen blocking + labels + 24 features + S1-grouped split + model | `work/model/pairwise_model.joblib`, `feature_columns.json`, `val_predictions.tsv` |
| `tune` | macro F0.5 sweep for the decision rule | `work/model/decision_config.json` |
| `predict` | test blocking, scoring, decision, output writing | `output/candidate_pairs.tsv`, `output/matching_results.tsv` |
| `validate` | format and consistency checks | exit 0 = PASS |

Run a single stage, e.g.:

```bash
python run_pipeline.py --stage predict --work-dir work --output-dir output
```

## Improved blocking (optional, recommended)

The default candidate generation is the frozen three-way union. Add the two
complementary blockers to raise the recall ceiling:

```bash
python run_pipeline.py all --data-dir dataset --output-dir output --advanced-blocking
```

* `char3` — shared character 3-grams (>= 2 shared), recovers typos.
* `phonetic` — country-keyed consonant skeleton of the full name and of the
  first meaningful token; bridges transliteration / cross-script pairs such as
  `green logistics` vs `grina lajistiks`.

Measure the ceiling before deciding:

```bash
python run_pipeline.py --stage evaluate-blocking --work-dir work --advanced-blocking
```

## Validate before submitting

```bash
python -m src.validate_submission \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The official `utils/validate_submission.py` can be run against the same files;
this validator implements the same checks (stdlib only).

## How it works

1. **Normalisation** — `src/normalization.py`
2. **Blocking** (exact name + name token + address token, union) — `src/blocking.py`
3. **Features** (6 base + 18 complementary = 24) — `src/features.py`
4. **Model** (HistGradientBoosting, S1-grouped validation split) — `src/training.py`
5. **Decision layer** (per-S1 threshold, macro F0.5) — `src/decision.py`
6. **Inference and submission** — `src/inference.py`

Country is treated as an open set of string labels everywhere; token frequency
thresholds are recomputed from whichever corpus is being blocked, so unseen
countries (e.g. France) need no code change.

## Output format

Both files are tab separated with one row per test Source 1 entity:

| File | Columns |
| ---- | ------- |
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` |

ID lists are comma separated, contain only S2-/S3- ids that exist in the test
set, have no duplicates, and `matched_entity_ids` is always a subset of
`candidate_entity_ids`. Entities with no match get an empty list.
