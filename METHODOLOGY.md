# Methodology — index

The authoritative, submission-ready write-up is **[`Documentation_template.md`](Documentation_template.md)**.
The runnable pipeline and its reproduction steps are in
**[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)**.

This file only maps the repository.

## Pipeline

```
raw sources
  -> normalise            notebooks/03, code/.../src/normalization.py
  -> blocking (frozen + advanced)  notebooks/04, 09, code/.../src/blocking.py
  -> pairwise features    notebooks/05, 06, code/.../src/features.py
  -> model (HistGB)       notebooks/06, code/.../src/training.py
  -> S1 decision layer    notebooks/07, code/.../src/decision.py
  -> submission           notebooks/08, code/.../src/inference.py
```

## Notebooks

| Notebook | Role |
| -------- | ---- |
| `01_Environment_Setup.ipynb` | Colab / Drive setup |
| `02_EDA.ipynb` | Exploratory analysis |
| `03_Normalization.ipynb` | Cleaning + `name_core` |
| `04_Blocking.ipynb` | Blocking experiments (frozen ~87.24% S2 recall) |
| `05_Feature_Engineering.ipynb` | Feature / hard-negative exploration |
| `06_Model.ipynb` | Pairwise features + model training |
| `07_Decision_Layer.ipynb` | S1 decision layer + macro F0.5 tuning |
| `08_Test_Inference.ipynb` | Test inference + submission files |
| `09_Blocking_Improvement.ipynb` | Measures frozen vs improved blocking recall |

## Submission artefacts

* `output/matching_results.tsv` — final matches (scored)
* `output/candidate_pairs.tsv` — candidate set fed to the model
* `code/business_entity_resolution/` — runnable pipeline
* `Documentation_template.md` — filled methodology document

Validate before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
