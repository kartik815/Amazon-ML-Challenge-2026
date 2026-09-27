# Business Entity Resolution — Methodology

Team submission for the Amazon ML Challenge 2026 — Business Entity Resolution.

## 1. Problem and objective

Source 1 (S1) is the deduplicated reference. For every S1 business we must find
all matching records in Source 2 and Source 3, where an S1 entity may match
zero, one or many records. Evaluation is macro **F0.5** (beta = 0.5) averaged
over S1 entities; precision is weighted twice as heavily as recall and
correctly predicting an empty list for a singleton earns a full 1.0.

Everything is offline. No external databases, APIs, geocoding, web lookup or
scraping are used.

## 2. Methodology used

The pipeline is a classic **blocking → feature engineering → pairwise learning →
per-entity decision** cascade:

```
raw sources
   │  normalisation (NFKC, casing, abbreviation + legal-suffix normalisation,
   │                 name_core, token counts)
   ▼
frozen blocking  (exact name ∪ name token ∪ address token, country keyed)
   │  candidate_pairs (the exact set fed to the model)
   ▼
pairwise features (6 base + 18 complementary = 24)
   │
   ▼
HistGradientBoostingClassifier  →  P(match | pair)
   │
   ▼
S1-level decision layer (threshold tuned on macro F0.5)
   │
   ▼
matching_results.tsv / candidate_pairs.tsv
```

Stage 1 is the ceiling: no true match dropped by blocking can ever be
recovered by the model, so the blockers were tuned for recall while keeping the
candidate set tractable.

## 3. Candidate generation / blocking strategy

**Baseline (frozen) — three blockers unioned:**

1. **Exact name block** — key `(country_norm, name_core)`.
2. **Name-token block** — token, keeping tokens with frequency in `[2, 500]` so
   both singletons and ubiquitous stop-words are excluded.
3. **Address-token block** — same frequency filter over address tokens.

**Improved blocker set — adds two complementary blockers:**

4. **Character n-gram block (`char3`)** — shared 3-grams with frequency in
   `[2, 1000]`, requiring at least 2 shared grams. Recovers typos and local
   spelling noise that share no whole token.
5. **Phonetic block** — a country-keyed *consonant skeleton* of the full name
   and of the first meaningful token. Vowels are dropped and equivalence
   classes are collapsed (`g/j`, `c/k/q`, `s/z`, `v/w`) so transliteration and
   cross-script pairs collapse to the same key: `green logistics` and
   `grina lajistiks` both map to `jrnljstks`. This directly targets the ~26% of
   missed matches that have non-Latin names.

The union of all five blockers is the improved ceiling; block sizes above a
cap are dropped so a very common skeleton cannot explode the candidate set.
Notebook `09_Blocking_Improvement.ipynb` measures each blocker's recall, the
union recall of both sets, and the candidate volume so the trade-off is
quantified on the training data before it is adopted.

Country is only ever used as an exact normalised key — never hard-coded,
filtered or one-hot encoded — so the unseen test country (France) works with no
code change. Token-frequency thresholds are recomputed on whichever corpus is
being blocked, making the step self-contained for train and test.

Frozen S2 blocking recall on the training set: **87.24 %**
(3,222,405 / 3,693,619 true S2 matches). This is the recall ceiling, not the
final matching score; the `char3` and `phonetic` blockers are added to raise it.

`candidate_pairs.tsv` is written **after** all blocking stages — it is exactly
the candidate set the model scores, one comma-separated list per test S1.

## 4. Model architecture and feature engineering

**Features (24).** Base string similarities, all recomputed in one code path so
train and test cannot diverge:

`name_char_similarity`, `name_3gram_jaccard`, `name_token_jaccard`,
`address_char_similarity`, `address_token_jaccard`, `country_exact`.

Complementary:

| Group | Features |
| ----- | -------- |
| Name  | `name_exact`, `name_core_exact`, `name_length_ratio`, `name_token_count_diff`, `name_first_token_match`, `name_last_token_match`, `name_contains` |
| Address | `address_exact`, `address_length_ratio`, `address_number_match`, `address_number_jaccard` |
| Transliteration | `name_translit_exact`, `name_translit_core_exact`, `name_translit_char_similarity`, `name_translit_3gram_jaccard`, `name_translit_token_jaccard` |
| Source | `source_is_s2`, `source_is_s3` |

**Transliteration.** Indian scripts are detected and transliterated exactly
once (ITRANS), then NFKD-stripped, lower-cased and cleaned. The earlier bug that
transliterated already-transliterated output was fixed; transliteration is an
additional signal, never a replacement for the normalised name.

**Model.** `HistGradientBoostingClassifier` (`max_iter=300`,
`learning_rate=0.05`, `max_leaf_nodes=31`, `min_samples_leaf=30`,
`l2_regularization=1.0`, `random_state=42`). Gradient-boosted trees handle
missing/NaN similarities natively and are fast on the tabular candidate set.

**Validation split.** Done **per S1 entity** via a deterministic MD5 hash bucket
(~2 % of S1 entities to validation). No S1 entity ever appears in both splits.
Validation entities keep *all* their candidates so the decision layer is tuned
on realistic candidate sets.

## 5. Decision layer and metric

Pair probabilities are not the final answer. For each S1 entity:

```
cut       = max(threshold, relative × max_probability_within_S1)
predicted = { candidate : P(match) >= cut }
```

`(threshold, relative)` is swept on validation to maximise macro F0.5, where the
per-entity score is

```
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

with `empty/empty → 1.0`, `empty/non-empty → 0.0`, `non-empty/empty → 0.0`.
Two numbers are reported: macro F0.5 over candidate-covered entities, and the
**true** macro F0.5 over all S1 entities (which additionally exposes recall lost
to blocking).

## 6. Other relevant information

**Memory discipline.** The dataset can exhaust a default Colab session. The
dominant cost was indexing the noisy sources and holding their text: the source
posting lists plus the source text lookup together run to several GB per source.
We therefore **invert the join**: only the reference side (S1) is indexed and
kept in memory, while S2/S3 are streamed chunk by chunk. For each source record
the blocking keys are computed once and looked up against the S1 index, so
candidate pairs are identical to the source-indexed version with a fraction of
the memory. Training accumulates feature rows into small numpy buffers (never a
Python list of per-row lists) and keeps ids only for validation rows. Inference
spills scored pairs into bounded buckets keyed by S1 position, so the two output
files are produced one slice at a time rather than holding every pair.

All stages stream with `chunksize`, process one source at a time, use `usecols`
and explicit dtypes, prefer vectorised `.str` operations over `.apply`, and
`gc.collect()` between stages. Chunk size and bucket count are tunable
(`--chunk-size`, `--buckets`) and all intermediate artefacts are checkpointed to
persistent storage so a crash never loses earlier work.

**Reproducibility.** `code/business_entity_resolution/` reproduces both outputs
end-to-end:

```bash
pip install -r requirements.txt
python run_pipeline.py all --data-dir dataset --output-dir output
python -m src.validate_submission \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

**Output format.** One row per test S1 entity, tab separated, id lists comma
separated, only S2-/S3- ids that exist in the test set, no duplicates, and
matches always a subset of candidates. Singletons are emitted with an empty
`matched_entity_ids`.

**Licences.** scikit-learn (BSD-3), pandas (BSD-3), numpy (BSD-3), rapidfuzz
(MIT), indic-transliteration (MIT). No pretrained models; well under the 8B
parameter limit.

**Known limitations and next steps.** The relaxed character n-gram and
phonetic blockers (`char3`, `phonetic`) are implemented and can be enabled with
`--advanced-blocking`; `09_Blocking_Improvement.ipynb` quantifies the gain. Any
remaining ceiling is dominated by very short or highly noisy names. Pairwise
probabilities are correlated within an S1, so a ranking-aware (learning-to-rank)
objective could improve the decision layer. A cross-encoder text model is
intentionally not used to stay offline and lightweight.
