"""Central configuration shared across the whole pipeline.

Only path-independent constants live here so every stage agrees on the same
feature order, thresholds and vocabulary.
"""

# ------------------------------------------------------------------
# Feature order. This list is the single source of truth: the trained
# model, the training matrix and inference all build their columns from it.
# ------------------------------------------------------------------
BASE_FEATURE_COLUMNS = [
    "name_char_similarity",
    "name_3gram_jaccard",
    "name_token_jaccard",
    "address_char_similarity",
    "address_token_jaccard",
    "country_exact",
]

EXTRA_FEATURE_COLUMNS = [
    "name_exact",
    "name_core_exact",
    "name_length_ratio",
    "name_token_count_diff",
    "name_first_token_match",
    "name_last_token_match",
    "name_contains",
    "address_exact",
    "address_length_ratio",
    "address_number_match",
    "address_number_jaccard",
    "name_translit_exact",
    "name_translit_core_exact",
    "name_translit_char_similarity",
    "name_translit_3gram_jaccard",
    "name_translit_token_jaccard",
    "source_is_s2",
    "source_is_s3",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS

# ------------------------------------------------------------------
# Normalisation vocabularies
# ------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    "limited": "ltd",
    "ltd": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "highway": "hwy",
    "hwy": "hwy",
    "place": "pl",
    "pl": "pl",
    "court": "ct",
    "ct": "ct",
    "square": "sq",
    "sq": "sq",
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste",
    "building": "bldg",
    "bldg": "bldg",
}

# Suffixes removed from the tail of a name to obtain its "core".
CORE_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "plc", "pvt", "private", "co", "company",
    "org", "organization",
}

NAME_STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "company", "co", "corp",
    "corporation", "group", "services", "partners",
    "holdings", "associates", "center", "india",
    "and", "of",
}

# ------------------------------------------------------------------
# Blocking / sampling / split
# ------------------------------------------------------------------
MIN_TOKEN_FREQ = 2
MAX_TOKEN_FREQ = 500

# ------------------------------------------------------------------
# Blockers
#
# FROZEN_BLOCKERS reproduces the original 87.24% candidate set.
# ADVANCED_BLOCKERS adds two complementary blockers that recover matches
# the frozen set misses (typos and cross-script / transliteration pairs).
# ------------------------------------------------------------------
FROZEN_BLOCKERS = ["exact_name", "name_token", "address_token"]
ADVANCED_BLOCKERS = [
    "exact_name",
    "name_token",
    "address_token",
    "char3",
    "phonetic",
]

# Character n-gram blocker: keep n-grams whose S2/S3 frequency is in the
# window and require at least CHAR3_MIN_SHARED shared n-grams.
CHAR3_N = 3
CHAR3_MIN_FREQ = 2
CHAR3_MAX_FREQ = 1000
CHAR3_MIN_SHARED = 2

# Phonetic blocker: block sizes above this are dropped to bound candidate
# volume (a very common consonant skeleton is not discriminative).
MAX_PHONETIC_BLOCK = 500

CHUNK_SIZE = 200_000
FEATURE_BATCH = 200_000

# Inference spilling: scored pairs are bucketed by S1 position so the outputs
# can be produced one bounded slice at a time instead of holding every pair.
INFERENCE_BUCKETS = 256
BUCKET_BUFFER_ROWS = 1_000_000

RANDOM_SEED = 42

# Deterministic S1 split for validation (~2%).
VAL_HASH_MOD = 1000
VAL_HASH_REM = 20

# Training keeps every positive and samples negatives.
TRAIN_NEG_RATE = 0.10
MAX_TRAIN_POS = 400_000
MAX_TRAIN_NEG = 800_000

# Model hyper-parameters (baseline HistGradientBoosting).
MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": RANDOM_SEED,
}

# Decision layer defaults (overridden by the tuned config file).
DEFAULT_THRESHOLD = 0.95
DEFAULT_RELATIVE = 0.0
