"""Normalisation of raw source TSVs.

Produces cleaned copies that keep every original column and add:

    name_norm, name_core, address_norm, country_norm,
    name_missing, address_missing, name_token_count, address_token_count

Row count and entity_id are always preserved.
"""

import gc
import os
import re

import pandas as pd

from .config import ADDRESS_ABBREVIATIONS, CHUNK_SIZE, LEGAL_SUFFIX_MAP
from .text_utils import create_name_core

_ID_DTYPE = {
    "entity_id": "string",
    "business_name": "string",
    "business_address": "string",
    "country": "string",
}


def normalize_basic_series(series):
    return (
        series.fillna("")
        .astype("string")
        .str.normalize("NFKC")
        .str.lower()
        .str.replace(r"[/\\|,_;:]+", " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize_name_series(series):
    result = normalize_basic_series(series)
    for old, new in LEGAL_SUFFIX_MAP.items():
        result = result.str.replace(rf"\b{re.escape(old)}\b", new, regex=True)
    return result


def normalize_address_series(series):
    result = normalize_basic_series(series)
    for old, new in ADDRESS_ABBREVIATIONS.items():
        result = result.str.replace(rf"\b{re.escape(old)}\b", new, regex=True)
    return result


def normalize_country_series(series):
    return (
        series.fillna("")
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def add_cleaned_columns(df):
    df["name_norm"] = normalize_name_series(df["business_name"])
    df["address_norm"] = normalize_address_series(df["business_address"])
    df["country_norm"] = normalize_country_series(df["country"])
    df["name_core"] = create_name_core(df["name_norm"])

    df["name_missing"] = (df["name_norm"].str.len() == 0).astype("int8")
    df["address_missing"] = (df["address_norm"].str.len() == 0).astype("int8")
    df["name_token_count"] = df["name_norm"].str.count(r"\S+").astype("int16")
    df["address_token_count"] = df["address_norm"].str.count(r"\S+").astype("int16")

    return df


def normalize_file(input_path, output_path, chunk_size=CHUNK_SIZE):
    """Normalise one TSV, streaming so memory stays flat."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    first = True
    rows = 0

    for chunk in pd.read_csv(
        input_path, sep="\t", dtype=_ID_DTYPE, chunksize=chunk_size
    ):
        original = len(chunk)
        chunk = add_cleaned_columns(chunk)
        assert len(chunk) == original, "row count changed during normalisation"

        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False
        rows += original

        del chunk
        gc.collect()

    return rows


def normalize_all(data_dir, work_dir, chunk_size=CHUNK_SIZE):
    """Normalise train and test sources into ``work_dir/cleaned``."""
    cleaned_dir = os.path.join(work_dir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    files = [
        ("train", "train_source1.tsv"),
        ("train", "train_source2.tsv"),
        ("train", "train_source3.tsv"),
        ("test", "test_source1.tsv"),
        ("test", "test_source2.tsv"),
        ("test", "test_source3.tsv"),
    ]

    written = {}
    for split, filename in files:
        src = os.path.join(data_dir, split, filename)
        if not os.path.exists(src):
            continue
        dst = os.path.join(cleaned_dir, filename.replace("_source", "_s").replace(".tsv", "_cleaned.tsv"))
        rows = normalize_file(src, dst, chunk_size=chunk_size)
        written[dst] = rows
        print(f"normalised {filename}: {rows:,} rows -> {dst}")

    return cleaned_dir
