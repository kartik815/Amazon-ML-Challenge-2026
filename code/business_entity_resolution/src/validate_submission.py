"""Local submission validator (stdlib only).

Mirrors the rules of the official ``utils/validate_submission.py`` so a broken
submission is caught locally:

* tab separated with the exact column names
* every test Source 1 entity present exactly once
* id lists contain only S2-/S3- ids that exist in the test set
* no duplicate ids inside a list
* final matches are a subset of the candidate set

Usage:
    python -m src.validate_submission \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import argparse
import csv
import os
import sys


def _read_ids(path):
    ids = set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header:
            return ids
        try:
            idx = header.index("entity_id")
        except ValueError:
            idx = 0
        for row in reader:
            if row and len(row) > idx and row[idx].strip():
                ids.add(row[idx].strip())
    return ids


def _read_table(path, id_column):
    if not os.path.exists(path):
        return None, None, [f"file not found: {path}"]
    issues = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = list(csv.reader(handle, delimiter="\t"))
    if not reader:
        return None, None, ["file is empty"]
    header = reader[0]
    if header != ["source1_entity_id", id_column]:
        issues.append(f"unexpected header {header}, expected ['source1_entity_id', '{id_column}']")
        if "source1_entity_id" not in header or id_column not in header:
            return None, None, issues
    s1_idx = header.index("source1_entity_id")
    id_idx = header.index(id_column)

    rows = {}
    duplicate_rows = set()
    for row in reader[1:]:
        if not row:
            issues.append("blank line found")
            continue
        s1 = row[s1_idx].strip() if len(row) > s1_idx else ""
        values = row[id_idx].strip() if len(row) > id_idx else ""
        if s1 in rows:
            duplicate_rows.add(s1)
        rows[s1] = values

    if duplicate_rows:
        issues.append(f"{len(duplicate_rows)} duplicate source1_entity_id rows")

    return header, rows, issues


def _check_ids(rows, id_field, valid_ids, issues):
    for s1, values in rows.items():
        if not values:
            continue
        parts = [p.strip() for p in values.split(",")]
        if any(p == "" for p in parts):
            issues.append(f"{s1}: empty id inside list")
        if len(parts) != len(set(parts)):
            issues.append(f"{s1}: duplicate ids inside list")
        for pid in parts:
            if not (pid.startswith("S2-") or pid.startswith("S3-")):
                issues.append(f"{s1}: id '{pid}' is not an S2-/S3- id")
            elif pid not in valid_ids:
                issues.append(f"{s1}: id '{pid}' does not exist in the test set")


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_ids = _read_ids(os.path.join(test_dir, "test_source1.tsv"))
    s2_ids = _read_ids(os.path.join(test_dir, "test_source2.tsv"))
    s3_ids = _read_ids(os.path.join(test_dir, "test_source3.tsv"))
    valid_ids = s2_ids | s3_ids

    if not s1_ids:
        issues.append("no Source 1 entities found in test_source1.tsv")

    _, matches, m_issues = _read_table(matching_path, "matched_entity_ids")
    issues.extend(m_issues)

    _, candidates, c_issues = _read_table(candidate_path, "candidate_entity_ids")
    issues.extend(c_issues)

    if matches is not None:
        _check_ids(matches, "matched_entity_ids", valid_ids, issues)
        missing = s1_ids - set(matches)
        if missing:
            issues.append(f"{len(missing)} test S1 entities missing from matching_results.tsv")

    if candidates is not None:
        _check_ids(candidates, "candidate_entity_ids", valid_ids, issues)
        missing = s1_ids - set(candidates)
        if missing:
            issues.append(f"{len(missing)} test S1 entities missing from candidate_pairs.tsv")

    if matches is not None and candidates is not None:
        for s1, values in matches.items():
            if not values:
                continue
            cand_set = set(
                p.strip() for p in candidates.get(s1, "").split(",") if p.strip()
            )
            for pid in (p.strip() for p in values.split(",")):
                if pid and pid not in cand_set:
                    issues.append(
                        f"{s1}: matched id '{pid}' is not in candidate_pairs.tsv"
                    )

    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args(argv)

    issues = validate(args.matching, args.candidate, args.test_dir)

    if issues:
        print("FAIL")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
