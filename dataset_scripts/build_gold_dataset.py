"""Build the gold-standard (span-labeled) Indian addresses dataset from dataprep.db.

Bank/BC gold records that contain PII patterns (phone numbers, S/O|D/O|W/O|C/O
relational name markers referring to a private customer, not a company
director) are DROPPED rather than redacted in place: the gold spans_json is
character-offset based into raw_address, and in-place text redaction changes
string length, which would silently shift every downstream span offset —
some of which (e.g. "poi") can legitimately overlap a "C/O <name>" region.
Dropping the small number of affected records (verified: ~174/2500 bank gold
records) is safer than attempting offset-remapping surgery on curated gold
labels. MCA gold records are never dropped for this reason — their "C/O
<name>" pattern names a company director, already public via MCA's own CIN
disclosure, not a private individual.
"""
import argparse
import json
import sqlite3

import pandas as pd
from redact import has_pii

ALL_FIELDS = (
    "houseNumber", "houseName", "poi", "street", "subsubLocality", "subLocality",
    "locality", "village", "subDistrict", "district", "city", "state", "pincode",
)


def spans_to_fields(raw_address: str, spans: list[dict]) -> dict:
    fields = {f: None for f in ALL_FIELDS}
    for span in spans:
        label = span.get("label")
        if label not in fields:
            continue
        text = raw_address[span["start"]:span["end"]].strip()
        if not text:
            continue
        fields[label] = (fields[label] + " " + text) if fields[label] else text
    return fields


def build(db_path: str, out_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT record_id, source_type, raw_address, review_state, reviewer,
               reviewed_at, spans_json, rule_confidence_at_review_time
        FROM gold_labels
        """
    ).fetchall()

    records = []
    dropped = 0
    for r in rows:
        addr = r["raw_address"]
        if r["source_type"] == "bank" and has_pii(addr):
            dropped += 1
            continue
        spans = json.loads(r["spans_json"])
        fields = spans_to_fields(addr, spans)
        records.append(
            {
                "record_id": r["record_id"],
                "source_type": r["source_type"],
                "raw_address": addr,
                "review_state": r["review_state"],
                "reviewer": r["reviewer"],
                "reviewed_at": r["reviewed_at"],
                "rule_confidence_at_review_time": r["rule_confidence_at_review_time"],
                "spans": spans,
                **fields,
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} records to {out_path}")
    print(f"Dropped (bank source, PII pattern detected): {dropped:,}")
    print(df["source_type"].value_counts())
    print(df["reviewer"].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="/Users/baneet/Desktop/claude/IndianAddressTokenizer/DataPrep/dataprep.db")
    p.add_argument("--out", default="gold_addresses.parquet")
    args = p.parse_args()
    build(args.db, args.out)
