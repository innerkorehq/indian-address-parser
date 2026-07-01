"""Build the raw (unlabeled) Indian addresses dataset from dataprep.db.

Bank/BC source addresses have PII redaction applied (see redact.py) since
they're KYC-style records that can embed real customer phone numbers and
name markers. MCA source is left untouched — its "C/O <name>" convention
names company directors, already public via MCA's own CIN disclosure.
"""
import argparse
import sqlite3

import pandas as pd
from redact import has_pii, redact


def build(db_path: str, out_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT record_id, source_type, raw_address, lifecycle_state,
               aux_state, aux_pincode_declared, aux_lat, aux_lon, source_metadata
        FROM raw_records
        """
    ).fetchall()

    records = []
    redacted_count = 0
    for r in rows:
        addr = r["raw_address"]
        if r["source_type"] == "bank" and has_pii(addr):
            addr = redact(addr)
            redacted_count += 1
        records.append(
            {
                "record_id": r["record_id"],
                "source_type": r["source_type"],
                "raw_address": addr,
                "lifecycle_state": r["lifecycle_state"],
                "aux_state": r["aux_state"],
                "aux_pincode_declared": r["aux_pincode_declared"],
                "aux_lat": r["aux_lat"],
                "aux_lon": r["aux_lon"],
                "source_metadata": r["source_metadata"],
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} records to {out_path}")
    print(f"PII-redacted (bank source): {redacted_count:,}")
    print(df["source_type"].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="/Users/baneet/Desktop/claude/IndianAddressTokenizer/DataPrep/dataprep.db")
    p.add_argument("--out", default="raw_addresses.parquet")
    args = p.parse_args()
    build(args.db, args.out)
