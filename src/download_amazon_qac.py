"""
Download the Amazon QAC dataset and create a 10% parquet sample.

Streams the full AmazonQAC train split (~395M rows) from HuggingFace,
randomly samples 10% (~39.5M rows) in a single pass, and saves the
result as a parquet file with the prefixes list column preserved.

Output
------
    amazon_qac_sample_10pct.parquet

Usage
-----
    python download_amazon_qac.py
"""

import os, time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from datasets import load_dataset

SAMPLE_RATE  = 0.10
SEED         = 1234
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PARQUET_OUT  = os.path.join(OUTPUT_DIR, "..",
                            "amazon_qac_sample_10pct.parquet")
BATCH_SIZE   = 100_000
ROW_GROUP    = 500_000


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("Downloading AmazonQAC (train) from HuggingFace — streaming mode")
    print(f"  Sample rate   : {SAMPLE_RATE*100:.0f}%")
    print(f"  Random seed   : {SEED}")
    print(f"  Output parquet: {PARQUET_OUT}")
    print("=" * 70)

    ds = load_dataset("amazon/AmazonQAC", split="train", streaming=True)

    writer = None
    schema = None
    buf = []
    total_seen = 0
    total_kept = 0
    t0 = time.time()

    for row in ds:
        total_seen += 1

        if rng.random() >= SAMPLE_RATE:
            continue

        buf.append(row)
        total_kept += 1

        if len(buf) >= BATCH_SIZE:
            table = pa.Table.from_pylist(buf)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(PARQUET_OUT, schema,
                                          compression="snappy")
            writer.write_table(table)
            buf.clear()

            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0
            print(f"  {total_seen:>12,} seen | {total_kept:>10,} kept | "
                  f"{rate:,.0f} rows/s")

    # Flush remaining rows
    if buf:
        table = pa.Table.from_pylist(buf)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(PARQUET_OUT, schema,
                                      compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Rows seen : {total_seen:,}")
    print(f"  Rows kept : {total_kept:,}")
    print(f"  Saved to  : {PARQUET_OUT}")


if __name__ == "__main__":
    main()
