"""
Stream the AmazonQAC train split (~395M rows) from HuggingFace and keep a
random 10% sample.

    python src/download_amazon_qac.py

Output: data/amazon_qac_sample_10pct.parquet
"""

import os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SAMPLE_RATE = 0.10
SEED = 1234
BATCH_SIZE = 100_000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
PARQUET_OUT = os.path.join(DATA_DIR, "amazon_qac_sample_10pct.parquet")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ds = load_dataset("amazon/AmazonQAC", split="train", streaming=True)

    writer, buf = None, []
    seen = kept = 0
    t0 = time.time()

    def flush():
        nonlocal writer
        table = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_OUT, table.schema, compression="snappy")
        writer.write_table(table)
        buf.clear()

    for row in ds:
        seen += 1
        if rng.random() >= SAMPLE_RATE:
            continue
        buf.append(row)
        kept += 1
        if len(buf) >= BATCH_SIZE:
            flush()
            print(f"{seen:>12,} seen | {kept:>10,} kept | {seen / (time.time() - t0):,.0f} rows/s")

    if buf:
        flush()
    if writer is not None:
        writer.close()
    print(f"Done in {(time.time() - t0) / 60:.1f} min: kept {kept:,} of {seen:,} -> {PARQUET_OUT}")


if __name__ == "__main__":
    main()
