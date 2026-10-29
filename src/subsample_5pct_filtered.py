"""
Take a 5% sample (50% of each row group of the 10% parquet) and keep the
prefix rows of queries whose final query matches the ethical vocabulary.

The first prefix of every query is dropped, and a row is kept if either the
prefix or the final query matches.

    python src/subsample_5pct_filtered.py

Output: data/amazon_qac_5pct_ethics_filtered.csv
"""

import os, time
import numpy as np
import pyarrow.parquet as pq

from ethical_analysis import ETHICAL_VOCABULARY

SEED = 1234
SAMPLE_RATE = 0.50
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
PARQUET_IN = os.path.join(DATA_DIR, "amazon_qac_sample_10pct.parquet")
CSV_OUT = os.path.join(DATA_DIR, "amazon_qac_5pct_ethics_filtered.csv")

ETHICS_PATTERN = r"\b(?:" + "|".join(
    t.replace(r"\b", "") for terms in ETHICAL_VOCABULARY.values() for t in terms) + r")\b"


def vec_match(series):
    return series.str.contains(ETHICS_PATTERN, case=False, regex=True, na=False)


def sample_batch(batch, rng):
    """Rows `SAMPLE_RATE` of the batch, drawn without replacement, in original order."""
    n = len(batch)
    idx = np.sort(rng.choice(n, size=max(1, round(n * SAMPLE_RATE)), replace=False))
    return batch.iloc[idx].reset_index(drop=True)


def process_batch(batch, rng):
    df = sample_batch(batch, rng)
    df = df[vec_match(df["final_search_term"])].reset_index(drop=True)
    if df.empty:
        return df

    df = df.explode("prefixes").reset_index(drop=True)
    first = df.groupby("query_id", sort=False).cumcount() == 0
    df = df[~first].reset_index(drop=True)
    if df.empty:
        return df
    return df[vec_match(df["final_search_term"]) | vec_match(df["prefixes"])].reset_index(drop=True)


def main():
    t0 = time.time()
    pf = pq.ParquetFile(PARQUET_IN)
    n_groups = pf.metadata.num_row_groups
    rng = np.random.default_rng(SEED)
    written = 0

    for i in range(n_groups):
        result = process_batch(pf.read_row_group(i).to_pandas(), rng)
        if not result.empty:
            result.to_csv(CSV_OUT, mode="w" if written == 0 else "a",
                          header=written == 0, index=False)
            written += len(result)
        print(f"row group {i + 1:>3}/{n_groups}: {len(result):>6,} rows (total {written:,})")

    print(f"Done in {time.time() - t0:.0f}s: {written:,} rows -> {CSV_OUT}")


if __name__ == "__main__":
    main()
