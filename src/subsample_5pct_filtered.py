"""
Create a 5 pct sample from the 10 pct parquet, filtering for ethics keywords.

Reads the parquet in row-group batches (never fully in RAM), samples ~50 pct
of each batch (50 pct of 10 pct = 5 pct of full), pre-filters on the ethics
vocabulary, explodes prefixes, and writes matched rows incrementally to CSV.

Output
------
    amazon_qac_5pct_ethics_filtered.csv
"""

import os, time, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED        = 1234
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
PARQUET_IN  = os.path.join(DATA_DIR, "amazon_qac_sample_10pct.parquet")
CSV_OUT     = os.path.join(DATA_DIR, "amazon_qac_5pct_ethics_filtered.csv")
SAMPLE_RATE = 0.50

# Union of all five ethics categories (no word-boundary anchors here;
# the combined pattern adds them).
_ETHICS_TERMS = [
    # Environmental sustainability
    r"organic", r"natural", r"non[- ]?toxic", r"chemical[- ]?free",
    r"plastic[- ]?free", r"durable", r"reusable", r"refillable",
    r"second[- ]?hand", r"biodegradable", r"compostable", r"recyclable",
    r"zero[- ]?waste", r"upcycled", r"refurbished", r"energy[- ]?efficient",
    r"water[- ]?saving", r"low[- ]?carbon", r"carbon[- ]?neutral", r"eco[- ]?friendly",
    # Fair trade / labour
    r"fair[- ]?trade", r"ethically sourced", r"ethically made",
    r"responsibly sourced", r"living wage", r"fair wage", r"fair pay",
    r"worker rights", r"labor rights", r"no child labor",
    r"child[- ]?labor[- ]?free", r"forced[- ]?labor[- ]?free",
    r"sweatshop[- ]?free", r"union[- ]?made",
    r"handmade", r"hand[- ]?made", r"artisan",
    r"locally made", r"small[- ]?batch",
    # Animal welfare
    r"cruelty[- ]?free", r"not tested on animals", r"animal[- ]?test[- ]?free",
    r"vegan", r"vegetarian", r"free[- ]?range", r"cage[- ]?free",
    r"pasture[- ]?raised", r"humane", r"animal welfare",
    r"grass[- ]?fed", r"hormone[- ]?free", r"antibiotic[- ]?free",
    r"no hormones", r"no antibiotics",
    # Privacy & data ethics
    r"privacy", r"privacy[- ]?friendly", r"privacy[- ]?first",
    r"privacy[- ]?focused", r"no tracking", r"tracker[- ]?free",
    r"encrypted", r"end[- ]?to[- ]?end encrypted", r"secure",
    r"secure by design", r"data[- ]?minimal", r"local[- ]?only",
    r"on[- ]?device", r"no cloud", r"open[- ]?source", r"offline",
    r"no[- ]?subscription", r"ad[- ]?free",
    # Accessibility & inclusion
    r"accessible", r"accessibility", r"inclusive design", r"universal design",
    r"disability[- ]?friendly", r"screen[- ]?reader compatible",
    r"braille", r"large print", r"wheelchair[- ]?friendly",
    r"hearing[- ]?friendly", r"low[- ]?vision friendly",
    r"large[- ]?text", r"sensory[- ]?friendly", r"ergonomic",
    r"adaptive", r"handicap", r"hearing[- ]?aid",
]

ETHICS_PATTERN = r"\b(?:" + "|".join(_ETHICS_TERMS) + r")\b"


def vec_match(series):
    return series.str.contains(ETHICS_PATTERN, case=False, regex=True, na=False)


def process_batch(batch_df, rng):
    """Sample, pre-filter on final_search_term, explode prefixes, keep matches."""
    n = len(batch_df)
    keep_n = max(1, round(n * SAMPLE_RATE))
    idx = rng.choice(n, size=keep_n, replace=False)
    idx.sort()
    df = batch_df.iloc[idx].reset_index(drop=True)

    # Pre-filter on final search term
    df = df[vec_match(df["final_search_term"])].reset_index(drop=True)
    if df.empty:
        return df

    # Explode prefixes list column
    df = df.explode("prefixes").reset_index(drop=True)

    # Drop first prefix per query (mirrors R: slice_tail(n=-1))
    first = df.groupby("query_id", sort=False).cumcount() == 0
    df = df[~first].reset_index(drop=True)
    if df.empty:
        return df

    # Keep rows where either prefix or final matches
    return df[vec_match(df["final_search_term"]) |
              vec_match(df["prefixes"])].reset_index(drop=True)


def main():
    t0 = time.time()
    print(f"Input  : {PARQUET_IN}")
    print(f"Output : {CSV_OUT}")

    pf = pq.ParquetFile(PARQUET_IN)
    n_groups   = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"Parquet: {total_rows:,} rows in {n_groups} row-groups")
    print(f"Sampling {SAMPLE_RATE*100:.0f}% per batch\n")

    rng         = np.random.default_rng(SEED)
    matched_all = 0
    first_write = True

    for i in range(n_groups):
        batch_df = pf.read_row_group(i).to_pandas()
        result   = process_batch(batch_df, rng)
        matched_all += len(result)

        if not result.empty:
            result.to_csv(CSV_OUT, mode="w" if first_write else "a",
                          header=first_write, index=False)
            first_write = False

        print(f"  Batch {i+1:>3}/{n_groups}  "
              f"{len(batch_df):>8,} rows -> {len(result):>6,} matched  "
              f"(cumulative {matched_all:,})")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  —  {matched_all:,} rows written to {CSV_OUT}")


if __name__ == "__main__":
    main()
