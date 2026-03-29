"""
Token-level ethical-keyword retention analysis on AmazonQAC prefix data.

For each of five ethical vocabulary categories, we extract keyword matches
from both the last typed prefix and the final (autocompleted) search term,
then classify every match as Retained, Added, or Dropped by QAC.

Categories:
    1. Environmental Sustainability
    2. Fair Trade, Labour & Human Rights
    3. Animal Welfare
    4. Privacy & Data Ethics
    5. Accessibility & Inclusion

Usage
-----
    python ethical_analysis.py                        # default (1 pct sample)
    python ethical_analysis.py --total 19750000 \\
        --csv amazon_qac_5pct_ethics_filtered.csv \\
        --pre-filtered --output-suffix _5pct          # 5 pct sample
"""

import os, re, argparse, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV  = os.path.join(DATA_DIR, "amazon_qac_tokens_all.csv")
DEFAULT_TOTAL = 3_950_000

# ---------------------------------------------------------------------------
# Ethical vocabulary (regex patterns with word boundaries)
# ---------------------------------------------------------------------------
ETHICAL_VOCABULARY = {
    "environmental_sustainability": [
        r"\borganic\b",       r"\bnatural\b",          r"\bnon[- ]?toxic\b",
        r"\bchemical[- ]?free\b", r"\bplastic[- ]?free\b", r"\bdurable\b",
        r"\breusable\b",      r"\brefillable\b",       r"\bsecond[- ]?hand\b",
        r"\bbiodegradable\b", r"\bcompostable\b",      r"\brecyclable\b",
        r"\bzero[- ]?waste\b", r"\bupcycled\b",        r"\brefurbished\b",
        r"\benergy[- ]?efficient\b", r"\bwater[- ]?saving\b",
        r"\blow[- ]?carbon\b", r"\bcarbon[- ]?neutral\b", r"\beco[- ]?friendly\b",
    ],
    "fair_trade_labor_human_rights": [
        r"\bfair[- ]?trade\b",    r"\bethically sourced\b", r"\bethically made\b",
        r"\bresponsibly sourced\b", r"\bliving wage\b",     r"\bfair wage\b",
        r"\bfair pay\b",           r"\bworker rights\b",   r"\blabor rights\b",
        r"\bno child labor\b",     r"\bchild[- ]?labor[- ]?free\b",
        r"\bforced[- ]?labor[- ]?free\b", r"\bsweatshop[- ]?free\b",
        r"\bunion[- ]?made\b",     r"\bhandmade\b",        r"\bhand[- ]?made\b",
        r"\bartisan\b",            r"\blocally made\b",    r"\bsmall[- ]?batch\b",
    ],
    "animal_welfare": [
        r"\bcruelty[- ]?free\b",   r"\bnot tested on animals\b",
        r"\banimal[- ]?test[- ]?free\b", r"\bvegan\b",    r"\bvegetarian\b",
        r"\bfree[- ]?range\b",     r"\bcage[- ]?free\b",  r"\bpasture[- ]?raised\b",
        r"\bhumane\b",             r"\banimal welfare\b",  r"\bgrass[- ]?fed\b",
        r"\bhormone[- ]?free\b",   r"\bantibiotic[- ]?free\b",
        r"\bno hormones\b",        r"\bno antibiotics\b",
    ],
    "privacy_data_ethics": [
        r"\bprivacy\b",            r"\bprivacy[- ]?friendly\b",
        r"\bprivacy[- ]?first\b",  r"\bprivacy[- ]?focused\b",
        r"\bno tracking\b",        r"\btracker[- ]?free\b",
        r"\bencrypted\b",          r"\bend[- ]?to[- ]?end encrypted\b",
        r"\bsecure\b",             r"\bsecure by design\b",
        r"\bdata[- ]?minimal\b",   r"\blocal[- ]?only\b",
        r"\bon[- ]?device\b",      r"\bno cloud\b",
        r"\bopen[- ]?source\b",    r"\boffline\b",
        r"\bno[- ]?subscription\b", r"\bad[- ]?free\b",
    ],
    "accessibility_inclusion": [
        r"\baccessible\b",         r"\baccessibility\b",
        r"\binclusive design\b",   r"\buniversal design\b",
        r"\bdisability[- ]?friendly\b", r"\bscreen[- ]?reader compatible\b",
        r"\bbraille\b",            r"\blarge print\b",
        r"\bwheelchair[- ]?friendly\b", r"\bhearing[- ]?friendly\b",
        r"\blow[- ]?vision friendly\b", r"\blarge[- ]?text\b",
        r"\bsensory[- ]?friendly\b", r"\bergonomic\b",
        r"\badaptive\b",           r"\bhandicap\b",       r"\bhearing[- ]?aid\b",
    ],
}

CATEGORY_LABELS = {
    "environmental_sustainability":  "Environmental Sustainability",
    "fair_trade_labor_human_rights": "Fair Trade / Labour / Human Rights",
    "animal_welfare":                "Animal Welfare",
    "privacy_data_ethics":           "Privacy & Data Ethics",
    "accessibility_inclusion":       "Accessibility & Inclusion",
}


def _build_pattern(terms):
    """OR-combine a list of regex fragments into a single compiled pattern."""
    nc = [t.replace("(", "(?:") for t in terms]
    return re.compile("(?:" + "|".join(nc) + ")", re.IGNORECASE)


PATTERNS     = {cat: _build_pattern(ts) for cat, ts in ETHICAL_VOCABULARY.items()}
ALL_TERMS    = [t for ts in ETHICAL_VOCABULARY.values() for t in ts]
ANY_PATTERN  = _build_pattern(ALL_TERMS)


def word_count(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return 0
    return len(re.split(r"\W+", text.strip()))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ethical keyword retention analysis.")
    ap.add_argument("--csv",            default=DEFAULT_CSV)
    ap.add_argument("--total",          type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--pre-filtered",   action="store_true",
                    help="Skip first-prefix removal (input already filtered)")
    ap.add_argument("--output-suffix",  default="")
    args = ap.parse_args()

    suffix = args.output_suffix
    print(f"Input : {args.csv}")
    print(f"Total : {args.total:,}  |  Pre-filtered: {args.pre_filtered}")

    # 1. Load
    df = pd.read_csv(args.csv, dtype={"query_id": "int64"})
    print(f"Loaded {len(df):,} rows")

    # 2. Drop first prefix per query (unless already filtered)
    if not args.pre_filtered:
        first = df.groupby("query_id", sort=False).cumcount() == 0
        df = df[~first].reset_index(drop=True)
        print(f"After dropping first prefix: {len(df):,}")

    # 3. Keep last prefix per query
    df = df.groupby("query_id", sort=False).tail(1).reset_index(drop=True)
    print(f"One row per query: {len(df):,}")

    # 4. Extract keyword matches per category
    prefix_lower = df["prefixes"].fillna("").str.lower()
    final_lower  = df["final_search_term"].fillna("").str.lower()

    for cat, pat in PATTERNS.items():
        df[f"{cat}_prefix_raw"] = prefix_lower.str.findall(pat.pattern)
        df[f"{cat}_final_raw"]  = final_lower.str.findall(pat.pattern)

    # 5. Keep rows with at least one keyword match in any category
    has_any = pd.Series(False, index=df.index)
    for cat in PATTERNS:
        has_any |= df[f"{cat}_prefix_raw"].apply(len) > 0
        has_any |= df[f"{cat}_final_raw"].apply(len) > 0
    df = df[has_any].reset_index(drop=True)
    print(f"Rows with keyword match: {len(df):,}")

    # 6. Explode each category independently
    for cat in PATTERNS:
        for side in ("prefix", "final"):
            raw = f"{cat}_{side}_raw"
            col = f"{cat}_{side}"
            df[col] = df[raw].apply(lambda x: x if len(x) > 0 else [None])

    for cat in PATTERNS:
        df = df.explode(f"{cat}_prefix", ignore_index=True)
        df = df.explode(f"{cat}_final",  ignore_index=True)

    df = df.drop(columns=[c for c in df.columns if c.endswith("_raw")])

    any_match = pd.Series(False, index=df.index)
    for cat in PATTERNS:
        any_match |= df[f"{cat}_prefix"].notna()
        any_match |= df[f"{cat}_final"].notna()
    df = df[any_match].reset_index(drop=True)
    print(f"After explode + filter: {len(df):,}")

    # 7. Save string-level matches
    out_data = os.path.join(DATA_DIR, f"ethical_data_final{suffix}.csv")
    df.to_csv(out_data, index=False)
    print(f"Saved {out_data}")

    # 8. Boolean version (deduplicated)
    fl = df.copy()
    for cat in PATTERNS:
        fl[f"{cat}_prefix"] = fl[f"{cat}_prefix"].notna()
        fl[f"{cat}_final"]  = fl[f"{cat}_final"].notna()
    fl = fl.drop_duplicates().reset_index(drop=True)

    out_logical = os.path.join(DATA_DIR, f"ethical_logical{suffix}.csv")
    fl.to_csv(out_logical, index=False)
    print(f"Saved {out_logical}  ({len(fl):,} rows)")

    # 9. Per-category retention statistics
    fl["prefix_term_count"] = fl["prefixes"].apply(word_count)
    fl["query_term_count"]  = fl["final_search_term"].apply(word_count)

    for cat, label in CATEGORY_LABELS.items():
        pcol, fcol = f"{cat}_prefix", f"{cat}_final"

        retained_tt = ((fl[pcol]) & (fl[fcol])).sum()
        same_len    = ((fl["query_term_count"] == fl["prefix_term_count"])
                       & (~fl[pcol]) & (fl[fcol])).sum()
        retained    = retained_tt + same_len

        added = ((fl["query_term_count"] >= fl["prefix_term_count"])
                 & (~fl[pcol]) & (fl[fcol])).sum()

        dropped = ((fl[pcol]) & (~fl[fcol])).sum()

        total = retained + added + dropped
        pct   = total / args.total * 100 if args.total else 0

        print(f"\n  {label}")
        print(f"    Retained : {retained:>8,}")
        print(f"    Added    : {added:>8,}")
        print(f"    Dropped  : {dropped:>8,}")
        print(f"    % of all : {pct:.3f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
