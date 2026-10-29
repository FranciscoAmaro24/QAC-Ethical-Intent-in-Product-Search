"""
Match the ethical vocabulary against the last typed prefix and the final query
of every query in the 5% sample.

    python src/ethical_analysis.py

Outputs:
    data/ethical_data_final_5pct.csv  one row per matched term (token level)
    data/ethical_logical_5pct.csv     one row per query, with a boolean
                                      <category>_prefix / <category>_final pair
"""

import os, re, argparse, warnings
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")

# Regex patterns with word boundaries, five categories, 89 terms
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

PATTERNS = {cat: re.compile("|".join(terms), re.IGNORECASE)
            for cat, terms in ETHICAL_VOCABULARY.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(DATA_DIR, "amazon_qac_5pct_ethics_filtered.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"query_id": "int64"})
    # The input already lacks each query's first prefix; keep its last one
    df = df.groupby("query_id", sort=False).tail(1).reset_index(drop=True)
    print(f"{len(df):,} queries")

    prefix = df["prefixes"].fillna("").str.lower()
    final = df["final_search_term"].fillna("").str.lower()
    for cat, pat in PATTERNS.items():
        df[f"{cat}_prefix"] = prefix.str.findall(pat.pattern)
        df[f"{cat}_final"] = final.str.findall(pat.pattern)

    cols = [f"{cat}_{side}" for cat in PATTERNS for side in ("prefix", "final")]
    df = df[df[cols].apply(lambda col: col.str.len() > 0).any(axis=1)].reset_index(drop=True)
    print(f"{len(df):,} queries with a match")

    # One row per combination of matched terms; None where a side has no match
    for col in cols:
        df[col] = df[col].apply(lambda x: x if len(x) > 0 else [None])
        df = df.explode(col, ignore_index=True)
    df = df[df[cols].notna().any(axis=1)].reset_index(drop=True)

    out = os.path.join(DATA_DIR, "ethical_data_final_5pct.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df):,} rows)")

    flags = df.copy()
    flags[cols] = flags[cols].notna()
    flags = flags.drop_duplicates().reset_index(drop=True)
    out = os.path.join(DATA_DIR, "ethical_logical_5pct.csv")
    flags.to_csv(out, index=False)
    print(f"Saved {out} ({len(flags):,} rows)")


if __name__ == "__main__":
    main()
