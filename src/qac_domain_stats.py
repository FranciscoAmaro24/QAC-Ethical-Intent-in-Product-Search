"""
QAC ethical-keyword transition statistics by product domain.

For each ethics category and product domain, computes:
    Retained (T->T) : keyword in prefix AND in final search term
    Added    (F->T) : keyword absent from prefix, present in final
    Dropped  (T->F) : keyword in prefix, absent from final

Input
-----
    ethics_classified_5pct.csv  (product_domain + boolean flags)
    ethical_logical_5pct.csv    (prefixes column for word counts)

Output
------
    qac_domain_transitions.csv  (tidy long-form: category x domain x transition)
"""

import os
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "environmental_sustainability":  "Environmental Sustainability",
    "fair_trade_labor_human_rights": "Fair Trade / Labour / Human Rights",
    "animal_welfare":                "Animal Welfare",
    "privacy_data_ethics":           "Privacy & Data Ethics",
    "accessibility_inclusion":       "Accessibility & Inclusion",
}

DOMAIN_ORDER = [
    "Food and Grocery", "Health and Beauty",
    "Electronics and Computers", "Home Garden and DIY",
    "Clothes Shoes and Watches", "Books",
    "Toys Children and Baby", "Sports and Outdoors",
    "Stationery and Office Supplies", "Films TV Music and Games",
    "Car and Motorbike", "Business Industry and Science",
    "None of the above",
]


def word_count(s):
    if pd.isna(s):
        return 0
    return len(str(s).split())


def compute_transitions(df, cat):
    """Retained / Added / Dropped counts for one category in *df*."""
    p  = df[f"{cat}_prefix"].astype(bool)
    f_ = df[f"{cat}_final"].astype(bool)
    pw = df["prefix_wc"]
    qw = df["query_wc"]

    retained_tt = (p & f_).sum()
    same_len_ft = ((~p) & f_ & (pw == qw)).sum()
    retained    = retained_tt + same_len_ft

    added   = ((~p) & f_ & (qw > pw)).sum()
    dropped = (p & (~f_)).sum()

    return pd.Series({"Retained": retained, "Added": added, "Dropped": dropped})


def main():
    cl = pd.read_csv(os.path.join(DIR, "ethics_classified_5pct.csv"))
    lg = pd.read_csv(os.path.join(DIR, "ethical_logical_5pct.csv"),
                     usecols=["query_id", "prefixes"])

    df = cl.merge(lg[["query_id", "prefixes"]], on="query_id", how="left")
    df["product_domain"] = df["product_domain"].fillna("Unclassified")
    df["prefix_wc"] = df["prefixes"].apply(word_count)
    df["query_wc"]  = df["final_search_term"].apply(word_count)

    total = len(df)
    print(f"Total queries: {total:,}")
    print(f"Domains: {df['product_domain'].nunique()}")

    # Overall stats per category
    print("\n--- Overall (all domains) ---")
    for cat, label in CATEGORIES.items():
        t = compute_transitions(df, cat)
        ct = t.sum()
        if ct == 0:
            continue
        print(f"\n  {label}:")
        print(f"    Retained : {t['Retained']:>8,}  ({t['Retained']/ct*100:5.1f}%)")
        print(f"    Added    : {t['Added']:>8,}  ({t['Added']/ct*100:5.1f}%)")
        print(f"    Dropped  : {t['Dropped']:>8,}  ({t['Dropped']/ct*100:5.1f}%)")

    # Per-domain breakdown
    all_domains = [d for d in DOMAIN_ORDER if d in df["product_domain"].values]
    if "Unclassified" in df["product_domain"].values:
        all_domains.append("Unclassified")

    for cat, label in CATEGORIES.items():
        pcol, fcol = f"{cat}_prefix", f"{cat}_final"
        cat_mask = df[pcol].astype(bool) | df[fcol].astype(bool)
        if cat_mask.sum() == 0:
            continue

        print(f"\n--- {label} by domain ---")
        fmt = "  {:<35s} {:>8s} {:>8s} {:>8s} {:>8s}"
        print(fmt.format("Domain", "Ret", "Add", "Drop", "Total"))

        for domain in all_domains:
            sub = df[(df["product_domain"] == domain) & cat_mask]
            if len(sub) == 0:
                continue
            t = compute_transitions(sub, cat)
            dt = t.sum()
            if dt == 0:
                continue
            print(f"  {domain:<35s} {t['Retained']:>8,} {t['Added']:>8,} "
                  f"{t['Dropped']:>8,} {dt:>8,}")

    # Domain summary
    print("\n--- Queries per domain ---")
    for domain in all_domains:
        n_d = (df["product_domain"] == domain).sum()
        print(f"  {domain:<35s} {n_d:>8,}")

    # Save tidy long-form CSV
    rows = []
    for cat, label in CATEGORIES.items():
        pcol, fcol = f"{cat}_prefix", f"{cat}_final"
        cat_mask = df[pcol].astype(bool) | df[fcol].astype(bool)
        for domain in all_domains:
            sub = df[(df["product_domain"] == domain) & cat_mask]
            if len(sub) == 0:
                continue
            t = compute_transitions(sub, cat)
            for transition, count in t.items():
                rows.append({"category": label, "domain": domain,
                             "transition": transition, "count": int(count)})

    out = pd.DataFrame(rows)
    out_path = os.path.join(DIR, "qac_domain_transitions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
