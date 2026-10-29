"""
Analyse the union corpus from reextract_union.py: Table 2, the drop audit,
truncation-corrected addition, retention / addition by prefix length, and the
ethical vs frequency-matched control comparison.

    python src/union_analysis.py [--results-dir results] [--data-dir data]
"""

import os, re, json, argparse
import numpy as np
import pandas as pd

from ethical_analysis import PATTERNS, CATEGORY_LABELS
from revision_checks import LENGTH_BINS, matched_terms, tail_is_prefix_of, word_count

MIN_ACCEPTED = 30  # control comparison: skip terms with fewer accepted occurrences


def div(a, b):
    return a / b if b else np.nan


def flag_categories(df):
    prefix = df["prefixes"].fillna("").str.lower()
    final = df["final_search_term"].fillna("").str.lower()
    for cat, pat in PATTERNS.items():
        df[f"{cat}_prefix"] = prefix.str.contains(pat.pattern)
        df[f"{cat}_final"] = final.str.contains(pat.pattern)
    return df


def counts(df, cat, mask=True):
    p, f = df[f"{cat}_prefix"], df[f"{cat}_final"]
    return int((mask & p & f).sum()), int((mask & p & ~f).sum()), int((mask & ~p & f).sum())


def drop_kind(prefix, final, cat):
    """What happened to a typed term that is missing from the final query."""
    final = str(final).lower()
    for term in matched_terms(prefix, cat):
        t = term.lower()
        if re.search(r"\b" + re.escape(t) + r"\w", final):  # "natural" -> "naturalizer"
            return "extended"
        if re.sub(r"[- ]", "", t) in re.sub(r"[- ]", "", final):  # "hand made" -> "handmade"
            return "respelled"
    return "absent"


def control_comparison(tc):
    """Pool ethical terms, their nearest-frequency control terms, and all controls."""
    tc["accepted"] = tc["TT"] + tc["FT"]
    eth = tc[(tc["lexicon"] != "control") & (tc["accepted"] >= MIN_ACCEPTED)].copy()
    ctl = tc[(tc["lexicon"] == "control") & (tc["accepted"] >= MIN_ACCEPTED)]
    # nearest neighbour on log accepted frequency, with replacement
    log_ctl = np.log(ctl["accepted"].values)
    eth["match"] = [ctl["term"].values[np.argmin(np.abs(log_ctl - np.log(a)))]
                    for a in eth["accepted"]]
    matched = ctl.set_index("term").loc[eth["match"]].reset_index()

    def pooled(d, label):
        tt, tf, ft = d["TT"].sum(), d["TF"].sum(), d["FT"].sum()
        res = {"set": label, "terms": d["term"].nunique(), "accepted": tt + ft,
               "addition": div(ft, tt + ft), "retention": div(tt, tt + tf)}
        for kind in ("strict", "permissive"):
            k = d[f"FT_trunc_{kind}"].sum()
            res[f"addition_excl_{kind}"] = div(ft - k, tt + ft - k)
        res["median_term_addition"] = (d["FT"] / d["accepted"]).median()
        return res

    summary = pd.DataFrame([pooled(eth, "ethical"), pooled(matched, "control (matched)"),
                            pooled(ctl, "control (all)")]).set_index("set")
    pairs = eth[["lexicon", "term", "accepted", "FT", "match"]].copy()
    pairs["eth_addition"] = eth["FT"] / eth["accepted"]
    pairs["ctl_accepted"] = matched["accepted"].values
    pairs["ctl_addition"] = (matched["FT"] / matched["accepted"]).values
    return summary, pairs, matched


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(here, os.pardir, "results"))
    ap.add_argument("--data-dir", default=os.path.join(here, os.pardir, "data"))
    args = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    R = lambda name: os.path.join(args.results_dir, name)

    meta = json.load(open(R("union_meta.json")))
    un = pd.read_csv(R("ethics_union_5pct.csv"))
    old = pd.read_csv(os.path.join(args.data_dir, "ethical_logical_5pct.csv"), usecols=["query_id"])

    for k, v in meta.items():
        print(f"{k:<22s} {v:>12,}")
    print(f"final-query corpus queries missing from the union corpus: "
          f"{(~old['query_id'].isin(un['query_id'])).sum():,} of {len(old):,}")

    un = flag_categories(un)
    un["prefix_wc"] = un["prefixes"].apply(word_count)

    print("\n== Table 2 (retention and dropped over typed, addition over accepted)")
    rows = []
    for cat, label in CATEGORY_LABELS.items():
        tt, tf, ft = counts(un, cat)
        rows.append({"category": label, "TT": tt, "TF": tf, "FT": ft,
                     "typed": tt + tf, "accepted": tt + ft, "retention": div(tt, tt + tf),
                     "dropped": div(tf, tt + tf), "addition": div(ft, tt + ft)})
    t2 = pd.DataFrame(rows).set_index("category")
    print(t2.round(3).to_string())
    t2.to_csv(R("table2_union.csv"))

    print("\n== Drops by kind")
    drops = pd.DataFrame([
        {"category": label, "prefix": pr, "final": fi, "kind": drop_kind(pr, fi, cat)}
        for cat, label in CATEGORY_LABELS.items()
        for pr, fi in un.loc[un[f"{cat}_prefix"] & ~un[f"{cat}_final"],
                             ["prefixes", "final_search_term"]].itertuples(index=False)])
    print(drops.groupby(["category", "kind"]).size().unstack(fill_value=0).to_string())
    drops.to_csv(R("step3_drops.csv"), index=False)

    print("\n== Table 2 adjusted: extended/respelled drops count as retained, truncations excluded")
    rows = []
    for cat, label in CATEGORY_LABELS.items():
        tt, tf, ft = counts(un, cat)
        respelled = int((drops.loc[drops["category"] == label, "kind"] != "absent").sum())
        row = {"category": label, "typed": tt + tf, "dropped_raw": div(tf, tt + tf),
               "dropped_adj": div(tf - respelled, tt + tf)}
        added = un.loc[~un[f"{cat}_prefix"] & un[f"{cat}_final"], ["prefixes", "final_search_term"]]
        for kind, kw in (("strict", dict(max_words=1, min_chars=3)), ("permissive", {})):
            k = sum(any(tail_is_prefix_of(pr, t, **kw) for t in matched_terms(fi, cat))
                    for pr, fi in added.itertuples(index=False))
            row[f"trunc_{kind}"] = k
            row[f"addition_excl_{kind}"] = div(ft - k, tt + respelled + ft - k)
        row["addition_raw"] = div(ft, tt + ft)
        rows.append(row)
    adj = pd.DataFrame(rows).set_index("category")
    print(adj.round(3).to_string())
    adj.to_csv(R("table2_union_adjusted.csv"))

    print("\n== Retention and addition by prefix word count")
    rows = []
    for cat, label in CATEGORY_LABELS.items():
        for lo, hi, lbl in LENGTH_BINS:
            tt, tf, ft = counts(un, cat, un["prefix_wc"].between(lo, hi))
            rows.append({"category": label, "bin": lbl,
                         "retention": div(tt, tt + tf) if tt + tf >= 10 else np.nan,
                         "addition": div(ft, tt + ft) if tt + ft >= 10 else np.nan,
                         "n_typed": tt + tf, "n_accepted": tt + ft})
    by_len = pd.DataFrame(rows)
    print(by_len.pivot(index="category", columns="bin", values="addition").round(3).to_string())
    by_len.to_csv(R("step1_union_by_prefix_length.csv"), index=False)

    print("\n== Ethical vs frequency-matched control terms")
    summary, pairs, matched = control_comparison(pd.read_csv(R("term_counts_5pct.csv")))
    print(summary.round(3).to_string())
    summary.to_csv(R("step5_control_comparison.csv"))
    pairs.to_csv(R("step5_matched_pairs.csv"), index=False)
    for cat, label in CATEGORY_LABELS.items():
        e, c = pairs[pairs["lexicon"] == cat], matched[pairs["lexicon"].values == cat]
        if len(e):
            print(f"{label:<36s} ethical {div(e['FT'].sum(), e['accepted'].sum()):.3f}  "
                  f"control {div(c['FT'].sum(), c['accepted'].sum()):.3f}  ({len(e)} terms)")


if __name__ == "__main__":
    main()
