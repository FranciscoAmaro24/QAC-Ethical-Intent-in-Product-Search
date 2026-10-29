"""
Checks on the final-query corpus (queries whose accepted query matches the
vocabulary): Table 2 on that corpus, addition by length, prefix truncation,
domain counts and per-token Retained / Dropped / Added counts.

    python src/revision_checks.py [--data-dir data] [--out-dir results]
"""

import os, re, argparse
import numpy as np
import pandas as pd

from ethical_analysis import ETHICAL_VOCABULARY, CATEGORY_LABELS

LENGTH_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 99, "6+")]
CAT_PATTERNS = {cat: [re.compile(t, re.IGNORECASE) for t in terms]
                for cat, terms in ETHICAL_VOCABULARY.items()}


def word_count(s):
    return 0 if pd.isna(s) else len(str(s).split())


def tail_is_prefix_of(prefix, term, max_words=4, min_chars=2):
    """True if the user had begun typing `term` at the end of `prefix`."""
    words = str(prefix).lower().strip().split()
    t = str(term).lower().strip()
    for k in range(1, min(max_words, len(words)) + 1):
        tail = " ".join(words[-k:])
        if len(tail) >= min_chars and t.startswith(tail) and tail != t:
            return True
    return False


def matched_terms(text, cat):
    """Every vocabulary term of `cat` found in `text`, as written."""
    text = str(text)
    return [m.group(0) for p in CAT_PATTERNS[cat] for m in p.finditer(text)]


def flags(df, cat):
    return df[f"{cat}_prefix"].astype(bool), df[f"{cat}_final"].astype(bool)


def addition_by_length(lg):
    """Addition rate and cell size per length bin, by final-query and by prefix word count."""
    out = []
    for cat, label in CATEGORY_LABELS.items():
        p, f = flags(lg, cat)
        for basis in ("query_wc", "prefix_wc"):
            for lo, hi, lbl in LENGTH_BINS:
                m = lg[basis].between(lo, hi)
                tt, ft = (m & p & f).sum(), (m & ~p & f).sum()
                out.append({"category": label, "basis": basis, "bin": lbl, "n": tt + ft,
                            "addition": ft / (tt + ft) if tt + ft >= 10 else np.nan})
    df = pd.DataFrame(out)
    pivot = lambda v: df.pivot_table(index=["basis", "category"], columns="bin",
                                     values=v, sort=False)
    return pivot("addition"), pivot("n")


def truncation(lg):
    """Flag each added (FT) case where the prefix ends in a partly typed term."""
    rows = []
    for cat, label in CATEGORY_LABELS.items():
        p, f = flags(lg, cat)
        for prefix, final in lg.loc[~p & f, ["prefixes", "final_search_term"]].itertuples(index=False):
            terms = matched_terms(final, cat)
            rows.append({"category": label, "prefix": prefix, "final": final,
                         "terms": "|".join(terms),
                         "permissive": any(tail_is_prefix_of(prefix, t) for t in terms),
                         "strict": any(tail_is_prefix_of(prefix, t, 1, 3) for t in terms)})
    tr = pd.DataFrame(rows)

    summary = []
    for cat, label in CATEGORY_LABELS.items():
        p, f = flags(lg, cat)
        tt, ft = int((p & f).sum()), int((~p & f).sum())
        sub = tr[tr["category"] == label]
        row = {"category": label, "TT": tt, "FT": ft, "addition_raw": ft / (tt + ft)}
        for kind in ("strict", "permissive"):
            k = int(sub[kind].sum())
            row[f"trunc_{kind}"] = k
            row[f"trunc_{kind}_share"] = k / ft if ft else np.nan
            # truncations removed from FT and from TT+FT
            row[f"addition_excl_{kind}"] = (ft - k) / (tt + ft - k)
            # truncations counted as typed
            row[f"addition_as_typed_{kind}"] = (ft - k) / (tt + ft)
        summary.append(row)
    return tr, pd.DataFrame(summary).set_index("category")


def token_transitions(tok):
    """Per-token Retained / Dropped / Added counts on the token-level table."""
    out = []
    for cat, label in CATEGORY_LABELS.items():
        pcol, fcol = f"{cat}_prefix", f"{cat}_final"
        for token in pd.concat([tok[pcol].dropna(), tok[fcol].dropna()]).unique():
            has_p, has_f = tok[pcol] == token, tok[fcol] == token
            out.append({"category": label, "token": token,
                        "Retained": (has_p & has_f).sum(),
                        "Dropped": (has_p & (tok[fcol] != token)).sum(),  # NaN != token
                        "Added": (has_f & (tok[pcol] != token)).sum()})
    df = pd.DataFrame(out)
    df["n"] = df[["Retained", "Dropped", "Added"]].sum(axis=1)
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(here, os.pardir, "data"))
    ap.add_argument("--out-dir", default=os.path.join(here, os.pardir, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    D = lambda name: os.path.join(args.data_dir, name)
    R = lambda name: os.path.join(args.out_dir, name)

    tok = pd.read_csv(D("ethical_data_final_5pct.csv"))
    lg = pd.read_csv(D("ethical_logical_5pct.csv"))
    cl = pd.read_csv(D("ethics_classified_5pct.csv"), usecols=["query_id", "product_domain"])
    lg["prefix_wc"] = lg["prefixes"].apply(word_count)
    lg["query_wc"] = lg["final_search_term"].apply(word_count)

    print(f"token-level rows {len(tok):,}, query-level rows {len(lg):,}")
    typed_in_full = (lg["prefixes"].str.strip() == lg["final_search_term"].str.strip()).sum()
    print(f"queries whose last prefix equals the final query: {typed_in_full:,}")

    print("\n== Table 2 on the final-query corpus (TF = 0 by construction)")
    t2 = []
    for cat, label in CATEGORY_LABELS.items():
        p, f = flags(lg, cat)
        tt, tf, ft = (p & f).sum(), (p & ~f).sum(), (~p & f).sum()
        t2.append({"category": label, "TT": tt, "TF": tf, "FT": ft, "typed (TT+TF)": tt + tf,
                   "retention": tt / (tt + tf) if tt + tf else np.nan,
                   "dropped": tf / (tt + tf) if tt + tf else np.nan,
                   "addition": ft / (tt + ft) if tt + ft else np.nan})
    t2 = pd.DataFrame(t2).set_index("category")
    print(t2.round(3).to_string())
    t2.to_csv(R("table2_existing_corpus.csv"))

    print("\n== Addition rate by length (final-query vs prefix word count)")
    add, n = addition_by_length(lg)
    print((add * 100).round(1).to_string())
    add.to_csv(R("step1_addition_by_length.csv"))
    n.to_csv(R("step1_n_by_length.csv"))

    print("\n== Prefix truncation among added (FT) cases")
    tr, s2 = truncation(lg)
    print(s2.round(3).to_string())
    s2.to_csv(R("step2_truncation_summary.csv"))
    tr.to_csv(R("step2_truncation_rows.csv"), index=False)

    for kind in ("strict", "permissive"):
        lg_k = lg.copy()
        for cat, label in CATEGORY_LABELS.items():
            p, f = flags(lg_k, cat)
            truncated = tr.loc[tr["category"] == label, kind].values
            lg_k.loc[lg_k.index[~p & f][truncated], f"{cat}_final"] = False
        a, _ = addition_by_length(lg_k)
        a.to_csv(R(f"step1_addition_by_length_excl_{kind}.csv"))

    print("\n== Queries per product domain")
    domains = lg[["query_id"]].merge(cl, on="query_id", how="left")["product_domain"]
    print(domains.fillna("Unclassified").value_counts().to_string())

    tt = token_transitions(tok)
    tt.to_csv(R("token_transitions.csv"), index=False)
    print(f"\nSaved per-token counts for {len(tt)} tokens")


if __name__ == "__main__":
    main()
