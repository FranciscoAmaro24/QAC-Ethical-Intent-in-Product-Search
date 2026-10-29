"""
Rebuild the 5% corpus on the union condition and count per-term transitions
for the ethical and control lexicons.

Same parquet, seed and sampling as subsample_5pct_filtered.py, but a query is
kept if the vocabulary matches its final query OR its last typed prefix. As in
the original pipeline, the first prefix is discarded, so only queries with at
least two prefixes are analysed.

    python src/reextract_union.py [--parquet data/...] [--out-dir results]

Outputs:
    ethics_union_5pct.csv  one row per kept query (last prefix), with
                           matched_side in {final, prefix, both}
    term_counts_5pct.csv   per term: TT, TF, FT and the FT cases that are
                           prefix truncations (strict / permissive)
    union_meta.json        population counts
"""

import os, re, json, time, argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ethical_analysis import ETHICAL_VOCABULARY
from revision_checks import tail_is_prefix_of
from subsample_5pct_filtered import SEED, sample_batch, vec_match

# Non-ethical product modifiers covering the frequency range of the ethical
# terms; frequency matching happens in union_analysis.py
CONTROL_TERMS = [
    # materials / finishes
    r"\bleather\b", r"\bwooden\b", r"\bcotton\b", r"\bsilicone\b", r"\bmetal\b",
    r"\bglass\b", r"\bceramic\b", r"\bstainless[- ]?steel\b", r"\bvelvet\b",
    r"\bsatin\b", r"\blinen\b", r"\bsuede\b", r"\bdenim\b", r"\bflannel\b",
    r"\bfleece\b", r"\bcorduroy\b", r"\bmesh\b", r"\bchrome\b", r"\bbrass\b",
    r"\bcopper\b", r"\btitanium\b", r"\bporcelain\b", r"\bmatte\b", r"\bglossy\b",
    r"\bfrosted\b", r"\bholographic\b", r"\biridescent\b", r"\btranslucent\b",
    r"\bgalvanized\b", r"\banodized\b", r"\btempered\b", r"\btweed\b",
    # functional modifiers
    r"\bwireless\b", r"\bwaterproof\b", r"\bportable\b", r"\brechargeable\b",
    r"\badjustable\b", r"\belectric\b", r"\bcordless\b", r"\bbluetooth\b",
    r"\bdigital\b", r"\bmagnetic\b", r"\bheavy[- ]?duty\b", r"\bfoldable\b",
    r"\bcollapsible\b", r"\blightweight\b", r"\binsulated\b", r"\bretractable\b",
    r"\bdetachable\b", r"\bdimmable\b", r"\bprogrammable\b", r"\bweighted\b",
    r"\bscented\b", r"\bunscented\b", r"\bpadded\b", r"\bquilted\b",
    r"\breversible\b", r"\bpolarized\b", r"\btelescopic\b", r"\bbreathable\b",
    r"\bstackable\b", r"\bwashable\b", r"\bnon[- ]?slip\b", r"\banti[- ]?fog\b",
    r"\bglow in the dark\b", r"\bnoise cancell?ing\b", r"\bwall[- ]?mounted\b",
    r"\bbattery operated\b", r"\btouch[- ]?screen\b", r"\bquick release\b",
    r"\bclip[- ]?on\b", r"\bdouble[- ]?sided\b", r"\bself[- ]?adhesive\b",
    r"\bembroidered\b", r"\bmonogrammed\b", r"\bengraved\b", r"\bpersonalized\b",
    r"\bknitted\b", r"\bbeaded\b", r"\bsequin\b", r"\bperforated\b",
    r"\blaminated\b", r"\bcushioned\b", r"\boversized\b", r"\bsleeveless\b",
    r"\bhooded\b", r"\bslim[- ]?fit\b", r"\bwide[- ]?leg\b", r"\bhigh[- ]?waisted\b",
    # colours / sizes
    r"\bblack\b", r"\bwhite\b", r"\bpink\b", r"\bblue\b", r"\bred\b",
    r"\bgreen\b", r"\bpurple\b", r"\bbeige\b", r"\bturquoise\b", r"\bmaroon\b",
    r"\bmini\b", r"\blarge\b", r"\bsmall\b", r"\bking size\b", r"\btravel size\b",
]

ALL_TERMS = ([(cat, t) for cat, terms in ETHICAL_VOCABULARY.items() for t in terms]
             + [("control", t) for t in CONTROL_TERMS])
PY_TERMS = [re.compile(t, re.IGNORECASE) for _, t in ALL_TERMS]


def last_prefix(prefixes):
    return [p[-1] if p is not None and len(p) >= 2 else None for p in prefixes]


def count_terms(last, final, acc):
    """Add this batch's per-term TT, TF, FT and truncated-FT counts to `acc`."""
    la = pa.array(last, type=pa.string())
    fa = pa.array(final, type=pa.string())
    match = lambda arr, term: pc.match_substring_regex(arr, term, ignore_case=True) \
        .fill_null(False).to_numpy(zero_copy_only=False)
    for j, (_, term) in enumerate(ALL_TERMS):
        pm, fm = match(la, term), match(fa, term)
        acc[j, 0] += np.sum(pm & fm)
        acc[j, 1] += np.sum(pm & ~fm)
        ft_idx = np.flatnonzero(~pm & fm)
        acc[j, 2] += len(ft_idx)
        for i in ft_idx:
            m = PY_TERMS[j].search(final[i])
            if m is not None:
                acc[j, 3] += tail_is_prefix_of(last[i], m.group(0), 1, 3)
                acc[j, 4] += tail_is_prefix_of(last[i], m.group(0))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(here, os.pardir, "data",
                                                      "amazon_qac_sample_10pct.parquet"))
    ap.add_argument("--out-dir", default=os.path.join(here, os.pardir, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "ethics_union_5pct.csv")

    pf = pq.ParquetFile(args.parquet)
    n_groups = pf.metadata.num_row_groups
    rng = np.random.default_rng(SEED)
    acc = np.zeros((len(ALL_TERMS), 5), dtype=np.int64)
    meta = {"rows_10pct": pf.metadata.num_rows, "sampled": 0, "with_2plus_prefixes": 0,
            "kept": 0, "kept_final": 0, "kept_prefix_only": 0, "kept_prefix": 0}
    t0 = time.time()

    for g in range(n_groups):
        df = sample_batch(pf.read_row_group(g).to_pandas(), rng)
        meta["sampled"] += len(df)
        df["prefixes"] = last_prefix(df["prefixes"])
        df = df[df["prefixes"].notna()].reset_index(drop=True)
        meta["with_2plus_prefixes"] += len(df)

        last = df["prefixes"].tolist()
        count_terms(last, df["final_search_term"].fillna("").tolist(), acc)

        fm, pm = vec_match(df["final_search_term"]), vec_match(df["prefixes"])
        either = fm | pm
        keep = df[either].copy()
        keep["matched_side"] = np.select([fm[either] & pm[either], fm[either]],
                                         ["both", "final"], "prefix")
        meta["kept"] += len(keep)
        meta["kept_final"] += int(fm.sum())
        meta["kept_prefix"] += int(pm.sum())
        meta["kept_prefix_only"] += int((pm & ~fm).sum())

        if len(keep):
            first = meta["kept"] == len(keep)
            keep.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        print(f"row group {g + 1:>3}/{n_groups}: kept {meta['kept']:,} "
              f"({meta['kept_prefix_only']:,} prefix-only), {time.time() - t0:.0f}s", flush=True)

    tc = pd.DataFrame(acc, columns=["TT", "TF", "FT", "FT_trunc_strict", "FT_trunc_permissive"])
    tc.insert(0, "term", [t for _, t in ALL_TERMS])
    tc.insert(0, "lexicon", [c for c, _ in ALL_TERMS])
    tc.to_csv(os.path.join(args.out_dir, "term_counts_5pct.csv"), index=False)
    with open(os.path.join(args.out_dir, "union_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
