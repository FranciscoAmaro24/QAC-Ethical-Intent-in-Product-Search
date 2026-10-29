"""
Chi-square test of independence between product domain and each ethical
category (term present in the final query, yes / no), with Cramér's V.
Unclassified queries are excluded, leaving 13 domains (df = 12).

    python src/domain_chi2.py [--data-dir data] [--out-dir results]
"""

import os, argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ethical_analysis import CATEGORY_LABELS


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(here, os.pardir, "data"))
    ap.add_argument("--out-dir", default=os.path.join(here, os.pardir, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lg = pd.read_csv(os.path.join(args.data_dir, "ethical_logical_5pct.csv"))
    cl = pd.read_csv(os.path.join(args.data_dir, "ethics_classified_5pct.csv"),
                     usecols=["query_id", "product_domain"])
    lg = lg.merge(cl, on="query_id", how="inner").dropna(subset=["product_domain"])

    rows = []
    for cat, label in CATEGORY_LABELS.items():
        ct = pd.crosstab(lg["product_domain"], lg[f"{cat}_final"].astype(bool))
        chi2, p, dof, expected = chi2_contingency(ct)
        n = ct.values.sum()
        rows.append({"category": label, "n": n, "dof": dof, "chi2": chi2, "p": p,
                     "cramers_v": np.sqrt(chi2 / (n * (min(ct.shape) - 1))),
                     "cells_expected_lt5": int((expected < 5).sum())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(args.out_dir, "step4_domain_chi2.csv"), index=False)


if __name__ == "__main__":
    main()
