"""
Paper figures, sized for a single acmart column (3.33 in), as PDF.

    python src/make_figures.py [--data-dir data] [--results-dir results] [--out-dir figures]

Figure 1 and the per-token figure read the tables written by
revision_checks.py; Figure 2 needs the query-level table and domain labels.
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from ethical_analysis import CATEGORY_LABELS

COL_W = 3.33  # acmart single column, inches

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Category key -> (axis label, legend label, colour)
CATS = {
    "environmental_sustainability":  ("Environmental Sustainability", "Environmental", "#1f77b4"),
    "fair_trade_labor_human_rights": ("Fair Trade / Labour", "Fair Trade", "#ff7f0e"),
    "animal_welfare":                ("Animal Welfare", "Animal Welfare", "#2ca02c"),
    "privacy_data_ethics":           ("Privacy & Data Ethics", "Privacy", "#d62728"),
    "accessibility_inclusion":       ("Accessibility & Inclusion", "Accessibility", "#9467bd"),
}

DOMAIN_LABELS = {
    "Food and Grocery": "Food & Grocery",
    "Health and Beauty": "Health & Beauty",
    "Electronics and Computers": "Electronics & Computers",
    "Home Garden and DIY": "Home, Garden & DIY",
    "Clothes Shoes and Watches": "Clothes, Shoes & Watches",
    "Books": "Books",
    "Toys Children and Baby": "Toys, Children & Baby",
    "Sports and Outdoors": "Sports & Outdoors",
    "Stationery and Office Supplies": "Stationery & Office",
    "Films TV Music and Games": "Films, TV, Music & Games",
    "Car and Motorbike": "Car & Motorbike",
    "Business Industry and Science": "Business, Industry & Science",
    "None of the above": "None of the above",
}
BASELINE = "All ethical queries"
RDA_COLORS = {"Retained": "#1f77b4", "Dropped": "#ff7f0e", "Added": "#2ca02c"}


def strip_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure_length_reversal(by_length, path):
    """Addition rate by words typed (solid) and by accepted-query length (dotted)."""
    bins = list(by_length.columns)
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))

    for cat, (_, short, color) in CATS.items():
        label = CATEGORY_LABELS[cat]
        ax.plot(x, by_length.loc[("query_wc", label)] * 100, color=color,
                lw=0.8, ls=":", alpha=0.55, zorder=1)
        ax.plot(x, by_length.loc[("prefix_wc", label)] * 100, color=color,
                lw=1.3, marker="o", ms=2.6, label=short, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Words typed by the user")
    ax.set_ylabel("Addition rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    strip_spines(ax)
    ax.legend(loc="upper right", ncol=2, frameon=False,
              handlelength=1.4, columnspacing=1.0, borderpad=0.2)
    ax.annotate("dotted: same quantity by\naccepted-query length",
                xy=(2.4, 60), fontsize=5.5, color="0.45", linespacing=1.1)
    fig.savefig(path)
    plt.close(fig)


def figure_domain_heatmap(lg, path):
    """Share of each category's queries per domain, with the all-queries baseline below."""
    sub = lg[lg["product_domain"].isin(DOMAIN_LABELS)].copy()
    sub["dom"] = sub["product_domain"].map(DOMAIN_LABELS)

    rows = {}
    for cat, (label, _, _) in CATS.items():
        counts = sub.loc[sub[f"{cat}_final"].astype(bool), "dom"].value_counts()
        rows[label] = counts / counts.sum() * 100
    base = sub["dom"].value_counts()
    rows[BASELINE] = base / base.sum() * 100
    M = pd.DataFrame(rows).T.fillna(0)
    M = M[rows[BASELINE].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    im = ax.imshow(M.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(M.index)
    ax.get_yticklabels()[-1].set_fontweight("bold")

    for (i, j), v in np.ndenumerate(M.values):
        if v >= 1:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 55 else "0.2")

    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", length=0)
    ax.axhline(M.shape[0] - 1.5, color="0.2", lw=0.9)  # separates the baseline row

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% of row", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.savefig(path)
    plt.close(fig)


def figure_token_transitions(tokens, path, top=5, min_n=30):
    """Retained / Dropped / Added shares for the most frequent tokens per category."""
    fig, axes = plt.subplots(len(CATS), 1, figsize=(COL_W, 3.6))
    for ax, (cat, (_, short, _)) in zip(axes, CATS.items()):
        tdf = tokens[(tokens["category"] == CATEGORY_LABELS[cat]) & (tokens["n"] >= min_n)]
        tdf = tdf.sort_values("n").tail(top)
        y = np.arange(len(tdf))
        left = np.zeros(len(tdf))
        for kind, color in RDA_COLORS.items():
            pct = tdf[kind].values / tdf["n"].values * 100
            ax.barh(y, pct, left=left, height=0.65, color=color, label=kind)
            left += pct

        ax.set_yticks(y)
        ax.set_yticklabels([f"{t} ({n:,})" for t, n in zip(tdf["token"], tdf["n"])], fontsize=5.5)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xlim(0, 100)
        ax.set_title(short, loc="left", fontsize=6.5, pad=2)
        strip_spines(ax)
        if ax is axes[-1]:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xlabel("Share of token occurrences")
        else:
            ax.set_xticklabels([])

    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.96))  # leave a strip for the legend
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.2, columnspacing=1.0)
    fig.savefig(path)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(here, os.pardir, "data"))
    ap.add_argument("--results-dir", default=os.path.join(here, os.pardir, "results"))
    ap.add_argument("--out-dir", default=os.path.join(here, os.pardir, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    D = lambda name: os.path.join(args.data_dir, name)
    R = lambda name: os.path.join(args.results_dir, name)
    out = lambda name: os.path.join(args.out_dir, name)

    by_length = pd.read_csv(R("step1_addition_by_length.csv"), index_col=["basis", "category"])
    figure_length_reversal(by_length, out("figure_length_reversal.pdf"))

    domains = pd.read_csv(D("ethics_classified_5pct.csv"), usecols=["query_id", "product_domain"])
    lg = pd.read_csv(D("ethical_logical_5pct.csv")).merge(domains, on="query_id", how="left")
    figure_domain_heatmap(lg, out("figure_domain_heatmap.pdf"))

    figure_token_transitions(pd.read_csv(R("token_transitions.csv")),
                             out("figure_token_transitions.pdf"))
    print(f"Wrote 3 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
