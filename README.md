# Ethical Intent in Online Product Search: Evidence from Query Auto-Completion

Code, result tables and figures for the paper. We take a 5% sample of the
[AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC) dataset (19,776,441 queries) and
match 89 ethical-intent terms in five categories against the last prefix the user typed and the
query they accepted. For each category we measure how often typed terms survive QAC (retention)
and how often accepted terms were never typed (addition). We check these against prefix
truncation, a frequency-matched control lexicon, and the product domain assigned by an LLM.

## Definitions

For each category, a query is **TT** (term in last prefix and in final query), **TF** (prefix
only) or **FT** (final query only).

- **Retention** = TT / (TT + TF): the share of typed ethical terms that survive.
- **Addition** = FT / (TT + FT): the share of ethical terms in accepted queries that were not typed.

## Setup

```bash
pip install -r requirements.txt
```

You need Python ≥ 3.10. Step 4 needs Apple Silicon and MLX (about 8 h on a 16 GB M-series
machine). Every other step runs on any platform. All data goes in `data/`, which is not
included because the files are large; step 1 downloads it.

## Pipeline

Run the steps in order from the repository root.

| Step | Command | Output |
|---|---|---|
| 1 | `python src/download_amazon_qac.py` | `data/amazon_qac_sample_10pct.parquet` (39,552,882 rows) |
| 2 | `python src/subsample_5pct_filtered.py` | `data/amazon_qac_5pct_ethics_filtered.csv` |
| 3 | `python src/ethical_analysis.py` | `data/ethical_data_final_5pct.csv` (token level), `data/ethical_logical_5pct.csv` (query level) |
| 4 | `python src/send_query_llm.py` | `data/ethics_classified_5pct.csv` (Qwen 2.5-3B product domain) |
| 5 | `python src/revision_checks.py` | Length, truncation, domain-count and per-token tables on the final-query corpus |
| 6 | `python src/reextract_union.py` | `results/ethics_union_5pct.csv`, `results/term_counts_5pct.csv`, `results/union_meta.json` (~6 min) |
| 7 | `python src/union_analysis.py` | Table 2, drop audit, truncation, length and control-lexicon tables on the union corpus |
| 8 | `python src/domain_chi2.py` | `results/step4_domain_chi2.csv` |
| 9 | `python src/make_figures.py` | `figures/*.pdf` |

Steps 2 and 6 use the same seed (1234) and the same per-row-group sampling. Step 6 keeps a
query if the final query **or** the last typed prefix matches the vocabulary. This is the corpus
behind Table 2.

## Ethical vocabulary

There are 89 regex terms across five categories: Environmental Sustainability (20), Fair Trade &
Labour (19), Animal Welfare (15), Privacy & Data Ethics (18) and Accessibility & Inclusion (17).
The full list is in `src/ethical_analysis.py`. The 104 control terms are in `src/reextract_union.py`.
