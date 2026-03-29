# Ethical Intent in Online Product Search: Evidence from Query Auto-Completion

Query auto-completion (QAC) shapes what consumers search for, yet its interaction with ethically motivated queries such as organic, cruelty-free, fair-trade have not been studied. We analyse a 5% sample of the [AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC) dataset (~135k ethical queries) and classify each keyword occurrence as **Retained**, **Dropped**, or **Added** by the auto-complete system across five ethical-intent categories. An LLM then maps every query to one of 13 product domains for cross-domain analysis.

## Pipeline

Run steps 1–5 in order, then open the notebook to reproduce all tables and figures.

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/download_amazon_qac.py` | 10% sample parquet from AmazonQAC (~395M → ~39.5M rows) |
| 2 | `src/subsample_5pct_filtered.py` | 5% ethics-filtered CSV (~1.6M rows) |
| 3 | `src/ethical_analysis.py` | Per-token and per-query keyword match CSVs |
| 4 | `src/send_query_llm.py` | Product-domain classification via Qwen 2.5-3B (MLX) |
| 5 | `src/qac_domain_stats.py` | Domain × category transition statistics |
| — | `ethics_results.ipynb` | All paper tables and figures |

Step 4 requires Apple Silicon + MLX (~8 h on M-series, 16 GB). All other steps run on any platform. See each script's docstring for CLI flags.

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. Data files are downloaded in Step 1 and excluded from the repo via `.gitignore`.

## Ethical Vocabulary

89 regex terms across 5 categories: Environmental Sustainability (20), Fair Trade & Labour (19), Animal Welfare (15), Privacy & Data Ethics (18), Accessibility & Inclusion (17). Full list in `src/ethical_analysis.py`.
