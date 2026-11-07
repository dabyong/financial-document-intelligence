# Project 5 — Final AI Answer Evaluation

**Review status:** User-reviewed and approved scores recorded in `reports/answer_eval_review.csv`.
**Model:** `gemini-3.5-flash-lite`  
**Retrieval:** `multi`; top-k = 10; sources: SEC 10-K text + XBRL-backed DuckDB metrics.  
**Scope:** 10 fixed test questions, including one insufficient-evidence question.  
**Evaluation:** Four human-reviewed dimensions, each scored 0–2; equal weight.

## Summary

- **Answer quality:** **79/80 (98.75%)**.
- **Latency:** mean **12.65s**, median **15.26s**, min **2.82s**, max **21.05s**.
- **Latency definition:** script elapsed time for retrieval, evidence assembly, and model response; not isolated LLM inference time.
- **Original citation ID checks:** 9/10 reported PASSED.
- **Grouped-citation re-audit:** 10/10 passed explicit ID checks; original flags remain unchanged.
- **Important:** ID validation is not proof that every claim is supported. Citation support was separately human-reviewed.

## Dimension results

| Dimension | Score |
|---|---:|
| Numerical accuracy | 20/20 (100.00%) |
| Citation support | 19/20 (95.00%) |
| Completeness | 20/20 (100.00%) |
| Uncertainty handling | 20/20 (100.00%) |

## Case-level results

| Case | Category | Numerical | Citations | Completeness | Uncertainty | Total | Elapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| A01 | Causal financial analysis | 2 | 2 | 2 | 2 | 8/8 | 8.39s |
| A02 | Financial comparison | 2 | 1 | 2 | 2 | 7/8 | 3.39s |
| A03 | Year-over-year growth | 2 | 2 | 2 | 2 | 8/8 | 6.88s |
| A04 | Non-GAAP calculation | 2 | 2 | 2 | 2 | 8/8 | 2.82s |
| A05 | Profitability comparison | 2 | 2 | 2 | 2 | 8/8 | 17.01s |
| A06 | Risk disclosure | 2 | 2 | 2 | 2 | 8/8 | 18.84s |
| A07 | Business drivers | 2 | 2 | 2 | 2 | 8/8 | 17.28s |
| A08 | Regulatory risk | 2 | 2 | 2 | 2 | 8/8 | 17.33s |
| A09 | Financial position | 2 | 2 | 2 | 2 | 8/8 | 21.05s |
| A10 | Evidence sufficiency | 2 | 2 | 2 | 2 | 8/8 | 13.50s |

## Quality findings and known limitations

1. **A02 CAGR citation precision:** For calculated growth rates, cite both baseline and ending-year records.
2. **A04 free-cash-flow definitions:** Meta-reported FY2025 non-GAAP FCF is $43.585B, while project-derived operating cash flow minus capex is $46.109B. The $2.524B difference is finance-lease principal payments. Name the definition being used.
3. **A10 abstention:** The exact commercial success rate for Meta AI research projects is not established by the supplied SEC evidence; the system should not invent a percentage.
4. **Citation parser:** Earlier code could flag legitimate grouped citations such as `[D1, D3, D5]` as missing. Keep the original test record for auditability; use the corrected parser in the application and regression tests.
5. **Small, curated sample:** Ten reviewed questions do not estimate production-wide factual accuracy; broaden the holdout set, especially for ambiguous questions and conflicting metrics.
6. **Operating constraints:** Real-world latency, daily API quotas, burst limits, costs, and service availability can change. Add request throttling, timeouts and clear errors.

## Reproducibility

- Review CSV SHA-256: `914f55e40dbc3d412fe9e29b3beee144ed4e50fe7b8eb2c00b918bb35e0d382f`
- Results JSONL SHA-256: `a66ab050bcea9b5877e1541077245e1682676280b55573754dc24192517e5d4b`
- Question-set CSV SHA-256: `ea27b756341b9a3c9d3c7da4c125531bc3ca180e7778ffb3977d86606e4c23e7`

## Limited model comparison — shared cases only

- Compared the same **4** cases: A01, A02, A03, A04.
- Gemini 3.8 Flash approved score: **31/32 (96.88%)**.
- Gemini 3.5 Flash-Lite approved score: **31/32 (96.88%)**.
- Observed mean elapsed time on shared cases: **405.19s** (3.8 Flash) vs **5.37s** (3.5 Flash-Lite).
- **Fairness caveat:** These timings were measured under different API conditions. Do not equate them with a controlled provider benchmark, nor compare 4-case and 10-case scores as if the sample sizes matched.

See `final_model_comparison_shared_cases.csv` for individual case comparisons.

## Deployment decision

**Recommended candidate:** `gemini-3.5-flash-lite` for the public Streamlit demo, subject to load testing, current quotas and continued evidence-grounding tests. It passed the reviewed question set with practical observed latency. Use validated SQL for numbers and SEC passages for explanations, and clearly show citations, definitions, and insufficient-evidence responses.

*This report is descriptive model evaluation, not investment advice.*
