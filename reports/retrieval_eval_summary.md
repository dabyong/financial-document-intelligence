# Baseline RAG Retrieval Evaluation

- Questions: 12
- Candidate chunks judged: 60
- Collection: `sec_10k_minilm_v1`
- Index fingerprint: `4065a39094d2d8530630435c622d4795760e9596a6553a2b1fb4dd7b0ce464f0`
- Relevant threshold: grade >= 1 (human labels)
- Grades: 0 = not useful; 1 = partial/contextual; 2 = directly useful

## Aggregate metrics (macro average)

- hit_at_5: 1.000
- precision_at_3: 0.944
- mrr_at_5: 0.958
- ndcg_at_5: 0.944
- direct_evidence_at_5: 1.000

## Limitations

- The evaluation labels only the top five results retrieved by this baseline index.
- This is a pooled judgment of retrieved candidates, not an exhaustive gold standard across all 3,284 chunks.
- The number of questions is small and should be expanded for production claims.
- This measures retrieval relevance, NOT the accuracy, citation correctness, or safety of LLM-generated answers.
- Compare retrieval versions only after judging a shared candidate pool or assigning stable gold evidence passages.

## Per-question data

See `reports\retrieval_eval_metrics.csv`.
