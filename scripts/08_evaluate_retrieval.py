"""Human-grounded retrieval evaluation for the SEC 10-K Chroma index.

From the project root:
  python scripts/08_evaluate_retrieval.py prepare
  # Label reports/retrieval_eval_review.csv (relevance_grade: 0, 1, or 2)
  python scripts/08_evaluate_retrieval.py score

Labels:
  2 = direct, specific evidence that answers a substantial part of the question
  1 = useful partial/contextual evidence, but not a direct answer
  0 = unrelated, misleading, or not useful to answer the question

This evaluation is restricted to a fixed 12-question / top-5 baseline candidate
pool. Scores are judged retrieval quality, NOT LLM answer correctness.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "sec_10k_minilm_v1"
REPORTS_DIR = ROOT / "reports"
REVIEW_PATH = REPORTS_DIR / "retrieval_eval_review.csv"
METRICS_PATH = REPORTS_DIR / "retrieval_eval_metrics.csv"
SUMMARY_PATH = REPORTS_DIR / "retrieval_eval_summary.md"
TOP_K = 5

# Keep this question set fixed across baseline and subsequent comparisons.
# Questions are intended for evaluation; they are NOT labeled relevant by fiat.
QUESTIONS = [
    ("Q01", "alphabet", 2025, "What factors drove Google Search advertising revenue growth in 2025?"),
    ("Q02", "alphabet", 2025, "What changed in Google Cloud revenue and operating profitability during 2025?"),
    ("Q03", "alphabet", 2024, "What AI-related business and product risks did Alphabet describe in 2024?"),
    ("Q04", "alphabet", 2024, "How did Alphabet discuss capital expenditures and technical infrastructure investment in 2024?"),
    ("Q05", "alphabet", 2023, "What competitive risks did Alphabet describe for search and online advertising?"),
    ("Q06", "alphabet", 2023, "What were the reasons for Google Cloud revenue changes in 2023?"),
    ("Q07", "meta", 2025, "What risks did Meta disclose about AI infrastructure and capital spending?"),
    ("Q08", "meta", 2025, "What factors affected Meta's operating expenses and investment spending in 2025?"),
    ("Q09", "meta", 2024, "What regulatory risks affected personalized advertising and data usage in 2024?"),
    ("Q10", "meta", 2024, "What factors contributed to Meta's advertising revenue growth in 2024?"),
    ("Q11", "meta", 2023, "What privacy and data protection regulation risks did Meta describe?"),
    ("Q12", "meta", 2023, "What does Meta report about Reality Labs operating losses and investment in 2023?"),
]

FIELDNAMES = [
    "question_id", "company", "fiscal_year", "question", "rank", "chunk_id",
    "cosine_distance", "source_url", "index_fingerprint", "passage",
    "relevance_grade", "review_notes",
]


def open_collection():
    if not CHROMA_DIR.is_dir():
        raise SystemExit("Vectorstore not found. Run Step 7A first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise SystemExit(f"Collection '{COLLECTION_NAME}' not found.") from exc
    if collection.count() < 1:
        raise SystemExit("Collection is empty. Run Step 7A first.")
    return collection


def prepare(overwrite: bool):
    if REVIEW_PATH.exists() and not overwrite:
        raise SystemExit(
            f"Review CSV already exists: {REVIEW_PATH}\n"
            "It may contain your manual labels. Use --overwrite only if you want to replace it."
        )
    collection = open_collection()
    fingerprint = str((collection.metadata or {}).get("input_fingerprint", ""))
    if not fingerprint:
        raise SystemExit("Collection is missing its input fingerprint; inspect Step 7A indexing.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for question_id, company, year, question in QUESTIONS:
        result = collection.query(
            query_texts=[question],
            n_results=TOP_K,
            where={"$and": [{"company": company}, {"fiscal_year": year}]},
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        if len(ids) != TOP_K:
            raise RuntimeError(f"{question_id}: expected {TOP_K} results, got {len(ids)}")
        for rank, (chunk_id, passage, meta, distance) in enumerate(
            zip(ids, docs, metas, distances), start=1
        ):
            if meta["company"] != company or int(meta["fiscal_year"]) != year:
                raise RuntimeError(f"{question_id}: metadata mismatch in {chunk_id}")
            rows.append({
                "question_id": question_id,
                "company": company,
                "fiscal_year": year,
                "question": question,
                "rank": rank,
                "chunk_id": chunk_id,
                "cosine_distance": f"{float(distance):.5f}",
                "source_url": meta["source_url"],
                "index_fingerprint": fingerprint,
                "passage": " ".join(passage.split()),
                "relevance_grade": "",
                "review_notes": "",
            })
        print(f"{question_id}: {company} FY{year} | {len(ids)} candidates | top distance {distances[0]:.4f}")

    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("\nRETRIEVAL EVALUATION CANDIDATES READY")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Candidate passages: {len(rows)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Index fingerprint: {fingerprint[:16]}...")
    print(f"Review file: {REVIEW_PATH.relative_to(ROOT)}")
    print("Next: read each passage and fill relevance_grade with 0, 1, or 2.")
    print("Do not change the other columns. Save the CSV before scoring.")


def ndcg_at_k(grades: list[int]) -> float:
    def dcg(values):
        return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(values, start=1))
    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal else 0.0


def score():
    if not REVIEW_PATH.exists():
        raise SystemExit(f"Missing review CSV. Run 'prepare' first: {REVIEW_PATH}")

    collection = open_collection()
    actual_fingerprint = str((collection.metadata or {}).get("input_fingerprint", ""))
    with REVIEW_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(QUESTIONS) * TOP_K:
        raise SystemExit(f"Expected {len(QUESTIONS)*TOP_K} rows, found {len(rows)}.")

    by_question: dict[str, list[dict]] = defaultdict(list)
    missing = []
    for row in rows:
        qid = row["question_id"]
        if row["index_fingerprint"] != actual_fingerprint:
            raise SystemExit(
                f"Index changed since candidate generation ({qid}). "
                "Do not compare scores across mismatched indexes without a new review."
            )
        grade = row["relevance_grade"].strip()
        if grade not in {"0", "1", "2"}:
            missing.append(f"{qid} rank {row['rank']}")
        else:
            row["grade_int"] = int(grade)
        by_question[qid].append(row)
    if missing:
        raise SystemExit(
            f"Missing/invalid relevance grades ({len(missing)}): "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
            + "\nFill ALL relevance_grade cells with 0, 1, or 2."
        )

    expected = {qid for qid, _, _, _ in QUESTIONS}
    if set(by_question) != expected:
        raise SystemExit("Question IDs differ from the fixed evaluation set.")

    stats = []
    for qid, company, year, question in QUESTIONS:
        candidate_rows = by_question[qid]
        if len(candidate_rows) != TOP_K or sorted(int(r["rank"]) for r in candidate_rows) != list(range(1, TOP_K + 1)):
            raise SystemExit(f"Invalid or duplicate ranks in {qid}")
        candidate_rows.sort(key=lambda r: int(r["rank"]))
        grades = [r["grade_int"] for r in candidate_rows]
        relevant = [1 if g >= 1 else 0 for g in grades]
        first_hit = next((i for i, flag in enumerate(relevant, start=1) if flag), None)
        stats.append({
            "question_id": qid,
            "company": company,
            "fiscal_year": year,
            "hit_at_5": int(any(relevant)),
            "precision_at_3": round(sum(relevant[:3]) / 3, 4),
            "mrr_at_5": round(1 / first_hit if first_hit else 0, 4),
            "ndcg_at_5": round(ndcg_at_k(grades), 4),
            "direct_evidence_at_5": int(2 in grades),
        })

    with METRICS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    keys = ["hit_at_5", "precision_at_3", "mrr_at_5", "ndcg_at_5", "direct_evidence_at_5"]
    overall = {key: sum(float(s[key]) for s in stats) / len(stats) for key in keys}
    lines = [
        "# Baseline RAG Retrieval Evaluation",
        "",
        f"- Questions: {len(stats)}",
        f"- Candidate chunks judged: {len(rows)}",
        f"- Collection: `{COLLECTION_NAME}`",
        f"- Index fingerprint: `{actual_fingerprint}`",
        "- Relevant threshold: grade >= 1 (human labels)",
        "- Grades: 0 = not useful; 1 = partial/contextual; 2 = directly useful",
        "",
        "## Aggregate metrics (macro average)",
        "",
    ]
    for key in keys:
        lines.append(f"- {key}: {overall[key]:.3f}")
    lines += [
        "",
        "## Limitations",
        "",
        "- The evaluation labels only the top five results retrieved by this baseline index.",
        "- This is a pooled judgment of retrieved candidates, not an exhaustive gold standard across all 3,284 chunks.",
        "- The number of questions is small and should be expanded for production claims.",
        "- This measures retrieval relevance, NOT the accuracy, citation correctness, or safety of LLM-generated answers.",
        "- Compare retrieval versions only after judging a shared candidate pool or assigning stable gold evidence passages.",
        "",
        "## Per-question data",
        "",
        f"See `{METRICS_PATH.relative_to(ROOT)}`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nRETRIEVAL RELEVANCE EVALUATION COMPLETE")
    print(f"Human-reviewed questions: {len(stats)}")
    print(f"Reviewed candidates: {len(rows)}")
    for key in keys:
        print(f"{key}: {overall[key]:.3f}")
    print(f"Metrics CSV: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--overwrite", action="store_true", help="Replace existing review CSV (prepare only).")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(overwrite=args.overwrite)
    elif args.command == "score":
        if args.overwrite:
            parser.error("--overwrite is only for 'prepare'")
        score()


if __name__ == "__main__":
    main()
