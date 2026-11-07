"""Step 8A: prepare and run a reproducible, human-reviewed financial Q&A test suite.

From the financial-document-intelligence project root:
    python scripts/10_evaluate_ai_answers.py prepare
    python scripts/10_evaluate_ai_answers.py run --limit 2
    python scripts/10_evaluate_ai_answers.py run

The script reuses scripts/09_ask_financial_ai.py and never sends expected answers
or grading notes to Gemini. Its automatic citation check validates IDs, not truth.
Keep the private .env local; review outputs contain public SEC evidence only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
QUESTIONS_PATH = REPORTS / "answer_eval_questions.csv"
RESULTS_PATH = REPORTS / "answer_eval_results.jsonl"
REVIEW_PATH = REPORTS / "answer_eval_review.csv"
QA_SCRIPT = ROOT / "scripts" / "09_ask_financial_ai.py"

# Ground-truth expectations are for manual evaluation ONLY; never sent to the LLM.
CASES = [
    (
        "A01", "Causal financial analysis",
        "Why did Meta's net margin decline in FY2025 compared with FY2024?",
        "Net margin 37.91% -> 30.08%; tax provision increase; OBBBA $15.93B charge; distinguish operating versus tax drivers.",
    ),
    (
        "A02", "Financial comparison",
        "Compare Alphabet and Meta revenue growth from FY2023 to FY2025. Which grew faster on a CAGR basis?",
        "Alphabet revenue CAGR 14.48%; Meta 22.05%; Meta grew faster; cite XBRL-backed metrics.",
    ),
    (
        "A03", "Year-over-year growth",
        "How much did Alphabet's revenue grow in FY2025 compared with FY2024?",
        "Alphabet FY2024 revenue $350.018B; FY2025 revenue $402.836B; YoY +15.09%.",
    ),
    (
        "A04", "Non-GAAP calculation",
        "What was Meta's free cash flow in FY2025, and how is that metric calculated?",
        "Meta FY2025 free cash flow approximately $46.11B; operating cash flow minus capital expenditures; derived, not a GAAP line item.",
    ),
    (
        "A05", "Profitability comparison",
        "Compare Alphabet and Meta operating margins in FY2025. Which had the higher margin?",
        "Alphabet 32.03%; Meta 41.44%; Meta higher by 9.41 percentage points.",
    ),
    (
        "A06", "Risk disclosure",
        "What risks did Meta disclose about AI infrastructure investment in FY2025?",
        "Should discuss specific risks substantiated in Meta FY2025 10-K; no invented probabilities, amounts, or outcomes.",
    ),
    (
        "A07", "Business drivers",
        "What factors influenced Alphabet's advertising and Google Cloud revenue growth in FY2025?",
        "Distinguish company-disclosed business drivers from reported segment figures; support causal claims with SEC passages.",
    ),
    (
        "A08", "Regulatory risk",
        "What privacy and regulatory risks did Meta describe in its FY2023 annual report?",
        "Discuss disclosed government restrictions, privacy/data regulations and their effects; cite FY2023 passages.",
    ),
    (
        "A09", "Financial position",
        "Compare Alphabet's and Meta's total assets in FY2025. Which had a larger asset base?",
        "Use FY2025 year-end total assets for both companies from SQL; do not substitute revenue.",
    ),
    (
        "A10", "Evidence sufficiency",
        "What exact percentage of Meta's FY2025 AI research projects were commercially successful?",
        "A precise percentage is not established by the available annual filings; explicitly state insufficient evidence rather than inventing one.",
    ),
]
FIELDS = ["case_id", "category", "question", "expected_checks"]
REVIEW_FIELDS = [
    "case_id", "category", "question", "retrieval_mode", "model",
    "citation_id_check", "latency_seconds", "answer_text", "source_ids",
    "numerical_accuracy_0_to_2", "citation_support_0_to_2",
    "completeness_0_to_2", "uncertainty_handling_0_to_2", "review_notes",
]
GRADE_FIELDS = REVIEW_FIELDS[-5:]


def prepare():
    REPORTS.mkdir(parents=True, exist_ok=True)
    if QUESTIONS_PATH.exists():
        with QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
            prior = list(csv.DictReader(f))
        if len(prior) != len(CASES):
            raise SystemExit(
                f"Existing question set has {len(prior)} cases, not {len(CASES)}. "
                "Back it up before changing it."
            )
        print(f"Question set already exists: {QUESTIONS_PATH}")
    else:
        with QUESTIONS_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(CASES)
        print(f"Created question set: {QUESTIONS_PATH}")
    print(f"Cases: {len(CASES)}")
    print("Contains quantitative, qualitative and insufficient-evidence questions.")
    print("Review expected_checks before generating responses; they are NEVER sent to Gemini.")
    print("PREPARATION COMPLETE — No Gemini calls made.")


def load_cases():
    if not QUESTIONS_PATH.is_file():
        raise SystemExit("Run 'prepare' before 'run'.")
    with QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
        cases = list(csv.DictReader(f))
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)) or not all(c.get("question", "").strip() for c in cases):
        raise SystemExit("Question IDs must be unique and questions nonempty.")
    return cases


def load_results():
    results = {}
    if not RESULTS_PATH.exists():
        return results
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = row["case_id"]
            if key in results:
                raise SystemExit(f"Duplicate result in JSONL: {key}")
            results[key] = row
    return results


def import_qa_module():
    if not QA_SCRIPT.exists():
        raise SystemExit(f"Missing existing Q&A script: {QA_SCRIPT}")
    spec = importlib.util.spec_from_file_location("financial_qa_eval", QA_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit("Could not import financial Q&A script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    needed = ["pick_scope", "sql_context", "documents_context_multi", "documents_context",
              "assemble_prompt", "generate_answer", "validate_citation_ids"]
    if not all(hasattr(module, name) for name in needed):
        raise SystemExit("Q&A script is missing a required function; use Step 7C multquery script.")
    return module


def existing_grades():
    grades = {}
    if REVIEW_PATH.exists():
        with REVIEW_PATH.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                grades[row["case_id"]] = (
                    hashlib.sha256(row.get("answer_text", "").encode("utf-8")).hexdigest(),
                    {k: row.get(k, "") for k in GRADE_FIELDS},
                )
    return grades


def rebuild_review(cases, results):
    prior = existing_grades()
    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            result = results.get(case["case_id"])
            if not result:
                continue
            row = {key: result.get(key, "") for key in REVIEW_FIELDS}
            row["source_ids"] = ", ".join(s["id"] for s in result.get("sources", []))
            row["latency_seconds"] = f"{result.get('latency_seconds', 0):.2f}"
            prev_hash, grades = prior.get(case["case_id"], ("", {}))
            if prev_hash == hashlib.sha256(str(row["answer_text"]).encode("utf-8")).hexdigest():
                row.update(grades)
            writer.writerow(row)
    print(f"Review sheet: {REVIEW_PATH}")
    print("Grade the four 0-2 columns manually; automatic ID checks do NOT establish truth.")


def run(limit, retrieval_mode, top_k, cooldown):
    from dotenv import load_dotenv
    import os

    load_dotenv(ROOT / ".env")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
    cases = load_cases()
    results = load_results()
    if results:
        configurations = {
            (r.get("retrieval_mode"), r.get("model"), r.get("top_k"))
            for r in results.values()
        }
        if configurations != {(retrieval_mode, model, top_k)}:
            raise SystemExit(
                "The existing results used a different retrieval mode, model, or top-k. "
                "Archive the JSONL and review CSV before starting a different run."
            )
    qa = import_qa_module()
    completed = 0
    for case in cases:
        cid = case["case_id"]
        if cid in results:
            print(f"SKIP {cid}: already generated")
            continue
        if completed >= limit:
            break
        question = case["question"]
        print(f"\nGENERATING {cid}: {question}", flush=True)
        started = time.perf_counter()
        try:
            companies, years = qa.pick_scope(question)
            metrics = qa.sql_context(companies, years)
            if retrieval_mode == "multi":
                docs, _plans = qa.documents_context_multi(question, companies, years, top_k)
            else:
                docs = qa.documents_context(question, companies, years, top_k)
            prompt, sources = qa.assemble_prompt(question, metrics, docs)
            answer = qa.generate_answer(prompt, model)
            check = "PASSED"
            try:
                qa.validate_citation_ids(answer, sources)
            except (RuntimeError, ValueError) as exc:
                check = f"FAILED: {exc}"
            record = {
                "case_id": cid,
                "category": case.get("category", ""),
                "question": question,
                "retrieval_mode": retrieval_mode,
                "model": model,
                "top_k": top_k,
                "citation_id_check": check,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "answer_text": answer,
                "sources": sources,
                "source_passages": [
                    {"id": d["id"], "chunk_id": d["chunk_id"], "passage": d["passage"],
                     "url": d["url"], "company": d["company"], "year": d["year"]}
                    for d in docs
                ],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        except Exception as exc:
            print(f"ERROR on {cid}: {type(exc).__name__}: {exc}", flush=True)
            print("No result recorded for failed case; rerun to retry. Earlier results are preserved.")
            rebuild_review(cases, results)
            raise SystemExit(1) from exc
        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        results[cid] = record
        completed += 1
        print(f"SAVED {cid} | citation IDs: {check} | {record['latency_seconds']:.1f}s", flush=True)
        if cooldown and completed < limit:
            time.sleep(cooldown)
    rebuild_review(cases, results)
    print(f"\nANSWER EVALUATION GENERATION COMPLETE: {len(results)}/{len(cases)} cases saved")
    print("Now manually inspect answer claims against stored passages and SEC filings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--limit", type=int, default=10, help="Maximum new API generations (default: 10)")
    parser.add_argument("--retrieval-mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--cooldown", type=float, default=2.0, help="Pause between Gemini calls")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.limit < 1 or args.cooldown < 0:
            parser.error("--limit must be positive and --cooldown nonnegative")
        run(args.limit, args.retrieval_mode, args.top_k, args.cooldown)


if __name__ == "__main__":
    main()
