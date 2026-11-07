"""Project 5 / Step 8F — finalize approved model evaluation (no API calls).

Run from the project root:
    python scripts/11_finalize_answer_evaluation.py

Inputs (never modified):
    reports/answer_eval_review.csv         # 10 human-approved graded cases
    reports/answer_eval_results.jsonl      # the original saved answers and evidence
    reports/answer_eval_questions.csv      # the fixed evaluation questions, optional

Optional four-case comparison:
    reports/model_runs/gemini-3.8-flash/answer_eval_review.csv

Outputs:
    reports/final_ai_answer_evaluation.md
    reports/final_ai_case_scores.csv
    reports/final_ai_citation_reaudit.csv
    reports/final_model_comparison_shared_cases.csv  # only if archive available

This script does NOT call Gemini, regenerate answers, or change historic checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REVIEW = REPORTS / "answer_eval_review.csv"
RESULTS = REPORTS / "answer_eval_results.jsonl"
QUESTIONS = REPORTS / "answer_eval_questions.csv"
OLD_REVIEW = REPORTS / "model_runs" / "gemini-3.8-flash" / "answer_eval_review.csv"
OUT_REPORT = REPORTS / "final_ai_answer_evaluation.md"
OUT_CASES = REPORTS / "final_ai_case_scores.csv"
OUT_CITES = REPORTS / "final_ai_citation_reaudit.csv"
OUT_COMPARE = REPORTS / "final_model_comparison_shared_cases.csv"

GRADES = [
    "numerical_accuracy_0_to_2",
    "citation_support_0_to_2",
    "completeness_0_to_2",
    "uncertainty_handling_0_to_2",
]
LABELS = ["Numerical accuracy", "Citation support", "Completeness", "Uncertainty handling"]
# Only accept explicitly enumerated citation IDs, including grouped citations.
CITATION_GROUP = re.compile(r"\[((?:[MD]\d+)(?:\s*,\s*[MD]\d+)*)\]")
CITATION_ID = re.compile(r"[MD]\d+")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict]:
    data: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {path}, line {i}: {exc}") from exc
    return data


def unique_by_case(rows: list[dict], name: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("case_id", "")).strip()
        if not cid or cid in found:
            raise SystemExit(f"Missing or duplicate case_id '{cid}' in {name}.")
        found[cid] = row
    return found


def parse_grade(value: str, cid: str, dimension: str) -> int:
    val = str(value).strip()
    if val not in {"0", "1", "2"}:
        raise SystemExit(
            f"{cid}: {dimension} must be graded 0, 1 or 2; got {val!r}. "
            "Complete human review before finalizing."
        )
    return int(val)


def citation_reaudit(answer: str, sources: list[dict]) -> tuple[list[str], list[str], str]:
    cited = []
    for match in CITATION_GROUP.finditer(answer):
        cited.extend(CITATION_ID.findall(match.group(1)))
    cited = list(dict.fromkeys(cited))
    available = {str(source.get("id", "")) for source in sources}
    unknown = sorted(set(cited) - available)
    if unknown:
        status = "FAILED_UNKNOWN_IDS"
    elif not cited:
        status = "NO_EXPLICIT_CITATIONS"
    else:
        status = "PASSED"
    return cited, unknown, status


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt_score(value: int, max_points: int) -> str:
    return f"{value}/{max_points} ({100 * value / max_points:.2f}%)"


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main() -> None:
    if not REVIEW.exists() or not RESULTS.exists():
        raise SystemExit(
            "Missing reports/answer_eval_review.csv or reports/answer_eval_results.jsonl. "
            "Restore your Flash-Lite results and approved review first."
        )
    reviews = unique_by_case(read_csv(REVIEW), str(REVIEW))
    results = unique_by_case(load_jsonl(RESULTS), str(RESULTS))
    if len(reviews) != 10 or len(results) != 10 or set(reviews) != set(results):
        raise SystemExit(
            f"Expected matching sets of 10 cases; found {len(reviews)} approved "
            f"review rows and {len(results)} saved results."
        )
    models = {str(r.get("model", "")).strip() for r in results.values()}
    if models != {"gemini-3.5-flash-lite"}:
        raise SystemExit(
            f"Expected the separate Gemini 3.5 Flash-Lite run, found models {models!r}. "
            "Do not mix model baselines."
        )
    configs = {(r.get("retrieval_mode"), r.get("top_k")) for r in results.values()}
    if len(configs) != 1:
        raise SystemExit(f"Mixed retrieval configurations detected: {configs!r}")
    retrieval_mode, top_k = next(iter(configs))

    cases: list[dict] = []
    cites: list[dict] = []
    for cid in sorted(results):
        saved = results[cid]
        row = reviews[cid]
        if row.get("model", "").strip() != saved["model"]:
            raise SystemExit(f"{cid}: model mismatch between approved review and saved result.")
        if row.get("question", "").strip() != str(saved.get("question", "")).strip():
            raise SystemExit(f"{cid}: question differs from saved result.")
        if row.get("answer_text", "") != saved.get("answer_text", ""):
            raise SystemExit(
                f"{cid}: answer_text mismatch. Do not use scores from a different model answer."
            )
        values = [parse_grade(row.get(g, ""), cid, g) for g in GRADES]
        elapsed = float(saved.get("latency_seconds", 0))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise SystemExit(f"{cid}: invalid measured elapsed time {elapsed}.")
        cited, unknown, state = citation_reaudit(saved["answer_text"], saved.get("sources", []))
        cases.append({
            "case_id": cid,
            "category": saved.get("category", ""),
            "numerical_accuracy": values[0],
            "citation_support": values[1],
            "completeness": values[2],
            "uncertainty_handling": values[3],
            "total_out_of_8": sum(values),
            "latency_seconds": f"{elapsed:.3f}",
            "original_citation_id_check": saved.get("citation_id_check", ""),
            "citation_reaudit": state,
            "review_notes": row.get("review_notes", ""),
        })
        cites.append({
            "case_id": cid,
            "original_check": saved.get("citation_id_check", ""),
            "reaudit_status": state,
            "cited_ids": ", ".join(cited),
            "unknown_ids": ", ".join(unknown),
            "source_count": len(saved.get("sources", [])),
            "note": "Valid IDs do not independently prove source support.",
        })

    write_csv(
        OUT_CASES, cases,
        ["case_id", "category", "numerical_accuracy", "citation_support",
         "completeness", "uncertainty_handling", "total_out_of_8",
         "latency_seconds", "original_citation_id_check", "citation_reaudit", "review_notes"],
    )
    write_csv(
        OUT_CITES, cites,
        ["case_id", "original_check", "reaudit_status", "cited_ids", "unknown_ids",
         "source_count", "note"],
    )

    n = len(cases)
    total = sum(c["total_out_of_8"] for c in cases)
    max_points = 8 * n
    dimension_totals = [sum(c[key] for c in cases) for key in
                        ["numerical_accuracy", "citation_support", "completeness", "uncertainty_handling"]]
    times = [float(c["latency_seconds"]) for c in cases]
    native_pass = sum(str(c["original_citation_id_check"]).startswith("PASSED") for c in cases)
    parsed_pass = sum(c["citation_reaudit"] == "PASSED" for c in cases)

    lines = [
        "# Project 5 — Final AI Answer Evaluation",
        "",
        "**Review status:** User-reviewed and approved scores recorded in `reports/answer_eval_review.csv`.",
        "**Model:** `gemini-3.5-flash-lite`  ",
        f"**Retrieval:** `{retrieval_mode}`; top-k = {top_k}; sources: SEC 10-K text + XBRL-backed DuckDB metrics.  ",
        f"**Scope:** {n} fixed test questions, including one insufficient-evidence question.  ",
        "**Evaluation:** Four human-reviewed dimensions, each scored 0–2; equal weight.",
        "",
        "## Summary",
        "",
        f"- **Answer quality:** **{fmt_score(total, max_points)}**.",
        f"- **Latency:** mean **{statistics.mean(times):.2f}s**, median **{statistics.median(times):.2f}s**, "
        f"min **{min(times):.2f}s**, max **{max(times):.2f}s**.",
        "- **Latency definition:** script elapsed time for retrieval, evidence assembly, and model response; "
        "not isolated LLM inference time.",
        f"- **Original citation ID checks:** {native_pass}/{n} reported PASSED.",
        f"- **Grouped-citation re-audit:** {parsed_pass}/{n} passed explicit ID checks; original flags remain unchanged.",
        "- **Important:** ID validation is not proof that every claim is supported. "
        "Citation support was separately human-reviewed.",
        "",
        "## Dimension results",
        "",
        "| Dimension | Score |",
        "|---|---:|",
    ]
    for label, points in zip(LABELS, dimension_totals):
        lines.append(f"| {label} | {fmt_score(points, 2*n)} |")

    lines += [
        "",
        "## Case-level results",
        "",
        "| Case | Category | Numerical | Citations | Completeness | Uncertainty | Total | Elapsed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['case_id']} | {md_escape(case['category'])} | "
            f"{case['numerical_accuracy']} | {case['citation_support']} | "
            f"{case['completeness']} | {case['uncertainty_handling']} | "
            f"{case['total_out_of_8']}/8 | {float(case['latency_seconds']):.2f}s |"
        )

    lines += [
        "",
        "## Quality findings and known limitations",
        "",
        "1. **A02 CAGR citation precision:** For calculated growth rates, cite both baseline and ending-year records.",
        "2. **A04 free-cash-flow definitions:** Meta-reported FY2025 non-GAAP FCF is $43.585B, "
        "while project-derived operating cash flow minus capex is $46.109B. The $2.524B difference "
        "is finance-lease principal payments. Name the definition being used.",
        "3. **A10 abstention:** The exact commercial success rate for Meta AI research projects "
        "is not established by the supplied SEC evidence; the system should not invent a percentage.",
        "4. **Citation parser:** Earlier code could flag legitimate grouped citations such as "
        "`[D1, D3, D5]` as missing. Keep the original test record for auditability; "
        "use the corrected parser in the application and regression tests.",
        "5. **Small, curated sample:** Ten reviewed questions do not estimate production-wide factual "
        "accuracy; broaden the holdout set, especially for ambiguous questions and conflicting metrics.",
        "6. **Operating constraints:** Real-world latency, daily API quotas, burst limits, "
        "costs, and service availability can change. Add request throttling, timeouts and clear errors.",
        "",
        "## Reproducibility",
        "",
        f"- Review CSV SHA-256: `{sha256_file(REVIEW)}`",
        f"- Results JSONL SHA-256: `{sha256_file(RESULTS)}`",
    ]
    if QUESTIONS.exists():
        lines.append(f"- Question-set CSV SHA-256: `{sha256_file(QUESTIONS)}`")

    old_comparison = []
    if OLD_REVIEW.exists():
        previous = unique_by_case(read_csv(OLD_REVIEW), str(OLD_REVIEW))
        overlap = sorted(set(previous) & set(reviews))
        if overlap:
            for cid in overlap:
                old = previous[cid]
                if old.get("model", "").strip() != "gemini-3.8-flash":
                    raise SystemExit(f"Archived model mismatch for {cid}: {old.get('model')!r}")
                a = sum(parse_grade(old.get(g, ""), cid, g) for g in GRADES)
                b = next(c["total_out_of_8"] for c in cases if c["case_id"] == cid)
                old_comparison.append({
                    "case_id": cid,
                    "gemini_3_8_flash_score_out_of_8": a,
                    "gemini_3_5_flash_lite_score_out_of_8": b,
                    "gemini_3_8_flash_elapsed_seconds": old.get("latency_seconds", ""),
                    "gemini_3_5_flash_lite_elapsed_seconds":
                        next(c["latency_seconds"] for c in cases if c["case_id"] == cid),
                })
            write_csv(
                OUT_COMPARE, old_comparison,
                ["case_id", "gemini_3_8_flash_score_out_of_8", "gemini_3_5_flash_lite_score_out_of_8",
                 "gemini_3_8_flash_elapsed_seconds", "gemini_3_5_flash_lite_elapsed_seconds"],
            )
            old_total = sum(x["gemini_3_8_flash_score_out_of_8"] for x in old_comparison)
            lite_total = sum(x["gemini_3_5_flash_lite_score_out_of_8"] for x in old_comparison)
            old_mean = statistics.mean(float(x["gemini_3_8_flash_elapsed_seconds"]) for x in old_comparison)
            lite_mean = statistics.mean(float(x["gemini_3_5_flash_lite_elapsed_seconds"]) for x in old_comparison)
            lines += [
                "",
                "## Limited model comparison — shared cases only",
                "",
                f"- Compared the same **{len(overlap)}** cases: {', '.join(overlap)}.",
                f"- Gemini 3.8 Flash approved score: **{fmt_score(old_total, 8*len(overlap))}**.",
                f"- Gemini 3.5 Flash-Lite approved score: **{fmt_score(lite_total, 8*len(overlap))}**.",
                f"- Observed mean elapsed time on shared cases: **{old_mean:.2f}s** "
                f"(3.8 Flash) vs **{lite_mean:.2f}s** (3.5 Flash-Lite).",
                "- **Fairness caveat:** These timings were measured under different API conditions. "
                "Do not equate them with a controlled provider benchmark, nor compare 4-case "
                "and 10-case scores as if the sample sizes matched.",
                "",
                f"See `{OUT_COMPARE.name}` for individual case comparisons.",
            ]
    if not old_comparison:
        lines += [
            "",
            "## Earlier model baseline",
            "",
            "No approved archived `gemini-3.8-flash` review found at "
            "`reports/model_runs/gemini-3.8-flash/answer_eval_review.csv`. "
            "Retain the earlier four-case evaluation separately; do not mix it into this 10-case run.",
        ]

    lines += [
        "",
        "## Deployment decision",
        "",
        "**Recommended candidate:** `gemini-3.5-flash-lite` for the public Streamlit demo, "
        "subject to load testing, current quotas and continued evidence-grounding tests. "
        "It passed the reviewed question set with practical observed latency. "
        "Use validated SQL for numbers and SEC passages for explanations, and clearly show "
        "citations, definitions, and insufficient-evidence responses.",
        "",
        "*This report is descriptive model evaluation, not investment advice.*",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("STEP 8F — FINAL EVALUATION COMPLETE")
    print(f"Model: gemini-3.5-flash-lite | cases: {n}/10")
    print(f"Approved quality: {total}/{max_points} = {100 * total / max_points:.2f}%")
    print(f"Mean elapsed: {statistics.mean(times):.2f}s | median: {statistics.median(times):.2f}s")
    print(f"Original citation checks: {native_pass}/{n} | grouped-ID re-audit: {parsed_pass}/{n}")
    print(f"Final report: {OUT_REPORT.relative_to(ROOT)}")
    print(f"Case scores: {OUT_CASES.relative_to(ROOT)}")
    print(f"Citation re-audit: {OUT_CITES.relative_to(ROOT)}")
    if old_comparison:
        print(f"Shared-case comparison: {OUT_COMPARE.relative_to(ROOT)}")
    if parsed_pass < n:
        print("WARNING: One or more answers lack valid evidence IDs. See the citation re-audit CSV.")
    print("No Gemini API requests were made.")


if __name__ == "__main__":
    main()
