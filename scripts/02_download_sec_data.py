import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sec"
REPORTS = ROOT / "reports"

load_dotenv(ROOT / ".env")

USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()

if (
    not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", USER_AGENT)
    or "example.com" in USER_AGENT.lower()
):
    raise SystemExit(
        "Set SEC_USER_AGENT with a real contact "
        "email in .env."
    )

COMPANIES = {
    "alphabet": "0001652044",
    "meta": "0001326801",
}
YEARS = {"2023", "2024", "2025"}

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
})


def download(url, path):
    """Download once and reuse the local snapshot."""
    if path.exists() and path.stat().st_size > 0:
        print(f"  CACHED: {path.name}")
        return path.read_bytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {path.name}")

    try:
        response = session.get(url, timeout=(15, 120))
        response.raise_for_status()
        content = response.content
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Download failed for {url}: {exc}"
        ) from exc
    finally:
        # Conservative SEC request pacing.
        time.sleep(1)

    if not content:
        raise RuntimeError(f"Empty response: {url}")

    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)

    return content


def load_json(url, path):
    return json.loads(download(url, path))


def extract_filings(records):
    """Find annual 10-Ks by fiscal report date."""
    found = {}

    for form, end, filed, acc, document in zip(
        records["form"],
        records["reportDate"],
        records["filingDate"],
        records["accessionNumber"],
        records["primaryDocument"],
    ):
        year = end[:4]

        if form == "10-K" and year in YEARS:
            found.setdefault(year, {
                "report_date": end,
                "filing_date": filed,
                "accession": acc,
                "document": document,
            })

    return found


def find_company_filings(cik, folder, submissions):
    found = extract_filings(
        submissions["filings"]["recent"]
    )

    if YEARS - set(found):
        pages = submissions["filings"].get("files", [])

        for page in pages:
            if not (YEARS - set(found)):
                break

            if page["filingTo"] < "2023-01-01":
                continue

            name = page["name"]
            historical = load_json(
                f"https://data.sec.gov/submissions/{name}",
                folder / name,
            )

            for year, entry in extract_filings(
                historical
            ).items():
                found.setdefault(year, entry)

    missing = YEARS - set(found)

    if missing:
        raise RuntimeError(
            f"Missing 10-K fiscal years: {sorted(missing)}"
        )

    return found


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = []
    audit_lines = [
        "# SEC Financial Data Acquisition Audit",
        "",
        "Status: PASSED",
        "",
        "Source: SEC EDGAR",
        "Scope: Alphabet and Meta, FY2023-FY2025",
        "",
    ]

    for company, cik in COMPANIES.items():
        print(f"\n=== {company.upper()} ===")
        folder = RAW / company

        submissions = load_json(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            folder / "submissions.json",
        )

        facts = load_json(
            "https://data.sec.gov/api/xbrl/"
            f"companyfacts/CIK{cik}.json",
            folder / "companyfacts.json",
        )

        concepts = len(
            facts.get("facts", {}).get("us-gaap", {})
        )

        if concepts == 0:
            raise RuntimeError(
                f"No US-GAAP concepts for {company}"
            )

        filings = find_company_filings(
            cik, folder, submissions
        )

        audit_lines.append(
            f"## {company.title()}"
        )
        audit_lines.append(
            f"- US-GAAP concepts: {concepts}"
        )

        for year in sorted(YEARS):
            entry = filings[year]
            accession = entry["accession"]
            document = entry["document"]

            if not re.fullmatch(
                r"[A-Za-z0-9_.-]+", document
            ):
                raise RuntimeError(
                    f"Unsafe document name: {document}"
                )

            url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/"
                f"{accession.replace('-', '')}/"
                f"{document}"
            )

            path = folder / f"FY{year}_10K.html"
            content = download(url, path)

            # Audit basic document usability.
            soup = BeautifulSoup(content, "lxml")
            text_length = len(
                soup.get_text(" ", strip=True)
            )

            if text_length < 10000:
                raise RuntimeError(
                    f"Unexpectedly short filing: "
                    f"{company} FY{year}, "
                    f"{text_length} text characters"
                )

            relative_path = path.relative_to(ROOT)

            manifest.append({
                "company": company,
                "cik": cik,
                "fiscal_year": year,
                "report_date": entry["report_date"],
                "filing_date": entry["filing_date"],
                "accession": accession,
                "source_url": url,
                "local_path": relative_path.as_posix(),
                "file_bytes": len(content),
                "text_characters": text_length,
                "sha256": sha256(content),
            })

            audit_lines.append(
                f"- FY{year}: PASS "
                f"({len(content):,} bytes, "
                f"{text_length:,} text characters)"
            )

            print(
                f"  FY{year}: PASS | "
                f"{len(content):,} bytes"
            )

        print(f"  US-GAAP concepts: {concepts}")
        audit_lines.append("")

    if len(manifest) != 6:
        raise RuntimeError(
            f"Expected 6 reports, got {len(manifest)}"
        )

    manifest_path = REPORTS / "sec_filing_manifest.csv"

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(manifest[0])
        )
        writer.writeheader()
        writer.writerows(manifest)

    audit_lines.extend([
        "## Final Validation",
        "- Annual reports: 6/6",
        "- Company Facts files: 2/2",
        "- Each document passed the basic text-length check",
        "- SHA-256 recorded for each annual report",
        "",
        "Note: Financial metric reconciliation and "
        "document semantic accuracy are not yet tested.",
    ])

    (REPORTS / "sec_data_quality_audit.md").write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    print("\nDATA ACQUISITION COMPLETE")
    print("Annual reports verified: 6/6")
    print("Company Facts files: 2/2")
    print(f"Manifest: {manifest_path.name}")
    print("Audit: sec_data_quality_audit.md")


if __name__ == "__main__":
    main()
