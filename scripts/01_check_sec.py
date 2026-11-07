import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()

if (
    not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", USER_AGENT)
    or "example.com" in USER_AGENT.lower()
):
    raise SystemExit(
        "Configure SEC_USER_AGENT with a real "
        "contact email in .env."
    )

COMPANIES = {
    "Alphabet": "0001652044",
    "Meta": "0001326801",
}

YEARS = {"2023", "2024", "2025"}

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
})


def fetch_json(url):
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise SystemExit(
            f"SEC request failed: {url}\n{exc}"
        ) from exc
    finally:
        time.sleep(1)


def find_annual_filings(records):
    found = {}

    for form, report_date, filed, accession, document in zip(
        records["form"],
        records["reportDate"],
        records["filingDate"],
        records["accessionNumber"],
        records["primaryDocument"],
    ):
        year = report_date[:4]

        if form == "10-K" and year in YEARS:
            if year not in found:
                found[year] = {
                    "filed": filed,
                    "accession": accession,
                    "document": document,
                }

    return found


for label, cik in COMPANIES.items():
    print(f"\nChecking {label}...")

    submissions = fetch_json(
        f"https://data.sec.gov/submissions/CIK{cik}.json"
    )

    facts = fetch_json(
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{cik}.json"
    )

    filings = find_annual_filings(
        submissions["filings"]["recent"]
    )

    missing = YEARS - set(filings)

    # Search the older SEC filing-history pages.
    if missing:
        print("Searching historical filings...")

        for page in submissions["filings"].get("files", []):
            if not missing:
                break

            # Target annual reports are filed in or
            # after the fiscal year being reported.
            if page["filingTo"] < "2023-01-01":
                continue

            url = (
                "https://data.sec.gov/submissions/"
                + page["name"]
            )

            older_records = fetch_json(url)
            older_filings = find_annual_filings(
                older_records
            )

            for year, details in older_filings.items():
                filings.setdefault(year, details)

            missing = YEARS - set(filings)

    print(f"Company: {submissions['name']}")
    print(f"CIK: {cik}")

    for year in sorted(filings):
        entry = filings[year]
        print(
            f"FY{year} 10-K | "
            f"Filed: {entry['filed']} | "
            f"Accession: {entry['accession']}"
        )

    concept_count = len(
        facts.get("facts", {}).get("us-gaap", {})
    )

    print(f"US-GAAP concepts: {concept_count}")

    if missing:
        raise SystemExit(
            f"Missing {label} fiscal years: "
            f"{sorted(missing)}"
        )

    if concept_count == 0:
        raise SystemExit(
            f"No US-GAAP financial facts for {label}."
        )

    print("STATUS: PASSED")

print("\nALL SIX ANNUAL FILINGS VERIFIED")
print("SEC API CONNECTION TEST COMPLETE")
