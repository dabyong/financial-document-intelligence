import csv
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "sec_filing_manifest.csv"
RAW = ROOT / "data" / "raw" / "sec"
DB_PATH = ROOT / "data" / "warehouse" / "financial_intelligence.duckdb"
AUDIT_PATH = ROOT / "reports" / "financial_metrics_audit.csv"

# Core metrics are mandatory; additional metrics are optional until verified.
METRICS = {
    "revenue": ("duration", True, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ]),
    "net_income": ("duration", True, ["NetIncomeLoss"]),
    "total_assets": ("instant", True, ["Assets"]),
    "operating_income": ("duration", False, ["OperatingIncomeLoss"]),
    "total_liabilities": ("instant", False, ["Liabilities"]),
    "stockholders_equity": ("instant", False, ["StockholdersEquity"]),
    "operating_cash_flow": ("duration", False, [
        "NetCashProvidedByUsedInOperatingActivities",
    ]),
    "capital_expenditures": ("duration", False, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ]),
}


def select_fact(us_gaap, concepts, kind, accession, report_date):
    """Only select USD facts from this exact annual filing and period."""
    year = int(report_date[:4])
    expected_start = f"{year}-01-01"

    for concept in concepts:
        records = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
        matches = [
            fact for fact in records
            if fact.get("accn") == accession
            and fact.get("form") == "10-K"
            and fact.get("end") == report_date
            and (
                fact.get("start") == expected_start
                if kind == "duration"
                else "start" not in fact
            )
        ]
        if matches:
            distinct_values = {str(fact["val"]) for fact in matches}
            if len(distinct_values) != 1:
                raise RuntimeError(
                    f"Conflicting observations for {concept}, {accession}: "
                    f"{sorted(distinct_values)}"
                )
            return concept, matches[0]

    return None, None


def main():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        filings = list(csv.DictReader(file))

    expected = {(c, str(y)) for c in ("alphabet", "meta") for y in range(2023, 2026)}
    actual = {(f["company"], f["fiscal_year"]) for f in filings}
    if len(filings) != 6 or actual != expected:
        raise RuntimeError("Expected exactly six distinct company/year filings.")

    audit_rows = []
    fact_rows = []

    for filing in sorted(filings, key=lambda r: (r["company"], r["fiscal_year"])):
        company = filing["company"]
        year = int(filing["fiscal_year"])
        report_date = filing["report_date"]
        accession = filing["accession"]
        companyfacts = RAW / company / "companyfacts.json"
        facts_json = json.loads(companyfacts.read_text(encoding="utf-8-sig"))
        us_gaap = facts_json["facts"]["us-gaap"]

        found = 0
        for metric, (kind, required, concepts) in METRICS.items():
            concept, observation = select_fact(
                us_gaap, concepts, kind, accession, report_date
            )
            status = "FOUND" if observation else "MISSING"
            audit_rows.append({
                "company": company,
                "fiscal_year": year,
                "metric": metric,
                "required": required,
                "status": status,
                "concept": concept or "",
                "value_usd": observation["val"] if observation else "",
                "period_start": observation.get("start", "") if observation else "",
                "period_end": observation["end"] if observation else "",
                "accession": accession,
            })
            if not observation:
                if required:
                    raise RuntimeError(
                        f"Missing REQUIRED {metric}: {company} FY{year}. "
                        "Inspect XBRL tags before choosing a substitute."
                    )
                continue

            value = observation["val"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeError(f"Non-numeric fact for {company} FY{year}: {metric}")
            if metric in ("revenue", "total_assets") and value <= 0:
                raise RuntimeError(f"Unexpected nonpositive {metric}: {company} FY{year}")

            fact_rows.append((
                company, year, metric, value, "us-gaap", concept, "USD", kind,
                observation.get("start"), observation["end"], accession,
                observation.get("filed"), filing["source_url"],
            ))
            found += 1

        print(f"{company.upper()} FY{year}: {found}/{len(METRICS)} metrics extracted")

    # Diagnostic accounting-equation check where all components are present.
    lookup = {(r[0], r[1], r[2]): r[3] for r in fact_rows}
    for company in ("alphabet", "meta"):
        for year in range(2023, 2026):
            k = lambda metric: lookup.get((company, year, metric))
            a, l, e = k("total_assets"), k("total_liabilities"), k("stockholders_equity")
            if a is not None and l is not None and e is not None:
                difference = abs(a - l - e)
                if difference > max(1_000_000, abs(a) * 0.001):
                    raise RuntimeError(
                        f"Balance-sheet reconciliation failed: {company} FY{year}; "
                        f"assets - liabilities - equity = {difference}"
                    )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DB_PATH))
    committed = False
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fact_financial_metrics (
                company VARCHAR NOT NULL,
                fiscal_year INTEGER NOT NULL,
                metric VARCHAR NOT NULL,
                value_usd DECIMAL(38, 2) NOT NULL,
                taxonomy VARCHAR NOT NULL,
                concept VARCHAR NOT NULL,
                unit VARCHAR NOT NULL,
                metric_kind VARCHAR NOT NULL,
                period_start DATE,
                period_end DATE NOT NULL,
                accession VARCHAR NOT NULL,
                filing_date DATE,
                source_url VARCHAR NOT NULL,
                PRIMARY KEY (company, fiscal_year, metric)
            )
        """)
        connection.execute("DELETE FROM fact_financial_metrics")
        connection.executemany("""
            INSERT INTO fact_financial_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, fact_rows)
        connection.execute("""
            CREATE OR REPLACE VIEW v_financial_summary AS
            SELECT company, fiscal_year,
                MAX(CASE WHEN metric = 'revenue' THEN value_usd END) AS revenue_usd,
                MAX(CASE WHEN metric = 'net_income' THEN value_usd END) AS net_income_usd,
                MAX(CASE WHEN metric = 'total_assets' THEN value_usd END) AS total_assets_usd,
                MAX(CASE WHEN metric = 'operating_income' THEN value_usd END) AS operating_income_usd,
                MAX(CASE WHEN metric = 'operating_cash_flow' THEN value_usd END) AS operating_cash_flow_usd
            FROM fact_financial_metrics
            GROUP BY company, fiscal_year
        """)
        connection.execute("""
            CREATE OR REPLACE VIEW v_revenue_growth AS
            WITH revenue AS (
                SELECT company, fiscal_year, value_usd AS revenue_usd,
                    LAG(value_usd) OVER (
                        PARTITION BY company ORDER BY fiscal_year
                    ) AS prior_year_revenue_usd
                FROM fact_financial_metrics WHERE metric = 'revenue'
            )
            SELECT company, fiscal_year, revenue_usd, prior_year_revenue_usd,
                CASE WHEN prior_year_revenue_usd > 0
                    THEN ROUND(100.0 * (revenue_usd / prior_year_revenue_usd - 1), 2)
                END AS yoy_growth_pct
            FROM revenue
        """)
        connection.execute("COMMIT")
        committed = True

        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)

        print("\nFINANCIAL WAREHOUSE COMPLETE")
        print(f"Metrics stored: {len(fact_rows)}")
        print("Required metrics: 18/18 verified")
        print("\nRevenue (USD billions) and YoY growth:")
        results = connection.execute("""
            SELECT company, fiscal_year,
                ROUND(revenue_usd / 1000000000, 2), yoy_growth_pct
            FROM v_revenue_growth ORDER BY company, fiscal_year
        """).fetchall()
        for company, year, revenue_b, growth in results:
            growth_text = "N/A" if growth is None else f"{growth}%"
            print(f"  {company:8} FY{year}: ${revenue_b}B | YoY {growth_text}")
        print(f"\nDatabase: {DB_PATH.relative_to(ROOT)}")
        print(f"Audit: {AUDIT_PATH.relative_to(ROOT)}")
    except Exception:
        if not committed:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
