"""Build Step 6 analytical views, assert data quality and export reproducible results."""
import csv
from decimal import Decimal
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "warehouse" / "financial_intelligence.duckdb"
SQL_FILE = ROOT / "sql" / "01_financial_analysis.sql"
REPORTS = ROOT / "reports"
EXPECTED_COMPANIES = {"alphabet", "meta"}
EXPECTED_YEARS = {2023, 2024, 2025}
EXPECTED_METRICS = {
    "revenue", "net_income", "total_assets", "operating_income",
    "total_liabilities", "stockholders_equity", "operating_cash_flow",
    "capital_expenditures",
}


def assert_ok(condition, message):
    if not condition:
        raise AssertionError(message)


def export_query(con, sql, path):
    result = con.execute(sql)
    columns = [column[0] for column in result.description]
    records = result.fetchall()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(records)
    return records


def main():
    assert_ok(DATABASE.exists(), "Database missing: run Step 5B first.")
    assert_ok(SQL_FILE.exists(), "SQL file missing: save 01_financial_analysis.sql to sql/.")
    REPORTS.mkdir(exist_ok=True)
    con = duckdb.connect(str(DATABASE))
    try:
        con.execute(SQL_FILE.read_text(encoding="utf-8-sig"))

        # Check source records before trusting the derived views.
        source = con.execute("""
            SELECT company, fiscal_year, metric, COUNT(*) AS n
            FROM fact_financial_metrics
            GROUP BY company, fiscal_year, metric
        """).fetchall()
        assert_ok(len(source) == 48, f"Expected 48 source metric rows; found {len(source)}")
        assert_ok(all(n == 1 for _, _, _, n in source), "Duplicate company-year-metric")
        assert_ok({c for c, _, _, _ in source} == EXPECTED_COMPANIES,
                  "Unexpected company coverage")
        assert_ok({y for _, y, _, _ in source} == EXPECTED_YEARS,
                  "Unexpected fiscal year coverage")
        assert_ok({m for _, _, m, _ in source} == EXPECTED_METRICS,
                  "Unexpected metric coverage")
        for company in EXPECTED_COMPANIES:
            for year in EXPECTED_YEARS:
                actual = {metric for c, y, metric, _ in source if c == company and y == year}
                assert_ok(actual == EXPECTED_METRICS,
                          f"Missing metric: {company} FY{year}: {EXPECTED_METRICS - actual}")

        results = con.execute("""
            SELECT k.company, k.fiscal_year,
                   k.revenue_usd, k.net_income_usd,
                   k.total_assets_usd, k.operating_cash_flow_usd,
                   k.capital_expenditures_usd, k.free_cash_flow_usd,
                   k.revenue_yoy_pct, g.yoy_growth_pct,
                   k.balance_sheet_difference_usd
            FROM v_financial_kpis k
            JOIN v_revenue_growth g USING (company, fiscal_year)
            ORDER BY k.company, k.fiscal_year
        """).fetchall()
        assert_ok(len(results) == 6, f"Expected 6 KPI rows; found {len(results)}")
        for (company, year, revenue, net_income, assets, op_cash,
             capex, fcf, yoy, earlier_yoy, difference) in results:
            assert_ok(revenue > 0 and assets > 0,
                      f"Nonpositive revenue or assets: {company} FY{year}")
            assert_ok(capex >= 0, f"Negative capex outflow: {company} FY{year}")
            assert_ok(fcf == op_cash - capex, f"Incorrect FCF: {company} FY{year}")
            assert_ok(yoy == earlier_yoy,
                      f"YoY disagrees with Step 5B: {company} FY{year}: {yoy} != {earlier_yoy}")
            assert_ok((yoy is None) == (year == 2023),
                      f"Unexpected YoY availability: {company} FY{year}")
            assert_ok(abs(difference) <= max(Decimal('1000000'), abs(assets) * Decimal('0.001')),
                      f"Balance sheet doesn't reconcile: {company} FY{year}")

        comparison = con.execute("""
            SELECT company, revenue_2023_usd, revenue_2025_usd,
                   growth_2025_pct, revenue_2yr_cagr_pct
            FROM v_company_comparison ORDER BY company
        """).fetchall()
        assert_ok(len(comparison) == 2, "Expected 2 companies in comparison view")
        assert_ok(all(start > 0 and finish > 0 and cagr is not None
                      for _, start, finish, _, cagr in comparison),
                  "Invalid 2023–2025 comparison")

        export_query(con, """
            SELECT company, fiscal_year,
                   revenue_usd, net_income_usd, operating_income_usd,
                   total_assets_usd, total_liabilities_usd,
                   stockholders_equity_usd, operating_cash_flow_usd,
                   capital_expenditures_usd, free_cash_flow_usd,
                   revenue_yoy_pct, net_margin_pct,
                   operating_margin_pct, operating_cash_flow_margin_pct,
                   liabilities_to_assets_pct, balance_sheet_difference_usd
            FROM v_financial_kpis ORDER BY company, fiscal_year
        """, REPORTS / "financial_kpis.csv")
        export_query(con, """
            SELECT * FROM v_company_comparison ORDER BY company
        """, REPORTS / "company_comparison.csv")

        print("FINANCIAL ANALYTICS VALIDATION COMPLETE")
        print("Source metric coverage: 48/48")
        print("KPI company-year rows: 6/6")
        print("Source uniqueness, balance-sheet, FCF and YoY checks: PASSED")
        print("\nFY2025 business KPI summary (USD billions):")
        for company, revenue, income, op_margin, net_margin, cash in con.execute("""
            SELECT company, ROUND(revenue_usd / 1e9, 2),
                   ROUND(net_income_usd / 1e9, 2),
                   operating_margin_pct, net_margin_pct,
                   ROUND(free_cash_flow_usd / 1e9, 2)
            FROM v_financial_kpis
            WHERE fiscal_year = 2025 ORDER BY company
        """).fetchall():
            print(f"  {company}: revenue ${revenue}B | net income ${income}B | "
                  f"operating margin {op_margin}% | net margin {net_margin}% | "
                  f"free cash flow ${cash}B")
        print("\nRevenue growth (2023–2025 annualized):")
        for company, _, _, _, cagr in comparison:
            print(f"  {company}: {cagr}% CAGR")
        print("\nReports: reports/financial_kpis.csv, reports/company_comparison.csv")
    finally:
        con.close()


if __name__ == "__main__":
    main()
