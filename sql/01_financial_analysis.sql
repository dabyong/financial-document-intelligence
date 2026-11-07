-- Step 6A: Business-facing metrics from audited SEC XBRL facts.
-- All monetary amounts are USD; _usd values are not scaled.
-- Operating cash flow and capex cover the full fiscal year.
-- Capital expenditures are positive cash-outflow values in the source.

CREATE OR REPLACE VIEW v_financial_kpis AS
WITH pivoted AS (
    SELECT
        company,
        fiscal_year,
        MAX(CASE WHEN metric = 'revenue' THEN value_usd END) AS revenue_usd,
        MAX(CASE WHEN metric = 'net_income' THEN value_usd END) AS net_income_usd,
        MAX(CASE WHEN metric = 'total_assets' THEN value_usd END) AS total_assets_usd,
        MAX(CASE WHEN metric = 'operating_income' THEN value_usd END) AS operating_income_usd,
        MAX(CASE WHEN metric = 'total_liabilities' THEN value_usd END) AS total_liabilities_usd,
        MAX(CASE WHEN metric = 'stockholders_equity' THEN value_usd END) AS stockholders_equity_usd,
        MAX(CASE WHEN metric = 'operating_cash_flow' THEN value_usd END) AS operating_cash_flow_usd,
        MAX(CASE WHEN metric = 'capital_expenditures' THEN value_usd END) AS capital_expenditures_usd
    FROM fact_financial_metrics
    GROUP BY company, fiscal_year
),
with_prior AS (
    SELECT *,
        LAG(revenue_usd) OVER (
            PARTITION BY company ORDER BY fiscal_year
        ) AS prior_year_revenue_usd
    FROM pivoted
)
SELECT
    company,
    fiscal_year,
    revenue_usd,
    net_income_usd,
    total_assets_usd,
    operating_income_usd,
    total_liabilities_usd,
    stockholders_equity_usd,
    operating_cash_flow_usd,
    capital_expenditures_usd,
    ROUND(100.0 * (revenue_usd / NULLIF(prior_year_revenue_usd, 0) - 1), 2)
        AS revenue_yoy_pct,
    ROUND(100.0 * net_income_usd / NULLIF(revenue_usd, 0), 2)
        AS net_margin_pct,
    ROUND(100.0 * operating_income_usd / NULLIF(revenue_usd, 0), 2)
        AS operating_margin_pct,
    ROUND(100.0 * operating_cash_flow_usd / NULLIF(revenue_usd, 0), 2)
        AS operating_cash_flow_margin_pct,
    ROUND(100.0 * total_liabilities_usd / NULLIF(total_assets_usd, 0), 2)
        AS liabilities_to_assets_pct,
    CASE WHEN capital_expenditures_usd >= 0
        THEN operating_cash_flow_usd - capital_expenditures_usd
    END AS free_cash_flow_usd,
    total_assets_usd - total_liabilities_usd - stockholders_equity_usd
        AS balance_sheet_difference_usd
FROM with_prior;

CREATE OR REPLACE VIEW v_company_comparison AS
WITH yearly AS (
    SELECT
        company,
        MAX(CASE WHEN fiscal_year = 2023 THEN revenue_usd END) AS revenue_2023_usd,
        MAX(CASE WHEN fiscal_year = 2025 THEN revenue_usd END) AS revenue_2025_usd,
        MAX(CASE WHEN fiscal_year = 2025 THEN revenue_yoy_pct END) AS growth_2025_pct,
        MAX(CASE WHEN fiscal_year = 2025 THEN net_margin_pct END) AS net_margin_2025_pct,
        MAX(CASE WHEN fiscal_year = 2025 THEN free_cash_flow_usd END) AS free_cash_flow_2025_usd
    FROM v_financial_kpis
    GROUP BY company
)
SELECT
    *,
    ROUND(100.0 * (POWER(revenue_2025_usd / NULLIF(revenue_2023_usd, 0), 0.5) - 1), 2)
        AS revenue_2yr_cagr_pct
FROM yearly;
