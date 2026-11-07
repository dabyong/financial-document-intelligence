"""Project 5 — AI Financial Document Intelligence (Streamlit portfolio demo).

Run from repository root:
    python -m streamlit run app/streamlit_app.py

Prerequisites:
    Step 6 DuckDB file, Step 7 Chroma index, scripts/09_ask_financial_ai.py.
The hosted deployment will require a separate data-asset packaging step.
"""
from __future__ import annotations

import importlib.util
import os
import zipfile
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from deployment_bootstrap import ensure_assets
from public_api_guard import RequestDenied, live_ai_enabled, reserve_request

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "warehouse" / "financial_intelligence.duckdb"
CHROMA_PATH = ROOT / "data" / "vectorstore" / "chroma"
QA_PATH = ROOT / "scripts" / "09_ask_financial_ai.py"
CSS_PATH = ROOT / "app" / "portfolio_theme.css"

NAVY = "#0A0F2E"
DEEP_NAVY = "#05081C"
MINT = "#2FD18F"
PAPER = "#FAF9FE"
INK = "#0F111A"
MUTED = "#5B5F6E"

st.set_page_config(
    page_title="Financial Intelligence | Project 05",
    page_icon=Image.open(BytesIO(bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000040000000400203000000d707994d0000000c504c54450a0f2efdfdfd8c8e9d707384897480680000000474524e53ffffffff6d284679000000644944415478da636018d18011896daac0f07f0d1392c0070606060626742d0324c0c2c0c0c0c0e9c3f07f0d2566883a30303030c028321df67a0303030303c3eb0370157f90028c8189818181671084183e010358988a7a400244941433a0befbc2300a0008ff0cd977d4d3fa0000000049454e44ae426082"
        ))),
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_dotenv(ROOT / ".env")

try:
    ASSET_STATUS = ensure_assets(ROOT)
except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
    st.error(f"Deployment assets cannot be loaded: {exc}")
    st.stop()

if CSS_PATH.is_file():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load only curated, read-only metrics from the validated warehouse."""
    import duckdb

    if not DB_PATH.is_file():
        raise FileNotFoundError("The DuckDB financial warehouse is missing. Complete Steps 5B–6.")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = connection.execute("""
            SELECT company, fiscal_year, revenue_usd, net_income_usd,
                   total_assets_usd, operating_income_usd,
                   operating_margin_pct, net_margin_pct, revenue_yoy_pct,
                   operating_cash_flow_usd, capital_expenditures_usd,
                   free_cash_flow_usd
            FROM v_financial_kpis
            ORDER BY company, fiscal_year
        """).df()
    finally:
        connection.close()
    if df.empty:
        raise ValueError("Financial warehouse contains no KPI rows.")
    for col in (
        "revenue_usd", "net_income_usd", "total_assets_usd", "operating_income_usd",
        "operating_margin_pct", "net_margin_pct", "revenue_yoy_pct",
        "operating_cash_flow_usd", "capital_expenditures_usd", "free_cash_flow_usd",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_qa_engine():
    """Reuse the tested Step 7C engine; don't duplicate retrieval or LLM logic."""
    if not QA_PATH.is_file():
        raise FileNotFoundError("Missing scripts/09_ask_financial_ai.py.")
    spec = importlib.util.spec_from_file_location("financial_qa_engine", QA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the financial QA engine.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def billions(number) -> str:
    if pd.isna(number):
        return "N/A"
    return f"${float(number) / 1_000_000_000:,.2f}B"


def percent(number) -> str:
    return "N/A" if pd.isna(number) else f"{float(number):,.2f}%"


def chart_style(fig: go.Figure, y_label: str = "USD billions") -> go.Figure:
    fig.update_layout(
        height=360,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="IBM Plex Mono, JetBrains Mono, monospace", color=INK, size=12),
        title=None,
        margin=dict(l=10, r=8, t=20, b=12),
        legend=dict(orientation="h", y=1.13, x=0),
        yaxis=dict(title=y_label, gridcolor="#E2E3EA", zeroline=False),
        xaxis=dict(title=None, showgrid=False, dtick=1),
        hovermode="x unified",
    )
    return fig


def chart_revenue(df: pd.DataFrame):
    fig = go.Figure()
    for company, color in (("alphabet", NAVY), ("meta", MINT)):
        subset = df[df.company == company].sort_values("fiscal_year")
        fig.add_trace(go.Scatter(
            x=subset.fiscal_year,
            y=subset.revenue_usd / 1e9,
            name=company.title(),
            mode="lines+markers",
            line=dict(width=3, color=color),
            marker=dict(size=9),
            hovertemplate="FY%{x}: $%{y:,.2f}B<extra>%{fullData.name}</extra>",
        ))
    return chart_style(fig)


def chart_margin(df: pd.DataFrame, year: int):
    subset = df[df.fiscal_year == year].sort_values("company")
    fig = go.Figure()
    for field, label, color in (
        ("operating_margin_pct", "Operating margin", NAVY),
        ("net_margin_pct", "Net margin", MINT),
    ):
        fig.add_trace(go.Bar(
            x=subset.company.str.title(),
            y=subset[field],
            name=label,
            marker_color=color,
            hovertemplate="%{x}: %{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(barmode="group")
    return chart_style(fig, "Percent")


def citations_in_text(answer: str, sources: list[dict]) -> str:
    """Link citation IDs to approved SEC URLs, including grouped [D1, M2] citations."""
    mapping = {
        s["id"]: s["url"] for s in sources
        if s.get("id") and str(s.get("url", "")).startswith(
            "https://www.sec.gov/Archives/edgar/data/"
        )
    }
    pattern = re.compile(r"\[((?:[MD]\d+)(?:\s*,\s*[MD]\d+)*)\]")

    def replace(match):
        ids = [part.strip() for part in match.group(1).split(",")]
        return ", ".join(f"[{cid}]({mapping[cid]})" if cid in mapping else f"[{cid}]" for cid in ids)

    return pattern.sub(replace, answer)


def check_citation_ids(answer: str, sources: list[dict]) -> tuple[str, str]:
    """Syntax and membership check only; doesn't establish semantic groundedness."""
    supplied = {s["id"] for s in sources}
    ids = []
    for candidate in re.findall(r"\[([^\[\]\n]{1,120})\]", answer):
        pieces = [p.strip() for p in candidate.split(",")]
        if pieces and all(re.fullmatch(r"[MD]\d+", p) for p in pieces):
            ids.extend(pieces)
    if not ids:
        return "NO_CITATIONS", "No evidence citations recognized; assess whether this is an appropriate abstention."
    invalid = sorted(set(ids) - supplied)
    if invalid:
        return "UNKNOWN_IDS", f"Unknown citation IDs: {', '.join(invalid)}"
    return "PASSED", f"{len(ids)} citation occurrences map to supplied evidence IDs."


def ask_financial_ai(question: str, model: str) -> dict:
    engine = load_qa_engine()
    companies, years = engine.pick_scope(question)
    metrics = engine.sql_context(companies, years)
    docs, plans = engine.documents_context_multi(question, companies, years, 10)
    prompt, sources = engine.assemble_prompt(question, metrics, docs)
    start = time.monotonic()
    answer = engine.generate_answer(prompt, model)
    latency = time.monotonic() - start
    status, detail = check_citation_ids(answer, sources)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "documents": docs,
        "latency": latency,
        "status": status,
        "citation_detail": detail,
        "query_count": len(plans),
        "model": model,
    }


# Deployment status is available in the Sources & methodology tab. The native
# Streamlit sidebar is not used so the app can have one clear branded header.

def show_portfolio_header() -> None:
    """Project 04-style header, with static, trusted HTML only."""
    st.markdown("""
    <div class="p5-brandbar">
      <div class="p5-brand">
        <a href="https://allandavecabanlong.vercel.app/" target="_blank" rel="noopener noreferrer">ALLAN_DAVE_CABANLONG</a>
      </div>
      <div class="p5-header-actions">
        <span class="p5-subbrand">PROJECT_05 // DATA_ANALYST + AI_FOCUS</span>
        <a class="p5-portfolio-return" href="https://allandavecabanlong.vercel.app/" aria-label="Return to portfolio">← Return to Portfolio</a>
      </div>
    </div>
    <div class="p5-hero">
      <div class="p5-eyebrow"><span class="p5-prompt">&gt;</span> ./projects/05_financial_document_intelligence</div>
      <h1>FINANCIAL_INTELLIGENCE<br><span>+ SOURCE_BACKED_AI</span></h1>
      <p>Analyze Alphabet and Meta's SEC filings, compare verified financial metrics,
      and ask evidence-backed questions using DuckDB, retrieval-augmented generation,
      and Gemini.</p>
      <div class="p5-statusline">
        <span class="p5-chip p5-chip-accent">SEC_10K / FY2023-2025</span>
        <span class="p5-chip">DUCKDB + CHROMADB</span>
        <span class="p5-chip">GEMINI_FLASH_LITE</span>
      </div>
    </div>
    <div class="p5-note"><strong>RESEARCH DEMO.</strong> AI responses are evidence-assisted, not guaranteed correct;
    verify conclusions against the original SEC filings. Not investment advice.</div>
    """, unsafe_allow_html=True)


show_portfolio_header()

try:
    data = load_metrics()
except Exception as exc:
    st.error(f"The financial dashboard cannot load: {exc}")
    st.info("Run the Step 5B/6 pipeline locally before starting this app. The database is excluded from Git.")
    st.stop()

page_overview, page_ai, page_sources = st.tabs(["Financial overview", "Ask Financial AI", "Sources & methodology"])

with page_overview:
    st.subheader("Company performance")
    available_years = sorted(int(y) for y in data.fiscal_year.unique())
    selected_year = st.selectbox("Fiscal year", available_years, index=len(available_years)-1, key="dashboard_year")
    scoped = data.loc[data.fiscal_year == selected_year].copy()
    cols = st.columns(2)
    for idx, company in enumerate(("alphabet", "meta")):
        rows = scoped[scoped.company == company]
        with cols[idx]:
            st.markdown(f"#### {company.title()}")
            if rows.empty:
                st.warning(f"No FY{selected_year} metrics for {company}.")
                continue
            record = rows.iloc[0]
            st.metric("Revenue", billions(record.revenue_usd), delta=(f"{percent(record.revenue_yoy_pct)} YoY" if pd.notna(record.revenue_yoy_pct) else None))
            a, b = st.columns(2)
            with a:
                st.metric("Operating margin", percent(record.operating_margin_pct))
            with b:
                st.metric("Net margin", percent(record.net_margin_pct))
            st.caption(f"Net income: {billions(record.net_income_usd)} · FCF (OCF − capex): {billions(record.free_cash_flow_usd)}")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Revenue trajectory")
        st.plotly_chart(chart_revenue(data), use_container_width=True)
    with right:
        st.markdown(f"#### Margins · FY{selected_year}")
        st.plotly_chart(chart_margin(data, selected_year), use_container_width=True)

    st.markdown("#### Detailed financial metrics")
    frame = scoped[[
        "company", "fiscal_year", "revenue_usd", "net_income_usd",
        "operating_margin_pct", "net_margin_pct", "free_cash_flow_usd", "total_assets_usd",
    ]].copy()
    for field in ("revenue_usd", "net_income_usd", "free_cash_flow_usd", "total_assets_usd"):
        frame[field] = frame[field].map(billions)
    for field in ("operating_margin_pct", "net_margin_pct"):
        frame[field] = frame[field].map(percent)
    frame.columns = ["Company", "FY", "Revenue", "Net income", "Operating margin", "Net margin", "FCF (OCF−Capex)", "Assets"]
    frame["Company"] = frame["Company"].str.title()
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption("Numbers derive from the audited local SEC XBRL warehouse. Free cash flow is OCF minus capex, which may differ from company-reported non-GAAP FCF.")

with page_ai:
    st.subheader("Ask the filings")
    st.write("Ask about Alphabet or Meta, fiscal years 2023–2025. Responses use audited SQL metrics and retrieved SEC passages.")
    examples = [
        "Why did Meta's net margin decline in FY2025 compared with FY2024?",
        "Compare Alphabet and Meta revenue growth from FY2023 to FY2025. Which had the higher CAGR?",
        "What risks did Meta disclose about AI infrastructure investment in FY2025?",
        "What influenced Alphabet's advertising and Google Cloud revenue growth in FY2025?",
    ]
    example = st.selectbox("Example question", ["Write my own question", *examples])
    default_question = "" if example == "Write my own question" else example
    if not live_ai_enabled():
        st.info("Live AI requests are disabled for this deployment. The financial dashboard and methodology remain available.")
    elif os.getenv("P5_ACCESS_CODE"):
        st.caption("Live AI access requires the demo passcode supplied by the project owner.")
    with st.form("financial_qa"):
        access_code = st.text_input("Demo access code", type="password") if os.getenv("P5_ACCESS_CODE") else ""
        question = st.text_area("Your question", value=default_question, height=110, max_chars=350, placeholder="Ask a question about SEC financial results or risks...")
        send = st.form_submit_button("Analyze financial filings", type="primary")

    if send:
        import hmac

        question = question.strip()
        if not live_ai_enabled():
            st.warning("Live AI is disabled. The dashboard is available without an API key.")
        elif os.getenv("P5_ACCESS_CODE") and not hmac.compare_digest(
            access_code, os.getenv("P5_ACCESS_CODE", "")
        ):
            st.warning("A valid demo access code is required to submit an AI question.")
        elif len(question) < 12:
            st.warning("Please provide a more specific financial question.")
        elif any(int(y) not in (2023, 2024, 2025) for y in re.findall(r"\b(20\d{2})\b", question)):
            st.warning("The dataset only includes FY2023 through FY2025. Please adjust the year in your question.")
        elif not DB_PATH.is_file() or not CHROMA_PATH.is_dir():
            st.error("The audited financial database or SEC vector index is missing.")
        elif not os.getenv("GEMINI_API_KEY"):
            st.error("Live AI is not configured for this deployment.")
        else:
            try:
                used, limit = reserve_request(ROOT)
            except RequestDenied as exc:
                st.warning(str(exc))
            except Exception:
                st.error("The shared AI budget cannot be verified. Live AI is temporarily unavailable.")
            else:
                chosen_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                with st.spinner("Retrieving SEC evidence and generating a grounded answer..."):
                    try:
                        result = ask_financial_ai(question, chosen_model)
                    except Exception as exc:
                        msg = str(exc)
                        if "429" in msg or "RateLimit" in type(exc).__name__:
                            st.error("Gemini's project quota was reached. No answer was generated; please return after its reset.")
                        elif "503" in msg or "service_unavailable" in msg.lower():
                            st.error("Gemini is temporarily overloaded. The financial dashboard remains available.")
                        elif "timeout" in msg.lower() or "Timeout" in type(exc).__name__:
                            st.error("Gemini timed out. Please try again later.")
                        else:
                            st.error("The AI request could not be completed. Please try again later.")
                    else:
                        st.session_state["latest_qa"] = result
                        st.caption(f"Shared AI request budget: {used}/{limit} attempts in the app's rolling 24-hour window.")

    result = st.session_state.get("latest_qa")
    if result:
        st.divider()
        st.markdown("#### AI analysis")
        st.caption(f"Model: {result['model']} · LLM request: {result['latency']:.1f}s · Retrieval queries: {result['query_count']}")
        st.markdown(citations_in_text(result["answer"], result["sources"]))
        if result["status"] == "PASSED":
            st.success("Citation IDs match the supplied evidence. Individual claims are not automatically verified.")
        else:
            st.warning(f"Citation check: {result['status']}. {result['citation_detail']}")
        with st.expander("Evidence references and excerpts", expanded=False):
            doc_map = {d["id"]: d for d in result["documents"]}
            for source in result["sources"]:
                sid = source["id"]
                st.markdown(f"**[{sid}] {source['company'].title()} FY{source['year']}** — [SEC filing]({source['url']})")
                if sid in doc_map:
                    st.caption(f"Chunk: {doc_map[sid]['chunk_id']}")
                    st.write(doc_map[sid]["passage"])
                else:
                    st.caption("Structured financial facts from the audited DuckDB warehouse.")
        st.caption("AI-generated research aid; not investment advice. SEC links identify filings, not verified sentence-level citations.")

with page_sources:
    st.subheader("Data provenance & methodology")
    with st.expander("Deployment status", expanded=False):
        st.caption(f"Warehouse: {'Available' if DB_PATH.is_file() else 'Missing'}")
        st.caption(f"Vector index: {'Available' if CHROMA_PATH.is_dir() else 'Missing'}")
        st.caption(f"Gemini key: {'Configured' if os.getenv('GEMINI_API_KEY') else 'Missing'}")
        st.caption("Research demonstration · Not investment advice")
    st.markdown("**Primary source:** SEC EDGAR 10-K filings for Alphabet and Meta, FY2023–FY2025.")
    st.markdown("**Structured metrics:** US-GAAP facts extracted from the SEC Company Facts API, validated and stored in DuckDB.")
    st.markdown("**Unstructured search:** SEC filings processed into chunks and indexed in ChromaDB using local MiniLM embeddings.")
    st.markdown("**LLM:** Gemini 3.5 Flash-Lite via the validated Step 7C pipeline; citation IDs are checked, not automatically semantically verified.")
    st.markdown("**Evaluation:** 10 reviewed financial Q&A cases; approved 79/80 points (98.75%). This small fixed set is not a production reliability guarantee.")
    audit = ROOT / "reports" / "final_ai_answer_evaluation.md"
    if audit.is_file():
        with st.expander("Model evaluation summary"):
            st.markdown(audit.read_text(encoding="utf-8-sig")[:16000])
    st.info("Deployment note: the DuckDB warehouse and Chroma index are local generated assets excluded from Git. The cloud release needs a controlled data-asset build or packaging process.")


st.markdown(
    '<div class="p5-footer">PROJECT_05 // AI FINANCIAL DOCUMENT INTELLIGENCE · '
    'PORTFOLIO RESEARCH DEMONSTRATION</div>',
    unsafe_allow_html=True,
)
