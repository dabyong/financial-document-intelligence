"""Local Step 9A readiness check. No API calls or writes to project data."""
from __future__ import annotations

import ast
import os
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

required_files = {
    'Streamlit app': ROOT / 'app' / 'streamlit_app.py',
    'Portfolio stylesheet': ROOT / 'app' / 'portfolio_theme.css',
    'Streamlit theme': ROOT / '.streamlit' / 'config.toml',
    'QA engine': ROOT / 'scripts' / '09_ask_financial_ai.py',
    'DuckDB warehouse': ROOT / 'data' / 'warehouse' / 'financial_intelligence.duckdb',
}
failures = []
for label, path in required_files.items():
    present = path.is_file()
    print(f"{label:23}: {'OK' if present else 'MISSING'}")
    if not present:
        failures.append(label)

vector_path = ROOT / 'data' / 'vectorstore' / 'chroma'
print(f"Chroma vectorstore      : {'OK' if vector_path.is_dir() else 'MISSING'}")
if not vector_path.is_dir():
    failures.append('Chroma vectorstore')

for package in ('streamlit', 'duckdb', 'chromadb', 'plotly', 'google.genai'):
    try:
        __import__(package)
    except ImportError as exc:
        print(f'{package:23}: NOT INSTALLED ({exc})')
        failures.append(package)
    else:
        print(f'{package:23}: OK')

if (ROOT / 'app' / 'streamlit_app.py').is_file():
    ast.parse((ROOT / 'app' / 'streamlit_app.py').read_text(encoding='utf-8'))
if (ROOT / '.streamlit' / 'config.toml').is_file():
    with (ROOT / '.streamlit' / 'config.toml').open('rb') as handle:
        config = tomllib.load(handle)
    if config.get('theme', {}).get('primaryColor') != '#2FD18F':
        failures.append('Portfolio theme color')

if (ROOT / 'scripts' / '09_ask_financial_ai.py').is_file():
    try:
        spec = (ROOT / 'scripts' / '09_ask_financial_ai.py').read_text(encoding='utf-8')
        tree = ast.parse(spec)
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ('pick_scope', 'sql_context', 'documents_context_multi', 'assemble_prompt', 'generate_answer'):
            if name not in function_names:
                failures.append(f'QA function: {name}')
    except Exception as exc:
        failures.append(f'QA script parsing: {exc}')

try:
    import duckdb
    db = required_files['DuckDB warehouse']
    if db.is_file():
        conn = duckdb.connect(str(db), read_only=True)
        try:
            count = conn.execute('SELECT COUNT(*) FROM v_financial_kpis').fetchone()[0]
            print(f'KPI records            : {count}')
            if count != 6:
                failures.append(f'Expected 6 KPI records; got {count}')
        finally:
            conn.close()
except Exception as exc:
    failures.append(f'Warehouse read: {type(exc).__name__}: {exc}')

print(f"GEMINI_MODEL            : {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}")
print(f"GEMINI_API_KEY          : {'SET (value hidden)' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
if not os.getenv('GEMINI_API_KEY'):
    failures.append('GEMINI_API_KEY')

if failures:
    print('\nSTEP 9A SETUP: NOT READY')
    for item in failures:
        print('  -', item)
    sys.exit(1)
print('\nSTEP 9A SETUP: READY — No Gemini requests made')
