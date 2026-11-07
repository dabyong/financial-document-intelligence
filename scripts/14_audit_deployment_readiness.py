"""Offline pre-commit deployment audit. Prints only status, never secret values.

Run from the project root:
    python scripts/14_audit_deployment_readiness.py
Does not call Gemini, the SEC, or GitHub.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'deployment_assets' / 'financial_assets.zip'
MANIFEST = ROOT / 'deployment_assets' / 'manifest.json'
CHECKED = []
FAILS = []


def check(label: str, ok: bool, detail: str = ''):
    (CHECKED if ok else FAILS).append(label)
    print(('PASS ' if ok else 'FAIL ') + label + (f' — {detail}' if detail else ''))


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def main():
    check('Project is a Git repository', git('rev-parse', '--is-inside-work-tree').returncode == 0)
    for rel in ('.env', '.streamlit/secrets.toml', '.venv/example',
                'data/raw/excluded.json', 'data/runtime/ai_request_guard.sqlite3',
                '_step10a_update/installer.txt', '_step10b_update/installer.txt'):
        check(f'Git ignores {rel}', git('check-ignore', '-q', rel).returncode == 0)

    manifest = None
    if not ARCHIVE.is_file() or not MANIFEST.is_file():
        check('Deployment archive and manifest exist', False)
    else:
        check('Deployment archive and manifest exist', True)
        try:
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            with ARCHIVE.open('rb') as archive_file:
                digest = hashlib.file_digest(archive_file, 'sha256').hexdigest()
            check('Archive SHA-256 matches manifest', digest == manifest.get('archive_sha256'))
            check('Archive under 50 MiB', ARCHIVE.stat().st_size < 50 * 1024**2)
            with zipfile.ZipFile(ARCHIVE) as z:
                info = [i for i in z.infolist() if not i.is_dir()]
                names = [i.filename for i in info]
                valid = (
                    len(names) == len(set(names)) and
                    all(
                        not PurePosixPath(name).is_absolute()
                        and '..' not in PurePosixPath(name).parts
                        and '\\' not in name
                        and (name.startswith('data/warehouse/') or name.startswith('data/vectorstore/chroma/'))
                        for name in names
                    )
                )
                check('ZIP includes only expected SEC dataset paths', valid)
                check('ZIP internal CRC validation', z.testzip() is None)
                check('Archive file count matches manifest', len(names) == manifest.get('file_count'))
                check('Archive holds required warehouse and Chroma index',
                      'data/warehouse/financial_intelligence.duckdb' in names and
                      'data/vectorstore/chroma/chroma.sqlite3' in names)
                check('Manifest KPI and vector counts',
                      manifest.get('kpi_company_year_rows') == 6 and manifest.get('chroma_chunks') == 3284)
        except (ValueError, OSError, zipfile.BadZipFile, KeyError) as exc:
            check('Archive audit completed', False, type(exc).__name__)

    app = ROOT / 'app' / 'streamlit_app.py'
    guard = ROOT / 'app' / 'public_api_guard.py'
    bootstrap = ROOT / 'app' / 'deployment_bootstrap.py'
    check('App, deployment bootstrap and API guard exist',
          app.is_file() and guard.is_file() and bootstrap.is_file())
    if app.is_file():
        content = app.read_text(encoding='utf-8')
        check('App invokes shared request guard before Gemini',
              'reserve_request(ROOT)' in content and 'live_ai_enabled()' in content)
    if guard.is_file():
        content = guard.read_text(encoding='utf-8')
        check('Public live AI defaults to disabled',
              'os.getenv("P5_LIVE_AI_ENABLED", "false")' in content)

    req = (ROOT / 'requirements.txt')
    if req.exists():
        text = req.read_text(encoding='utf-8').lower()
        check('Cloud requirements include core dependencies',
              all(name in text for name in ['streamlit', 'duckdb', 'chromadb', 'google-genai', 'plotly']))
    else:
        check('Requirements file exists', False)

    # Scan Git candidate files (tracked and untracked, excluding ignored files).
    list_result = git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
    if list_result.returncode == 0:
        candidates = set(path for path in list_result.stdout.split('\x00') if path)

        allowed_placeholders = {
            'data/raw/.gitkeep',
            'data/processed/.gitkeep',
            'data/warehouse/.gitkeep',
            'data/vectorstore/.gitkeep',
        }


        forbidden = [
            p for p in candidates
            if p.replace('\\', '/').lower() not in allowed_placeholders
            and (
                p.lower() == '.env'
                or p.lower().endswith('/secrets.toml')
                or p.lower().startswith('data/raw/')
                or p.lower().startswith('data/runtime/')
                or p.lower().endswith(('.sqlite3', '.duckdb'))
            )
        ]

        check(
            'No sensitive/runtime paths are candidates for Git',
            not forbidden,
            f'{len(forbidden)} flagged paths: {", ".join(sorted(forbidden))}'
            if forbidden else ''
        )
        patterns = [
            re.compile(rb'AIza[A-Za-z0-9_-]{25,}'),
            re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
            re.compile(rb'(?i)GEMINI_API_KEY\s*=\s*["\']?AIza'),
        ]
        findings = []
        for rel in candidates:
            p = ROOT / rel
            if not p.is_file() or p.suffix in {'.zip', '.png', '.jpg', '.jpeg', '.pdf', '.duckdb'}:
                continue
            if p.stat().st_size > 5_000_000:
                continue
            data = p.read_bytes()
            if any(pattern.search(data) for pattern in patterns):
                findings.append(rel)
        check('No recognizable API keys/private-key headers in Git candidate text files',
              not findings, f'{len(findings)} flagged files' if findings else '')
    else:
        check('Git candidate audit available', False)

    print('\nSTEP 10B SECURITY AUDIT: ' + ('PASSED' if not FAILS else 'FAILED'))
    print(f'Checks passed: {len(CHECKED)} | failed: {len(FAILS)}')
    if FAILS:
        sys.exit(1)
    print('No Gemini requests made; this does not guarantee production security.')


if __name__ == '__main__':
    main()
