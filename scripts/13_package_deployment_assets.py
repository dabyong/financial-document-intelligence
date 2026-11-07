"""Step 10A: package ONLY generated, public SEC research assets for deployment.

Run from the repository root AFTER stopping Streamlit:
    python scripts/13_package_deployment_assets.py

Does not read .env, call Gemini, or upload data. Requires local DuckDB/Chroma.
Creates deployment_assets/financial_assets.zip and manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse" / "financial_intelligence.duckdb"
CHROMA = ROOT / "data" / "vectorstore" / "chroma"
ARCHIVE_DIR = ROOT / "deployment_assets"
ARCHIVE = ARCHIVE_DIR / "financial_assets.zip"
MANIFEST = ARCHIVE_DIR / "manifest.json"
COLLECTION = "sec_10k_minilm_v1"
EXPECTED_COUNT = 3284
LIMIT_MB = 50


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not DB.is_file():
        raise SystemExit("Missing local DuckDB file: data/warehouse/financial_intelligence.duckdb")
    if not (CHROMA / "chroma.sqlite3").is_file():
        raise SystemExit("Missing Chroma index: data/vectorstore/chroma/chroma.sqlite3")

    active = [p for p in [Path(str(DB) + ".wal"), CHROMA / "chroma.sqlite3-wal", CHROMA / "chroma.sqlite3-shm"] if p.exists()]
    if active:
        raise SystemExit("Stop Streamlit before packaging. Active database journal(s): " + ", ".join(str(p) for p in active))

    # Check saved assets using the same libraries as the running app.
    try:
        import duckdb
        import chromadb
    except ImportError as exc:
        raise SystemExit(f"Activate the project .venv and install requirements first: {exc}") from exc

    con = duckdb.connect(str(DB), read_only=True)
    try:
        row_count = con.execute("SELECT count(*) FROM v_financial_kpis").fetchone()[0]
        pairs = set(con.execute("SELECT company, fiscal_year FROM v_financial_kpis").fetchall())
    finally:
        con.close()
    expected_pairs = {(c, y) for c in ("alphabet", "meta") for y in (2023, 2024, 2025)}
    if row_count != 6 or pairs != expected_pairs:
        raise SystemExit(f"Invalid financial KPI dataset (rows={row_count}, pairs={pairs})")

    client = chromadb.PersistentClient(path=str(CHROMA))
    collection = client.get_collection(COLLECTION)
    indexed = collection.count()
    if indexed != EXPECTED_COUNT:
        raise SystemExit(f"Chroma index count mismatch: {indexed} != {EXPECTED_COUNT}")

    # A live SQLite WAL/journal can make a copy inconsistent. Do not bundle it.
    active_after_read = [p for p in [Path(str(DB) + ".wal"), CHROMA / "chroma.sqlite3-wal", CHROMA / "chroma.sqlite3-shm"] if p.exists()]
    if active_after_read:
        raise SystemExit("Database journals detected after index verification. Close all Streamlit/Chroma processes and re-run: " + ", ".join(str(p) for p in active_after_read))

    files = [DB] + sorted(p for p in CHROMA.rglob("*") if p.is_file())
    for path in files:
        if path.is_symlink():
            raise SystemExit(f"Symlink in deployment data is not allowed: {path}")
        if path.name.lower().endswith(("-wal", "-shm", ".wal", ".shm", ".lock", ".tmp", ".part")):
            raise SystemExit(f"Temporary/database lock file found: {path}")
    if len(files) < 2:
        raise SystemExit("No Chroma index files found")

    ARCHIVE_DIR.mkdir(exist_ok=True)
    tmp = ARCHIVE.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in files:
                z.write(path, arcname=path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(tmp, mode="r") as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"Corrupt archive entry: {bad}")
            if len(z.namelist()) != len(files):
                raise RuntimeError("Archive file count differs from source")
        tmp.replace(ARCHIVE)
    finally:
        if tmp.exists():
            tmp.unlink()

    archive_size = ARCHIVE.stat().st_size
    manifest = {
        "schema_version": 1,
        "archive": "financial_assets.zip",
        "archive_sha256": digest(ARCHIVE),
        "archive_bytes": archive_size,
        "file_count": len(files),
        "kpi_company_year_rows": row_count,
        "chroma_collection": COLLECTION,
        "chroma_chunks": indexed,
        "source": "Public SEC EDGAR FY2023-FY2025 Alphabet and Meta filings; generated local assets",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nSTEP 10A DEPLOYMENT ASSETS PREPARED")
    print(f"Warehouse KPI rows: {row_count}/6")
    print(f"Chroma indexed chunks: {indexed:,}/{EXPECTED_COUNT:,}")
    print(f"Packaged files: {len(files)}")
    print(f"Archive size: {archive_size / 1024**2:.2f} MiB")
    print(f"SHA-256: {manifest['archive_sha256']}")
    print("Output: deployment_assets/financial_assets.zip")
    print("Output: deployment_assets/manifest.json")
    if archive_size > LIMIT_MB * 1024**2:
        print("WARNING: archive exceeds 50 MiB. DO NOT COMMIT YET. We'll move it to an external release asset.")
    else:
        print("SIZE CHECK: under 50 MiB; candidate for repository inclusion after security audit.")
    print("No Gemini API requests made.")


if __name__ == "__main__":
    main()
