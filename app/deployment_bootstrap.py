"""Restore read-only, prebuilt SEC research assets on a new cloud instance.

This module does not download data or access credentials. It only reads the
checksummed ZIP packaged by scripts/13_package_deployment_assets.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


DB_REL = Path("data/warehouse/financial_intelligence.duckdb")
CHROMA_REL = Path("data/vectorstore/chroma/chroma.sqlite3")
ALLOWED_PREFIXES = ("data/warehouse/", "data/vectorstore/chroma/")


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


def ensure_assets(root: Path) -> str:
    root = root.resolve()
    db = root / DB_REL
    chroma_sqlite = root / CHROMA_REL
    if db.is_file() and chroma_sqlite.is_file():
        return "Existing local DuckDB and Chroma assets"

    assets_dir = root / "deployment_assets"
    archive = assets_dir / "financial_assets.zip"
    manifest_path = assets_dir / "manifest.json"
    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Financial dataset is missing. For cloud deployment, include "
            "deployment_assets/financial_assets.zip and manifest.json. "
            "For local development, run Steps 5B and 7A first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.get("archive_sha256", ""))
    if len(expected) != 64 or sha256_file(archive) != expected:
        raise ValueError("Deployment ZIP checksum mismatch; refusing to load the data")
    if manifest.get("kpi_company_year_rows") != 6 or manifest.get("chroma_chunks") != 3284:
        raise ValueError("Unexpected deployment asset manifest counts")

    # Validate the ZIP *before* creating any files under the project root.
    with zipfile.ZipFile(archive, "r") as z:
        names = [info.filename for info in z.infolist() if not info.is_dir()]
        if not names or len(names) != len(set(names)):
            raise ValueError("Empty archive or duplicate archive members")
        expected_db = DB_REL.as_posix()
        expected_chroma = CHROMA_REL.as_posix()
        if expected_db not in names or expected_chroma not in names:
            raise ValueError("Missing required DuckDB or Chroma files in archive")
        for name in names:
            pure = PurePosixPath(name)
            if (pure.is_absolute() or ".." in pure.parts or "\\" in name
                    or not name.startswith(ALLOWED_PREFIXES)):
                raise ValueError(f"Unsafe ZIP entry: {name}")

        # Extract into a separate temporary tree and move completed assets into place.
        # This avoids leaving half-extracted files on routine deployment failures.
        with tempfile.TemporaryDirectory(prefix="p5_assets_") as temp:
            staging = Path(temp)
            for info in z.infolist():
                if info.is_dir():
                    continue
                destination = staging.joinpath(*PurePosixPath(info.filename).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, destination.open("wb") as dest:
                    shutil.copyfileobj(src, dest)

            staged_db = staging / DB_REL
            staged_chroma = staging / "data/vectorstore/chroma"
            if not staged_db.is_file() or not (staged_chroma / "chroma.sqlite3").is_file():
                raise ValueError("Extracted archive is missing required files")
            db.parent.mkdir(parents=True, exist_ok=True)
            chroma_sqlite.parent.parent.mkdir(parents=True, exist_ok=True)
            # The cloud instance does not have these assets yet; local assets were
            # explicitly respected by the first if-check above.
            if not db.exists():
                shutil.copy2(staged_db, db)
            chroma_target = root / "data/vectorstore/chroma"
            if not chroma_target.exists():
                shutil.copytree(staged_chroma, chroma_target)
            elif not chroma_sqlite.exists():
                # A partial prior restore should not silently pass the readiness check.
                shutil.rmtree(chroma_target)
                shutil.copytree(staged_chroma, chroma_target)

    if not db.is_file() or not chroma_sqlite.is_file():
        raise RuntimeError("Deployment assets failed to restore")
    return "Restored checksum-verified public SEC dataset"
