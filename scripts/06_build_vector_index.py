"""Build a reproducible local Chroma index from six cleaned SEC 10-K texts.

Run from the project root:
    python scripts/06_build_vector_index.py

Rebuild only this project's collection if source texts or settings change:
    python scripts/06_build_vector_index.py --reset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reports" / "sec_filing_manifest.csv"
TEXT_DIR = ROOT / "data" / "processed" / "filings"
CHUNK_DIR = ROOT / "data" / "processed" / "rag_chunks"
CHROMA_DIR = ROOT / "data" / "vectorstore" / "chroma"
AUDIT_PATH = ROOT / "reports" / "vector_index_audit.csv"
COLLECTION_NAME = "sec_10k_minilm_v1"
CHUNK_WORDS = 150
OVERLAP_WORDS = 30
BATCH_SIZE = 64
MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED = {(c, str(y)) for c in ("alphabet", "meta") for y in (2023, 2024, 2025)}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def chunks_for(words: list[str], company: str, year: str, source: dict) -> list[dict]:
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + CHUNK_WORDS, len(words))
        if end <= start:
            break
        index = len(chunks) + 1
        chunks.append(
            {
                "id": f"{company}-fy{year}-ragv1-{index:05d}",
                "document": " ".join(words[start:end]),
                "metadata": {
                    "company": company,
                    "fiscal_year": int(year),
                    "accession": source["accession"],
                    "source_url": source["source_url"],
                    "report_date": source["report_date"],
                    "chunk_number": index,
                    "start_word": start,
                    "end_word": end,
                    "source_type": "SEC_10K",
                },
            }
        )
        if end == len(words):
            break
    return chunks


def prepare_records() -> tuple[list[dict], str, list[dict]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Missing SEC manifest. Complete Step 4 first.")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))

    pairs = [(r["company"], r["fiscal_year"]) for r in manifest]
    if len(manifest) != 6 or set(pairs) != EXPECTED or len(set(pairs)) != 6:
        raise RuntimeError("Expected exactly six distinct company-year SEC filings.")

    inputs = []
    all_records = []
    audits = []
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    for row in sorted(manifest, key=lambda r: (r["company"], r["fiscal_year"])):
        company, year = row["company"], row["fiscal_year"]
        path = TEXT_DIR / f"{company}_FY{year}_clean.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cleaned 10-K: {path}")
        data = path.read_bytes()
        words = data.decode("utf-8-sig").split()
        if len(words) < 1500:
            raise RuntimeError(f"Abnormally short cleaned report: {path}")

        records = chunks_for(words, company, year, row)
        if not records:
            raise RuntimeError(f"No chunks generated for {path}")

        chunk_path = CHUNK_DIR / f"{company}_FY{year}_rag_chunks.jsonl"
        with chunk_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_records.extend(records)
        inputs.append(f"{company}|{year}|{sha256(data)}|{row['accession']}")
        audits.append(
            {
                "company": company,
                "fiscal_year": year,
                "cleaned_words": len(words),
                "chunk_count": len(records),
                "chunk_words": CHUNK_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "cleaned_text_sha256": sha256(data),
                "source_accession": row["accession"],
                "status": "PASS",
            }
        )
        print(f"{company.upper()} FY{year}: {len(words):,} words -> {len(records):,} chunks", flush=True)

    fingerprint = sha256(
        ("\n".join(inputs) + f"\n{CHUNK_WORDS}/{OVERLAP_WORDS}/{MODEL_NAME}").encode("utf-8")
    )
    if len({r["id"] for r in all_records}) != len(all_records):
        raise RuntimeError("Duplicate chunk IDs detected.")
    return all_records, fingerprint, audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="Delete and recreate only this project's Chroma collection."
    )
    args = parser.parse_args()

    records, fingerprint, audits = prepare_records()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_names = {c.name for c in client.list_collections()}
    if args.reset and COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
        existing_names.remove(COLLECTION_NAME)
        print(f"Rebuilding collection: {COLLECTION_NAME}")

    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(COLLECTION_NAME)
        meta = collection.metadata or {}
        if (
            meta.get("input_fingerprint") != fingerprint
            or meta.get("embedding_model") != MODEL_NAME
            or meta.get("chunk_words") != CHUNK_WORDS
            or meta.get("overlap_words") != OVERLAP_WORDS
        ):
            raise RuntimeError(
                "Existing index settings or source documents changed. "
                "Run: python scripts/06_build_vector_index.py --reset"
            )
    else:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "input_fingerprint": fingerprint,
                "embedding_model": MODEL_NAME,
                "chunk_words": CHUNK_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "schema_version": "1",
            },
        )

    existing_ids = set(collection.get(include=[])["ids"])
    expected_ids = {r["id"] for r in records}
    if unexpected := (existing_ids - expected_ids):
        raise RuntimeError(f"Unexpected IDs in collection ({len(unexpected)}). Use --reset.")

    pending = [r for r in records if r["id"] not in existing_ids]
    print(f"\nIndexing {len(pending):,} new chunks out of {len(records):,} total.", flush=True)
    print("First run may download the local embedding model.", flush=True)
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        collection.add(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
        )
        print(
            f"  Added {min(start + len(batch), len(pending)):,}/{len(pending):,} new chunks",
            flush=True,
        )

    count = collection.count()
    if count != len(records):
        raise RuntimeError(f"Index count mismatch: expected {len(records)}, got {count}")

    counts = Counter((r["metadata"]["company"], str(r["metadata"]["fiscal_year"])) for r in records)
    for row in audits:
        key = (row["company"], row["fiscal_year"])
        actual = len(
            collection.get(
                where={"$and": [{"company": key[0]}, {"fiscal_year": int(key[1])}]},
                include=[],
            )["ids"]
        )
        if counts[key] != actual:
            raise RuntimeError(f"Company-year count mismatch: {key}, {counts[key]} != {actual}")
        row["indexed_count"] = actual

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    print("\nVECTOR INDEX BUILD COMPLETE")
    print("Reports indexed: 6/6")
    print(f"Chunks indexed: {count:,}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Input fingerprint: {fingerprint[:16]}...")
    print("Audit: reports/vector_index_audit.csv")


if __name__ == "__main__":
    main()
