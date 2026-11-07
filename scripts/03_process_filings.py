import csv
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "sec_filing_manifest.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 75

EXPECTED = {
    (company, str(year))
    for company in ("alphabet", "meta")
    for year in (2023, 2024, 2025)
}


def clean_filing(raw):
    """Extract text using the appropriate parser."""
    prefix = (
        raw[:512]
        .decode("utf-8", errors="ignore")
        .lstrip("\ufeff \t\r\n")
        .lower()
    )

    parser = "xml" if prefix.startswith("<?xml") else "lxml"
    soup = BeautifulSoup(raw, parser)

    # Exclude non-visible application and XBRL metadata.
    for tag in soup.find_all(True):
        if not tag.name:
            continue

        name = tag.name.lower()
        ix_metadata = (
            name in {"ix:header", "ix:hidden"}
            or (
                tag.prefix == "ix"
                and name in {"header", "hidden"}
            )
        )

        if name in {"script", "style", "noscript"} or ix_metadata:
            tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text, parser


def build_chunks(text, metadata):
    """Create overlapping, source-traceable chunks."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        end = min(start + CHUNK_SIZE, len(words))

        chunks.append({
            "chunk_id": (
                f"{metadata['company']}-"
                f"fy{metadata['fiscal_year']}-"
                f"{len(chunks) + 1:05d}"
            ),
            "company": metadata["company"],
            "fiscal_year": int(metadata["fiscal_year"]),
            "accession": metadata["accession"],
            "source_url": metadata["source_url"],
            "start_word": start,
            "end_word": end,
            "text": " ".join(words[start:end]),
        })

        if end == len(words):
            break

    return chunks


def process():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "Run Step 4 data acquisition first."
        )

    with MANIFEST.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    found = {
        (row["company"], row["fiscal_year"])
        for row in rows
    }

    if len(rows) != 6 or found != EXPECTED:
        raise RuntimeError(
            "Manifest must contain exactly six unique "
            "company-year reports."
        )

    text_dir = PROCESSED / "filings"
    chunk_dir = PROCESSED / "chunks"
    text_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    audit = []
    total_chunks = 0

    for row in sorted(
        rows, key=lambda r: (r["company"], r["fiscal_year"])
    ):
        company = row["company"]
        year = row["fiscal_year"]

        source = (ROOT / row["local_path"]).resolve()
        raw_root = (ROOT / "data" / "raw").resolve()

        if not source.is_relative_to(raw_root):
            raise RuntimeError("Invalid source file path.")

        raw = source.read_bytes()

        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch: {company} FY{year}"
            )

        text, parser = clean_filing(raw)
        word_count = len(text.split())

        if len(text) < 10000 or word_count < 1500:
            raise RuntimeError(
                f"Insufficient extracted text: "
                f"{company} FY{year}"
            )

        # Flag possible major extraction loss.
        previous_length = int(row["text_characters"])
        if len(text) < previous_length * 0.5:
            raise RuntimeError(
                f"Possible text loss: {company} FY{year}"
            )

        chunks = build_chunks(text, row)

        stem = f"{company}_FY{year}"

        (text_dir / f"{stem}_clean.txt").write_text(
            text + "\n", encoding="utf-8"
        )

        chunk_path = chunk_dir / f"{stem}_chunks.jsonl"
        with chunk_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(chunk, ensure_ascii=False)
                    + "\n"
                )

        # Candidate section mentions, not validated boundaries.
        section_mentions = [
            label
            for label, pattern in {
                "Item 1A": r"\bitem\s+1a\b",
                "Item 7": r"\bitem\s+7\b",
                "Item 8": r"\bitem\s+8\b",
            }.items()
            if re.search(pattern, text, re.IGNORECASE)
        ]

        audit.append({
            "company": company,
            "fiscal_year": year,
            "parser": parser,
            "text_characters": len(text),
            "word_count": word_count,
            "chunk_count": len(chunks),
            "section_mentions": "; ".join(section_mentions),
            "checksum_verified": True,
            "status": "PASS",
        })

        total_chunks += len(chunks)

        print(
            f"{company.upper()} FY{year}: PASS | "
            f"parser={parser} | "
            f"words={word_count:,} | "
            f"chunks={len(chunks):,}"
        )

    audit_path = REPORTS / "document_processing_audit.csv"

    with audit_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(audit[0])
        )
        writer.writeheader()
        writer.writerows(audit)

    print("\nDOCUMENT PROCESSING COMPLETE")
    print(f"Reports processed: {len(audit)}/6")
    print(f"Total chunks: {total_chunks:,}")
    print("Checksum verification: PASSED")
    print("Audit: reports/document_processing_audit.csv")


if __name__ == "__main__":
    process()
