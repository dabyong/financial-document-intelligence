"""Smoke-test semantic search and company/year metadata filters in Chroma.

Run from the project root after Step 7A indexing:
    python scripts/07_test_retrieval.py
"""

from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "sec_10k_minilm_v1"

TEST_QUERIES = [
    ("alphabet", 2025, "What factors influenced Alphabet's advertising and cloud revenue growth?"),
    ("meta", 2025, "What risks did Meta disclose about AI infrastructure and capital spending?"),
    ("meta", 2023, "What privacy and regulatory risks did Meta describe in its annual report?"),
]


def main() -> None:
    if not CHROMA_DIR.is_dir():
        raise SystemExit("Missing vector database. Run 06_build_vector_index.py first.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise SystemExit("Missing SEC collection. Run 06_build_vector_index.py first.") from exc

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed chunks: {collection.count():,}\n")

    for company, year, question in TEST_QUERIES:
        print("=" * 80)
        print(f"QUESTION: {question}")
        print(f"FILTER: company={company}, fiscal_year={year}")

        result = collection.query(
            query_texts=[question],
            n_results=3,
            where={"$and": [{"company": company}, {"fiscal_year": year}]},
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        metas = result["metadatas"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        if not ids:
            raise RuntimeError(f"No retrieved chunks for {company} FY{year}")

        for rank, (chunk_id, meta, doc, distance) in enumerate(
            zip(ids, metas, docs, distances), start=1
        ):
            if meta["company"] != company or meta["fiscal_year"] != year:
                raise RuntimeError(f"Cross-company/year leakage: {chunk_id}")
            preview = doc[:280].replace("\n", " ")
            print(f"\n  Rank {rank} | {chunk_id} | cosine distance {distance:.4f}")
            print(f"  Source: {meta['source_url']}")
            print(f"  Passage: {preview}...")

        print("\n  FILTER CHECK: PASSED")
        print("  RELEVANCE CHECK: Inspect passages manually; not graded yet.\n")

    print("RETRIEVAL SMOKE TEST COMPLETE")
    print("All company/year metadata filters: PASSED")
    print("Semantic relevance evaluation: PENDING")


if __name__ == "__main__":
    main()
