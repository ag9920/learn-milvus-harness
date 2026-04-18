#!/usr/bin/env python3
"""
S01 - Minimal Loop (Embed -> Insert -> Search)

Lesson Goal
- Build the smallest complete retrieval loop with Milvus Lite.
- Understand the exact role boundaries:
  - embedding model: text -> vector
  - Milvus: vector storage + nearest-neighbor retrieval

Why This Lesson Exists
- Many projects jump directly into filters, rerankers, and abstractions.
- This lesson removes all extras so the base mechanism is fully visible.

What You Should Observe
1. Documents are chunked into short text units.
2. Chunks are embedded and inserted into a collection.
3. A question is embedded and searched against stored vectors.
4. Returned chunks show semantic relevance without any filter/rerank stage.

Run Examples
```bash
./run.sh s01
./run.sh s01 --question "What are AI applications in drug discovery?"
./run.sh s01 --chunk_size 400 --topk 5
./run.sh s01 --input_dir ./your_docs
```

Out-of-Scope for S01
- Metadata filtering
- Reranking
- Multi-stage retrieval quality analysis

Those are introduced incrementally in S02-S05.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient, model
from chapters.sample_corpus import SAMPLE_CORPUS


def read_text(file_path: Path) -> str:
    """Read plain text sources for the minimal lesson."""
    suffix = file_path.suffix.lower()
    if suffix not in {".md", ".txt"}:
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Create fixed-size chunks to keep the retrieval unit simple."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    return [clean[i : i + chunk_size] for i in range(0, len(clean), chunk_size)]


def collect_chunks(input_dir: Path, chunk_size: int) -> list[dict[str, str]]:
    """Scan local files and return normalized chunk records."""
    rows: list[dict[str, str]] = []
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        text = read_text(file_path)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size)
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "text": chunk,
                    "source": str(file_path.relative_to(input_dir)),
                    "chunk_idx": str(idx),
                }
            )
    return rows


def collect_chunks_from_corpus(chunk_size: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for doc in SAMPLE_CORPUS:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size)
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "chunk_idx": str(idx),
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    """Execute the S01 workflow from raw text to top-k retrieval output."""
    print("[S01] step1: Initialize embedding model...")
    embedding_fn: Any = model.DefaultEmbeddingFunction()

    print("[S01] step2: Read files and split chunks...")
    if args.input_dir:
        chunks = collect_chunks(Path(args.input_dir), chunk_size=args.chunk_size)
    else:
        chunks = collect_chunks_from_corpus(chunk_size=args.chunk_size)
    if not chunks:
        raise RuntimeError(f"No indexable .md/.txt files found in {args.input_dir}")
    print(f"[S01] chunks: {len(chunks)}")

    print("[S01] step3: Convert chunks to vectors...")
    vectors = embedding_fn.encode_documents([c["text"] for c in chunks])

    print("[S01] step4: Insert vectors into Milvus Lite...")
    client = MilvusClient(args.db)
    if client.has_collection(collection_name=args.collection):
        client.drop_collection(collection_name=args.collection)
    client.create_collection(collection_name=args.collection, dimension=embedding_fn.dim)

    data = []
    for i, chunk in enumerate(chunks):
        data.append(
            {
                "id": i,
                "vector": vectors[i],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_idx": int(chunk["chunk_idx"]),
            }
        )
    client.insert(collection_name=args.collection, data=data)
    print(f"[S01] inserted: {len(data)}")

    print("[S01] step5: Search with a question vector...")
    query_vector = embedding_fn.encode_queries([args.question])
    res = client.search(
        collection_name=args.collection,
        data=query_vector,
        limit=args.topk,
        output_fields=["text", "source", "chunk_idx"],
    )
    hits = res[0] if res else []

    print(f"\nQuestion: {args.question}")
    for idx, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        text = entity.get("text", "").strip()
        preview = (text[:180] + "...") if len(text) > 180 else text
        print(
            f"[{idx}] score={hit.get('distance', 0):.4f} "
            f"source={entity.get('source')} chunk={entity.get('chunk_idx')}"
        )
        print(f"    {preview}")


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "s01",
        help="S01: minimal loop (embed -> insert -> search).",
        description=(
            "S01 demonstrates the smallest complete Milvus retrieval loop.\n"
            "Use this lesson to build retrieval intuition before filters and reranking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--db", default="milvus_s01.db")
    parser.add_argument("--collection", default="s01_kb")
    parser.add_argument(
        "--question",
        default="How should I balance RAG retrieval quality and latency?",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.set_defaults(handler=run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S01 minimal loop: embed -> insert -> search.\n"
            "Standalone entry for running this lesson file directly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--db", default="milvus_s01.db")
    parser.add_argument("--collection", default="s01_kb")
    parser.add_argument(
        "--question",
        default="How should I balance RAG retrieval quality and latency?",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
