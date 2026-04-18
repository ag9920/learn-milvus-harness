#!/usr/bin/env python3
"""
S03 - Schema and Filter Audit

Lesson Goal
- Enforce metadata discipline before tuning retrieval quality.
- Verify scalar filtering behavior is stable and explainable.

Why This Matters
- Retrieval quality is not only vectors; metadata quality is equally critical.
- Weak or inconsistent scalar fields make filter behavior brittle and hard to debug.

Run Examples
```bash
./run.sh s03
./run.sh s03 --chunk_size 500 --overlap 80 --smoke_question "How can I improve retrieval quality and latency?"
./run.sh s03 --input_dir ./your_docs
```

What This Lesson Checks
1. Required fields exist: `id, vector, text, source, subject, ext, chunk_idx`
2. Subject/extension distributions are observable.
3. Subject filters return stable hit counts for smoke queries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from core import (
    SUPPORTED_EXTENSIONS,
    ask,
    index_documents,
    load_embedding_fn,
    prepare_records,
)
from chapters.sample_corpus import SAMPLE_CORPUS


def run(args: argparse.Namespace) -> None:
    """Run schema validation and scalar-filter smoke tests."""
    client = MilvusClient(args.db)
    embedding_fn = load_embedding_fn()

    input_dir = Path(args.input_dir) if args.input_dir else None
    documents = None if input_dir is not None else SAMPLE_CORPUS
    stats = index_documents(
        client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
        input_dir=input_dir,
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        documents=documents,
    )
    print(f"Indexed chunks: {stats.total_records}")
    print(f"By subject: {stats.by_subject}")
    print(f"By extension: {stats.by_ext}")

    records, _ = prepare_records(
        input_dir=input_dir,
        embedding_fn=embedding_fn,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        documents=documents,
    )
    required_fields = {"id", "vector", "text", "source", "subject", "ext", "chunk_idx"}
    missing = 0
    for row in records:
        if not required_fields.issubset(row.keys()):
            missing += 1
    print(f"[S03] schema_required_fields={sorted(required_fields)}")
    print(f"[S03] rows_missing_required_fields={missing}")
    print(f"[S03] supported_extensions={sorted(SUPPORTED_EXTENSIONS)}")

    print("[S03] filter smoke tests:")
    for subject in sorted(stats.by_subject):
        hits = ask(
            client=client,
            collection_name=args.collection,
            embedding_fn=embedding_fn,
            question=args.smoke_question,
            topk=args.topk,
            filter_expr=f"subject == '{subject}'",
            candidate_multiplier=args.candidate_multiplier,
            alpha=args.alpha,
        )
        print(f"  subject={subject} hits={len(hits)}")


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "s03",
        help="S03: validate metadata/schema assumptions and run filter smoke tests.",
        description=(
            "S03 validates metadata shape and scalar filter behavior.\n"
            "Use this before deeper retrieval tuning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="milvus_demo.db")
    parser.add_argument("--collection", default="kb_collection")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument(
        "--smoke_question",
        default="How can I improve retrieval quality and latency?",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--candidate_multiplier", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.set_defaults(handler=run)
