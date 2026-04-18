#!/usr/bin/env python3
"""
S06 - Failure Analysis

Lesson Goal
- Explain why retrieval misses happen.
- Build a repeatable taxonomy for wrong hits and empty results.

Run Examples
```bash
./run.sh s06
./run.sh s06 --topk 5 --candidate_multiplier 5 --alpha 0.8
./run.sh s06 --input_dir ./your_docs
```

What This Lesson Prints
- Query-level diagnosis (`ok` / `miss`) with reason tags.
- Top hit explanations including subject mismatch, low vector score, and low lexical overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from chapters.sample_corpus import SAMPLE_CORPUS
from core import diagnose_query, index_documents, load_embedding_fn


def run(args: argparse.Namespace) -> None:
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

    scenarios = [
        (
            "How can we improve retrieval recall and latency in RAG?",
            "ai",
            None,
        ),
        (
            "How is AI used in drug discovery workflows?",
            "bio",
            None,
        ),
        (
            "How is AI used in drug discovery workflows?",
            "bio",
            "subject == 'ai'",
        ),
    ]

    print("[S06] Failure diagnostics:")
    for question, expected_subject, filter_expr in scenarios:
        report = diagnose_query(
            client=client,
            collection_name=args.collection,
            embedding_fn=embedding_fn,
            question=question,
            topk=args.topk,
            candidate_multiplier=args.candidate_multiplier,
            alpha=args.alpha,
            expected_subject=expected_subject,
            filter_expr=filter_expr,
        )
        print(
            f"- status={report['status']} reason={report['reason']} "
            f"expected_subject={expected_subject} filter={filter_expr}"
        )
        for ex in report["explanations"][: args.show]:
            print(
                f"  rank={ex['rank']} subject={ex['subject']} "
                f"vector={ex['vector_score']:.4f} rerank={ex['rerank_score']:.4f} "
                f"overlap={ex['lexical_overlap']:.4f} reasons={','.join(ex['reasons'])}"
            )


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "s06",
        help="S06: diagnose misses and wrong hits with explainable reasons.",
        description=(
            "S06 runs failure analysis on fixed scenarios and prints reason tags.\n"
            "Use this chapter to debug retrieval behavior before tuning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="milvus_demo.db")
    parser.add_argument("--collection", default="kb_collection")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--candidate_multiplier", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--show", type=int, default=3)
    parser.set_defaults(handler=run)
