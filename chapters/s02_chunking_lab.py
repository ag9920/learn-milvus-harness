#!/usr/bin/env python3
"""
S02 - Chunking Lab

Lesson Goal
- Quantify how chunk size and overlap influence retrieval behavior.
- Produce a first-pass parameter shortlist for your corpus.

Core Idea
- Chunking defines the atomic retrieval unit.
- If chunks are too short, context fragments and relevance becomes noisy.
- If chunks are too long, semantic precision drops and downstream context cost rises.

Run Examples
```bash
./run.sh s02 --question "How can I improve retrieval quality and latency?"
./run.sh s02 --question "How can I improve retrieval quality and latency?" --chunk_sizes 256,512,768 --overlaps 40,80,120
./run.sh s02 --question "How can I improve retrieval quality and latency?" --input_dir ./your_docs
```

What This Lesson Prints
- Indexed chunk count for each configuration.
- Top hit subject and score for quick sensitivity comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from core import ask, index_documents, load_embedding_fn, parse_int_csv
from chapters.sample_corpus import SAMPLE_CORPUS


def run(args: argparse.Namespace) -> None:
    """Run the chunking experiment grid and report top-hit behavior."""
    client = MilvusClient(args.db)
    embedding_fn = load_embedding_fn()
    chunk_sizes = parse_int_csv(args.chunk_sizes)
    overlaps = parse_int_csv(args.overlaps)

    input_dir = Path(args.input_dir) if args.input_dir else None
    documents = None if input_dir is not None else SAMPLE_CORPUS
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            collection = f"{args.collection}_s02_c{chunk_size}_o{overlap}".replace("-", "_")
            stats = index_documents(
                client=client,
                collection_name=collection,
                embedding_fn=embedding_fn,
                input_dir=input_dir,
                rebuild=True,
                chunk_size=chunk_size,
                overlap=overlap,
                documents=documents,
            )
            hits = ask(
                client=client,
                collection_name=collection,
                embedding_fn=embedding_fn,
                question=args.question,
                topk=args.topk,
                filter_expr=None,
                candidate_multiplier=args.candidate_multiplier,
                alpha=args.alpha,
            )
            top_subject = hits[0].get("entity", {}).get("subject") if hits else None
            top_score = hits[0].get("rerank_score", 0.0) if hits else 0.0
            print(
                f"[S02] chunk_size={chunk_size} overlap={overlap} "
                f"chunks={stats.total_records} top_subject={top_subject} top_score={top_score:.4f}"
            )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "s02",
        help="S02: run chunk-size/overlap experiments and compare retrieval outcomes.",
        description=(
            "S02 explores chunking sensitivity.\n"
            "Use multiple chunk_size/overlap combinations and compare retrieval outcomes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="milvus_demo.db")
    parser.add_argument("--collection", default="kb_collection")
    parser.add_argument("--question", required=True)
    parser.add_argument("--chunk_sizes", default="256,512,768")
    parser.add_argument("--overlaps", default="40,80")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--candidate_multiplier", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.set_defaults(handler=run)
