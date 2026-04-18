#!/usr/bin/env python3
"""
S04 - Retrieval Tuning

Lesson Goal
- Tune retrieval parameters with explicit quality-latency tradeoffs.
- Select a stable profile as a baseline for future regression checks.

Core Parameters
- `topk`: result window size
- `candidate_multiplier`: first-stage ANN candidate expansion factor
- `alpha`: rerank blending weight between vector score and lexical overlap

Run Examples
```bash
./run.sh s04
./run.sh s04 --topks 3,5,8 --candidate_multipliers 2,3,5 --alphas 0.6,0.75,0.9
./run.sh s04 --input_dir ./your_docs
```

What This Lesson Prints
- A leaderboard sorted by higher `subject_hit_ratio` then lower `latency_ms`.
- This gives a practical starting point for production-oriented defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from core import (
    index_documents,
    load_embedding_fn,
    parse_float_csv,
    parse_int_csv,
    run_l04_grid,
)
from chapters.sample_corpus import SAMPLE_CORPUS


def run(args: argparse.Namespace) -> None:
    """Sweep retrieval parameters and print a quality-latency leaderboard."""
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

    topks = parse_int_csv(args.topks)
    candidate_multipliers = parse_int_csv(args.candidate_multipliers)
    alphas = parse_float_csv(args.alphas)
    queries = [
        ("How can we improve retrieval recall and latency in RAG?", "ai"),
        ("How is AI used in drug discovery workflows?", "bio"),
    ]
    rows = run_l04_grid(
        client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
        queries=queries,
        topks=topks,
        candidate_multipliers=candidate_multipliers,
        alphas=alphas,
    )
    print("[S04] Leaderboard (higher subject_hit_ratio, lower latency_ms):")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"{i:02d}. topk={row['topk']} "
            f"candidate_multiplier={row['candidate_multiplier']} alpha={row['alpha']:.2f} "
            f"subject_hit_ratio={row['subject_hit_ratio']:.4f} latency_ms={row['latency_ms']:.2f}"
        )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "s04",
        help="S04: sweep retrieval params and print quality-latency leaderboard.",
        description=(
            "S04 performs parameter sweeps for retrieval tuning.\n"
            "Use the leaderboard to choose a baseline profile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="milvus_demo.db")
    parser.add_argument("--collection", default="kb_collection")
    parser.add_argument("--topks", default="3,5")
    parser.add_argument("--candidate_multipliers", default="2,3,5")
    parser.add_argument("--alphas", default="0.6,0.75,0.9")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.set_defaults(handler=run)
