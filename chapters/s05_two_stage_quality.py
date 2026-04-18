#!/usr/bin/env python3
"""
S05 - Two-Stage Quality Delta

Lesson Goal
- Measure whether reranking actually improves relevance on your benchmark queries.
- Compare first-stage vector ranking against two-stage ranking under fixed conditions.

Comparison Target
- Vector-only: rank by ANN vector score directly.
- Two-stage: keep ANN recall candidates, then rerank with blended score.

Run Examples
```bash
./run.sh s05
./run.sh s05 --topk 5 --candidate_multiplier 3 --alpha 0.75
./run.sh s05 --input_dir ./your_docs
```

What This Lesson Prints
- Query-level quality delta (`rerank - vector_only`)
- Aggregate delta across benchmark queries

Interpretation
- Positive delta: rerank adds signal.
- Near zero: rerank may be redundant.
- Negative delta: rerank likely hurts relevance and should be revised.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from core import (
    index_documents,
    load_embedding_fn,
    search_candidates,
    topk_reranked,
    topk_vector_only,
)
from chapters.sample_corpus import SAMPLE_CORPUS


def _subject_hit_ratio(hits: list[dict], expected_subject: str) -> float:
    """Proxy quality metric: subject-aligned hit ratio in returned top-k."""
    if not hits:
        return 0.0
    hit_count = sum(
        1 for h in hits if h.get("entity", {}).get("subject") == expected_subject
    )
    return hit_count / len(hits)


def run(args: argparse.Namespace) -> None:
    """Run vector-only vs reranked quality comparison on fixed benchmark queries."""
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

    queries = [
        ("How can we improve retrieval recall and latency in RAG?", "ai"),
        ("How is AI used in drug discovery workflows?", "bio"),
    ]
    vector_scores: list[float] = []
    rerank_scores: list[float] = []

    print("[S05] Query-level comparison:")
    for question, expected_subject in queries:
        candidates = search_candidates(
            client=client,
            collection_name=args.collection,
            embedding_fn=embedding_fn,
            question=question,
            topk=args.topk,
            filter_expr=None,
            candidate_multiplier=args.candidate_multiplier,
        )
        vector_hits = topk_vector_only(candidates, topk=args.topk)
        rerank_hits = topk_reranked(
            question=question,
            hits=candidates,
            topk=args.topk,
            alpha=args.alpha,
        )
        vector_ratio = _subject_hit_ratio(vector_hits, expected_subject)
        rerank_ratio = _subject_hit_ratio(rerank_hits, expected_subject)
        vector_scores.append(vector_ratio)
        rerank_scores.append(rerank_ratio)
        print(
            f"- expected_subject={expected_subject} "
            f"vector_only={vector_ratio:.4f} rerank={rerank_ratio:.4f} "
            f"delta={rerank_ratio - vector_ratio:+.4f}"
        )

    vector_avg = sum(vector_scores) / max(len(vector_scores), 1)
    rerank_avg = sum(rerank_scores) / max(len(rerank_scores), 1)
    print("\n[S05] Aggregate:")
    print(f"vector_only_avg={vector_avg:.4f}")
    print(f"rerank_avg={rerank_avg:.4f}")
    print(f"delta={rerank_avg - vector_avg:+.4f}")


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "s05",
        help="S05: measure vector-only vs reranked quality delta.",
        description=(
            "S05 compares vector-only ranking with two-stage reranking.\n"
            "Use this to validate that reranking gives measurable gains."
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
    parser.set_defaults(handler=run)
