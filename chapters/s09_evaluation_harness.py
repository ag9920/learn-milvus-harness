#!/usr/bin/env python3
"""
S09 - Evaluation Harness

Lesson Goal
- Build a reproducible offline evaluation loop for retrieval changes.
- Track Recall@K, MRR, and nDCG with fixed benchmark queries.

Run Examples
```bash
./run.sh s09
./run.sh s09 --topk 5 --alpha 0.8
```
"""

from __future__ import annotations

import argparse
import math
from statistics import mean

from boundaries import HarnessConfig, build_runtime


def recall_at_k(relevant_ranks: list[int], k: int) -> float:
    return 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0


def mrr(relevant_ranks: list[int]) -> float:
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def ndcg_at_k(relevant_ranks: list[int], k: int) -> float:
    # Binary relevance; ideal DCG is always 1.0 when at least one relevant doc exists.
    gains = [1.0 / math.log2(rank + 1.0) for rank in relevant_ranks if rank <= k]
    return max(gains) if gains else 0.0


def run(args: argparse.Namespace) -> None:
    runtime = build_runtime(
        HarnessConfig(
            db=args.db,
            collection=args.collection,
            input_dir=args.input_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            topk=args.topk,
            candidate_multiplier=args.candidate_multiplier,
            alpha=args.alpha,
        )
    )
    stats = runtime.index(rebuild=args.rebuild)
    print(f"Indexed chunks: {stats.total_records}")
    print(f"By subject: {stats.by_subject}")

    benchmark = [
        ("How can we improve retrieval recall and latency in RAG?", "ai"),
        ("How is AI used in drug discovery workflows?", "bio"),
        ("How should chunk overlap be tuned for retrieval?", "ai"),
        ("What is ADMET modeling used for?", "bio"),
    ]

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    print("\n[S09] Query-level metrics:")
    for question, expected_subject in benchmark:
        hits = runtime.ask(question=question, filter_expr=None)
        relevant_ranks = [
            idx
            for idx, hit in enumerate(hits, start=1)
            if hit.get("entity", {}).get("subject") == expected_subject
        ]
        r = recall_at_k(relevant_ranks, args.topk)
        rr = mrr(relevant_ranks)
        n = ndcg_at_k(relevant_ranks, args.topk)
        recalls.append(r)
        mrrs.append(rr)
        ndcgs.append(n)
        print(
            f"- expected={expected_subject} recall@{args.topk}={r:.3f} "
            f"mrr={rr:.3f} ndcg@{args.topk}={n:.3f}"
        )

    print("\n[S09] Aggregate:")
    print(f"Recall@{args.topk}: {mean(recalls):.3f}")
    print(f"MRR: {mean(mrrs):.3f}")
    print(f"nDCG@{args.topk}: {mean(ndcgs):.3f}")


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "s09",
        help="S09: run offline retrieval evaluation (Recall@K/MRR/nDCG).",
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
