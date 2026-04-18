#!/usr/bin/env python3
"""
S07 - Boundary Refactor

Lesson Goal
- Demonstrate cleaner interfaces between CLI and retrieval runtime.
- Keep orchestration code small by delegating to a dedicated boundary layer.

Run Examples
```bash
./run.sh s07
./run.sh s07 --question "How can I improve retrieval quality and latency?"
```

What This Lesson Shows
- A typed runtime config (`HarnessConfig`)
- A boundary service (`HarnessRuntime`) with explicit methods:
  - `index(rebuild=...)`
  - `ask(question=..., filter_expr=...)`
"""

from __future__ import annotations

import argparse

from boundaries import HarnessConfig, build_runtime
from core import print_hits


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
    print(f"By extension: {stats.by_ext}")

    hits = runtime.ask(question=args.question, filter_expr=args.filter)
    print_hits(args.question, hits)

    print("\n[S07] Boundary summary:")
    print("- CLI layer: argument parsing and command routing")
    print("- Boundary layer: runtime construction and service methods")
    print("- Core layer: retrieval/indexing primitives")


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "s07",
        help="S07: run retrieval via explicit boundary/service interfaces.",
        description=(
            "S07 illustrates cleaner module boundaries by using HarnessConfig + "
            "HarnessRuntime service methods."
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
    parser.add_argument(
        "--question",
        default="How can I improve retrieval quality and latency?",
    )
    parser.add_argument("--filter", default=None)
    parser.set_defaults(handler=run)
