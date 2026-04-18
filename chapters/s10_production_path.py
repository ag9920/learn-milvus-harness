#!/usr/bin/env python3
"""
S10 - Production Path

Lesson Goal
- Provide a practical migration checklist from Milvus Lite to hosted Milvus.
- Run a lightweight release-readiness doctor and smoke check.

Run Examples
```bash
./run.sh s10
./run.sh s10 --question "How can I improve retrieval quality and latency?"
```
"""

from __future__ import annotations

import argparse
import importlib
import platform

from boundaries import HarnessConfig, build_runtime
from core import print_hits


def dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ["pymilvus", "transformers", "pypdf"]:
        mod = importlib.import_module(name)
        out[name] = getattr(mod, "__version__", "unknown")
    return out


def run(args: argparse.Namespace) -> None:
    print("[S10] Doctor report:")
    print(f"- Python: {platform.python_version()}")
    versions = dependency_versions()
    for pkg, ver in versions.items():
        print(f"- {pkg}: {ver}")
    print(f"- Target DB file: {args.db}")
    print(f"- Collection: {args.collection}")

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
    print(f"- Indexed chunks: {stats.total_records}")
    print(f"- Subject distribution: {stats.by_subject}")

    hits = runtime.ask(question=args.question, filter_expr=args.filter)
    print_hits(args.question, hits)

    print("\n[S10] Hosted Milvus migration checklist:")
    print("- Separate local prototyping DB from hosted production cluster config.")
    print("- Define index/search params explicitly and version them with releases.")
    print("- Keep scalar schema stable (`subject/source/ext/chunk_idx`).")
    print("- Store benchmark query sets and compare S09 metrics before deployment.")
    print("- Add runbook for model download/network fallback in restricted regions.")
    print("- Gate release on smoke query success and non-empty retrieval outputs.")


def register_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "s10",
        help="S10: run production-path doctor and migration checklist.",
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
