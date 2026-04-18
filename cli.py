from __future__ import annotations

import argparse
from typing import Callable

from boundaries import HarnessConfig, build_runtime
from core import print_hits
from chapters import (
    s01_minimal_loop,
    s02_chunking_lab,
    s03_schema_filter,
    s04_retrieval_tuning,
    s05_two_stage_quality,
    s06_failure_analysis,
    s07_boundary_refactor,
    s08_pluggable_components,
    s09_evaluation_harness,
    s10_production_path,
)


Handler = Callable[[argparse.Namespace], None]


def _run_final_index(args: argparse.Namespace) -> None:
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
    stats = runtime.index(
        rebuild=args.rebuild,
    )
    _print_ingest_summary(stats.total_records, stats.by_subject, stats.by_ext)


def _run_final_ask(args: argparse.Namespace) -> None:
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
    hits = runtime.ask(question=args.question, filter_expr=args.filter)
    print_hits(args.question, hits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Milvus RAG harness CLI.\n\n"
            "Command groups:\n"
            "  final  - production-style entrypoints (index/ask/demo)\n"
            "  study  - chapter-by-chapter learning entrypoints (s01-s10)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="group", required=True)

    p_final = subparsers.add_parser(
        "final",
        help="Final system commands: index, ask, demo.",
    )
    final_subparsers = p_final.add_subparsers(dest="final_cmd", required=True)

    p_index = final_subparsers.add_parser("index", help="Ingest docs into Milvus.")
    p_index.add_argument("--db", default="milvus_demo.db")
    p_index.add_argument("--collection", default="kb_collection")
    p_index.add_argument("--input_dir", default=None)
    p_index.add_argument("--rebuild", action="store_true")
    p_index.add_argument("--chunk_size", type=int, default=500)
    p_index.add_argument("--overlap", type=int, default=80)
    p_index.add_argument("--topk", type=int, default=5)
    p_index.add_argument("--candidate_multiplier", type=int, default=3)
    p_index.add_argument("--alpha", type=float, default=0.75)
    p_index.set_defaults(handler=_run_final_index)

    p_ask = final_subparsers.add_parser(
        "ask",
        help="Search with optional filter and rerank.",
    )
    p_ask.add_argument("--db", default="milvus_demo.db")
    p_ask.add_argument("--collection", default="kb_collection")
    p_ask.add_argument("--input_dir", default=None)
    p_ask.add_argument("--chunk_size", type=int, default=500)
    p_ask.add_argument("--overlap", type=int, default=80)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--topk", type=int, default=5)
    p_ask.add_argument("--filter", default=None)
    p_ask.add_argument("--candidate_multiplier", type=int, default=3)
    p_ask.add_argument("--alpha", type=float, default=0.75)
    p_ask.set_defaults(handler=_run_final_ask)

    p_demo = final_subparsers.add_parser(
        "demo",
        help="Build index then run two fixed queries.",
    )
    p_demo.add_argument("--db", default="milvus_demo.db")
    p_demo.add_argument("--collection", default="kb_collection")
    p_demo.add_argument("--input_dir", default=None)
    p_demo.add_argument("--rebuild", action="store_true")
    p_demo.add_argument("--chunk_size", type=int, default=500)
    p_demo.add_argument("--overlap", type=int, default=80)
    p_demo.add_argument("--topk", type=int, default=3)
    p_demo.add_argument("--candidate_multiplier", type=int, default=3)
    p_demo.add_argument("--alpha", type=float, default=0.75)
    p_demo.set_defaults(handler=_run_demo)

    p_study = subparsers.add_parser(
        "study",
        help="Study chapters: s01-s10.",
    )
    study_subparsers = p_study.add_subparsers(dest="study_cmd", required=True)

    s01_minimal_loop.register_subparser(study_subparsers)
    s02_chunking_lab.register_subparser(study_subparsers)
    s03_schema_filter.register_subparser(study_subparsers)
    s04_retrieval_tuning.register_subparser(study_subparsers)
    s05_two_stage_quality.register_subparser(study_subparsers)
    s06_failure_analysis.register_subparser(study_subparsers)
    s07_boundary_refactor.register_subparser(study_subparsers)
    s08_pluggable_components.register_subparser(study_subparsers)
    s09_evaluation_harness.register_subparser(study_subparsers)
    s10_production_path.register_subparser(study_subparsers)

    return parser.parse_args()


def _print_ingest_summary(
    total_records: int, by_subject: dict[str, int], by_ext: dict[str, int]
) -> None:
    print(f"Indexed chunks: {total_records}")
    print(f"By subject: {by_subject}")
    print(f"By extension: {by_ext}")


def _run_demo(args: argparse.Namespace) -> None:
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
    stats = runtime.index(
        rebuild=args.rebuild,
    )
    _print_ingest_summary(stats.total_records, stats.by_subject, stats.by_ext)

    q1 = "How can we improve retrieval recall while keeping latency low in RAG?"
    q2 = "What are AI applications in drug discovery and molecular modeling?"
    hits1 = runtime.ask(question=q1, filter_expr="subject == 'ai'")
    hits2 = runtime.ask(question=q2, filter_expr="subject == 'bio'")
    print_hits(q1, hits1)
    print_hits(q2, hits2)


def main() -> None:
    args = parse_args()
    if not hasattr(args, "handler"):
        raise RuntimeError("Missing command handler. Use --help to inspect CLI groups.")
    handler: Handler = args.handler
    handler(args)


if __name__ == "__main__":
    main()
