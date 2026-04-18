# learn-milvus-harness

A practical Milvus-first RAG harness, built from first principles.

This repository is designed for serious builders who want to learn Milvus deeply, not just run a toy demo.

Core loop:
`Embed -> Index -> Retrieve -> Filter -> Rerank -> Evaluate`

## Why This Repo

Most tutorials stop at API calls. They do not explain why retrieval quality drifts, where latency comes from, or how to evolve a retriever safely.

This repo focuses on production-relevant retrieval engineering:

- Retrieval as a system, not a prompt trick
- Milvus schema and scalar filtering discipline
- Two-stage retrieval (ANN recall + rerank)
- Evaluation-first iteration (quality and latency)

## What You Will Build

You will build a two-track CLI:

- `final`: production-style commands (`index`, `ask`, `demo`)
- `study`: chapter-by-chapter commands (`s01` ~ `s10`)

## Roadmap (S01 ~ S10)

Goal: build a complete, high-quality retrieval system with Milvus, not a one-off demo.

- `S01` Minimal loop: `embed -> insert -> search`
- `S02` Chunking strategy: chunk size/overlap experiments
- `S03` Schema/filter discipline: scalar metadata audits
- `S04` Retrieval tuning: `topk/candidate_multiplier/alpha` tradeoffs
- `S05` Two-stage quality delta: vector-only vs reranked comparisons
- `S06` Failure analysis: explain misses and wrong hits
- `S07` Boundary refactor: cleaner module interfaces
- `S08` Pluggable components: extensibility without loop rewrites
- `S09` Evaluation harness: `Recall@K/MRR/nDCG` regression safety
- `S10` Production path: Milvus Lite -> hosted deployment checklist

## Quality Bar

Every retrieval change should satisfy these constraints:

- Include evidence for both quality and latency impact.
- Keep retrieval behavior explainable from source-level diagnostics.
- Keep metadata/schema decisions auditable (`subject/source/ext/chunk_idx` discipline).
- Preserve reproducibility (same data + same config => comparable outcomes).
- Maintain a clear migration path from Milvus Lite to hosted Milvus.
- Keep chunking parameters valid: `chunk_size > 0` and `0 <= overlap < chunk_size`.

Lesson authoring rule:
- The chapter Python files in `chapters/` are the primary teaching source.
- Keep explanations and runnable code together in the same chapter file.

## CLI Structure

- `final index|ask|demo`: final system workflow
- `study s01|s02|s03|s04|s05|s06|s07|s08|s09|s10`: independent chapter execution
- `run.sh` provides shortcuts for both tracks

## Quick Start

```bash
# 1) install deps
./run.sh setup
```

`run.sh` does not auto-install dependencies on every command.
If dependencies are missing, it will prompt you to run `./run.sh setup`.

### 5-Minute Study Path

```bash
# Start from the minimal chapter
./run.sh s01

# Continue with chapter experiments
./run.sh s02 --question "How can I improve retrieval quality and latency?"
./run.sh s03
./run.sh s04
./run.sh s05
./run.sh s06
./run.sh s07
./run.sh s08
./run.sh s09
./run.sh s10
```

### 10-Minute Final Path

```bash
# Build index (uses built-in sample corpus by default)
./run.sh final index --rebuild

# Query with optional scalar filter
./run.sh final ask --question "How can I improve retrieval quality and latency?" --topk 5 --filter "subject == 'ai'"

# End-to-end smoke run
./run.sh final demo --rebuild
```

If model download is blocked by network policy:

```bash
HF_ENDPOINT=https://hf-mirror.com ./run.sh s01
```

Primary chapter source files:
- `chapters/s01_minimal_loop.py`
- `chapters/s02_chunking_lab.py`
- `chapters/s03_schema_filter.py`
- `chapters/s04_retrieval_tuning.py`
- `chapters/s05_two_stage_quality.py`
- `chapters/s06_failure_analysis.py`
- `chapters/s07_boundary_refactor.py`
- `chapters/s08_pluggable_components.py`
- `chapters/s09_evaluation_harness.py`
- `chapters/s10_production_path.py`
- `chapters/sample_corpus.py`

## Project Layout

```text
learn-milvus-harness/
  README.md
  CONTRIBUTING.md
  RELEASE_CHECKLIST.md
  run.sh
  cli.py                  # canonical public CLI entry
  core.py                 # retrieval/indexing core logic
  chapters/
    s01_minimal_loop.py
    s02_chunking_lab.py
    s03_schema_filter.py
    s04_retrieval_tuning.py
    s05_two_stage_quality.py
    s06_failure_analysis.py
    s07_boundary_refactor.py
    s08_pluggable_components.py
    s09_evaluation_harness.py
    s10_production_path.py
    sample_corpus.py      # built-in corpus for first-run tutorials
  boundaries.py           # runtime/service boundary layer
```

## Philosophy

- Agency comes from the model.
- Reliability comes from the harness.
- Great retrieval systems are engineered, measured, and iterated.

## Release Prep

- Contributor guide: `CONTRIBUTING.md`
- Release gate: `RELEASE_CHECKLIST.md`
