# Contributing

Thanks for improving `learn-milvus-harness`.

## Development Principles

- Keep the repository fully English for global open-source readability.
- Keep `study` chapters (`s01`-`s10`) independently runnable.
- Keep `final` commands stable (`index`, `ask`, `demo`).
- Prefer explainability over hidden magic in retrieval behavior.

## Local Setup

```bash
./run.sh setup
```

## Quick Validation Before PR

```bash
python3 cli.py --help
python3 cli.py final --help
python3 cli.py study --help
python3 -m py_compile cli.py core.py boundaries.py chapters/*.py
```

## Style Expectations

- Keep module boundaries explicit (`cli` -> `boundaries` -> `core`).
- Keep chapter docs in file docstrings.
- Avoid breaking command names without a migration note.

## PR Checklist

- Command behavior is backward compatible or clearly documented.
- README examples are updated when interfaces change.
- `.gitignore` still blocks local DB/artifact noise.
- No internal/private notes are committed.
