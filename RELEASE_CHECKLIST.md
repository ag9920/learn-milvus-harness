# Release Checklist

Use this before publishing a new public version.

## 1) Repository Hygiene

- `LICENSE` exists and is correct.
- `README.md` matches current CLI behavior.
- `CONTRIBUTING.md` exists and is current.
- No internal-only files are tracked.

## 2) CLI & Command Surface

- `python3 cli.py --help` works.
- `python3 cli.py final --help` works.
- `python3 cli.py study --help` lists `s01`-`s10`.
- `./run.sh` usage examples match CLI commands.

## 3) Static Validation

- Python files compile:
  `python3 -m py_compile cli.py core.py boundaries.py chapters/*.py`
- No placeholder markers:
  `TODO/FIXME/TBD` scan is clean.

## 4) Functional Smoke

- `./run.sh setup` succeeds.
- `./run.sh final demo --rebuild` returns non-empty hits.
- `./run.sh s09 --rebuild` prints Recall@K/MRR/nDCG.
- `./run.sh s10 --rebuild` prints doctor + migration checklist.

## 5) Release Notes

- Summarize command changes and chapter additions.
- Mention any known limitations (e.g., network/model-download prerequisites).
