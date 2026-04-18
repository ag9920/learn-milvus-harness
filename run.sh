#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-final}"
shift || true

ensure_deps() {
  python3 - <<'PY'
import importlib.util
import sys

required = {
    "pymilvus": "pymilvus[milvus_lite,model]",
    "transformers": "transformers<5",
    "pypdf": "pypdf",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Missing dependencies: " + ", ".join(missing))
    print("Run: ./run.sh setup")
    sys.exit(1)
PY
}

run_cli() {
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    python3 cli.py "$@"
  else
    env -u HF_ENDPOINT python3 cli.py "$@"
  fi
}

case "$CMD" in
  setup)
    echo "[setup] Installing/checking dependencies..."
    python3 -m pip install "pymilvus[milvus_lite,model]" "setuptools<81" "transformers<5" "pypdf"
    echo "[setup] Dependencies are ready."
    ;;
  final)
    ensure_deps
    FINAL_CMD="${1:-demo}"
    shift || true
    echo "[run] final ${FINAL_CMD}"
    case "$FINAL_CMD" in
      index|ask|demo)
        run_cli final "$FINAL_CMD" "$@"
        ;;
      *)
        echo "Unknown final command: $FINAL_CMD"
        echo "Use: ./run.sh final [index|ask|demo] ..."
        exit 1
        ;;
    esac
    ;;
  s01)
    ensure_deps
    echo "[run] study s01"
    run_cli study s01 "$@"
    ;;
  s02)
    ensure_deps
    echo "[run] study s02"
    run_cli study s02 "$@"
    ;;
  s03)
    ensure_deps
    echo "[run] study s03"
    run_cli study s03 --rebuild "$@"
    ;;
  s04)
    ensure_deps
    echo "[run] study s04"
    run_cli study s04 --rebuild "$@"
    ;;
  s05)
    ensure_deps
    echo "[run] study s05"
    run_cli study s05 --rebuild "$@"
    ;;
  s06)
    ensure_deps
    echo "[run] study s06"
    run_cli study s06 --rebuild "$@"
    ;;
  s07)
    ensure_deps
    echo "[run] study s07"
    run_cli study s07 --rebuild "$@"
    ;;
  s08)
    ensure_deps
    echo "[run] study s08"
    run_cli study s08 --rebuild "$@"
    ;;
  s09)
    ensure_deps
    echo "[run] study s09"
    run_cli study s09 --rebuild "$@"
    ;;
  s10)
    ensure_deps
    echo "[run] study s10"
    run_cli study s10 --rebuild "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage:"
    echo "  ./run.sh setup"
    echo "  ./run.sh final index --rebuild"
    echo "  ./run.sh final ask --question 'How can I improve retrieval recall?' --topk 5 --filter \"subject == 'ai'\""
    echo "  ./run.sh final demo --rebuild"
    echo "  ./run.sh s01 --question 'How should I balance RAG latency and recall?'"
    echo "  ./run.sh s02 --question 'How can I improve retrieval quality and latency?'"
    echo "  ./run.sh s03"
    echo "  ./run.sh s04"
    echo "  ./run.sh s05"
    echo "  ./run.sh s06"
    echo "  ./run.sh s07"
    echo "  ./run.sh s08"
    echo "  ./run.sh s09"
    echo "  ./run.sh s10"
    echo ""
    echo "Optional mirror for restricted networks:"
    echo "  HF_ENDPOINT=https://hf-mirror.com ./run.sh s01"
    exit 1
    ;;
esac
