#!/usr/bin/env bash
# run_validation.sh — End-to-end posteriordb validation for eXMC
#
# Pulls posteriordb, preprocesses models, runs NUTS validation.
# Everything is self-contained — run from the exmc/ project root.
#
# Usage:
#   ./benchmark/posteriordb/run_validation.sh              # full run
#   ./benchmark/posteriordb/run_validation.sh --skip-pull   # reuse existing data
#   ./benchmark/posteriordb/run_validation.sh --parallel 4  # limit parallelism

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROCESSED_DIR="$SCRIPT_DIR/posteriordb_processed"
PDB_DIR="/tmp/posteriordb"
PARALLEL=""
SKIP_PULL=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --skip-pull) SKIP_PULL=true ;;
    --parallel) shift_next=true ;;
    *)
      if [[ "${shift_next:-}" == "true" ]]; then
        PARALLEL="--parallel $arg"
        shift_next=false
      fi
      ;;
  esac
done

# Also handle --parallel N as two separate args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL="--parallel $2"; shift 2 ;;
    --skip-pull) shift ;;
    *) shift ;;
  esac
done

echo "========================================"
echo "  eXMC posteriordb Validation Pipeline"
echo "========================================"
echo ""

# --- Step 1: Pull posteriordb ---
if [[ "$SKIP_PULL" == "true" ]] && [[ -d "$PDB_DIR" ]]; then
  echo "[1/4] Reusing existing posteriordb at $PDB_DIR"
else
  echo "[1/4] Cloning posteriordb (shallow)..."
  rm -rf "$PDB_DIR"
  git clone --depth 1 https://github.com/stan-dev/posteriordb.git "$PDB_DIR" 2>&1 | tail -1
  echo "      Done — $(du -sh "$PDB_DIR/posterior_database" | cut -f1) of model data"
fi
echo ""

# --- Step 2: Check Python deps ---
echo "[2/4] Checking Python dependencies..."
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "      ERROR: numpy not found. Install with: pip install numpy"
  exit 1
fi
echo "      numpy: OK"
echo ""

# --- Step 3: Preprocess ---
echo "[3/4] Preprocessing posteriordb models..."
python3 "$SCRIPT_DIR/preprocess_posteriordb.py" "$PDB_DIR"
N_MODELS=$(python3 -c "import json; m=json.load(open('$PROCESSED_DIR/manifest.json')); print(m['n_posteriors'])")
echo "      Preprocessed $N_MODELS models to $PROCESSED_DIR/"
echo ""

# --- Step 4: Run eXMC validation ---
echo "[4/4] Running eXMC NUTS validation..."
echo "      Protocol: 1000 warmup + 1000 sampling per model"
echo ""

# Force CPU-only for parallel sampling (GPU is a bottleneck for d<100)
CUDA_VISIBLE_DEVICES="" mix run "$SCRIPT_DIR/validate_posteriordb.exs" $PARALLEL

echo ""
echo "========================================"
echo "  Validation complete."
echo "  Report: $PROCESSED_DIR/validation_results.md"
echo "========================================"
