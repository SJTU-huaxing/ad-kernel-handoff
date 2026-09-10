#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source /cache/huaxing/miniforge3/etc/profile.d/conda.sh
conda activate nonlinear-qk
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
INFERENCE_OUT="${1:-$REPO_DIR/work/pretrain_100m_2b/inference_$(date -u +%Y%m%dT%H%M%SZ)}"
if [ "$#" -gt 0 ]; then shift; fi
python experiments/benchmark_100m_inference.py --out "$INFERENCE_OUT" "$@"
python experiments/benchmark_100m_inference_features.py --out "$INFERENCE_OUT/features.json"
python experiments/render_100m_inference.py "$INFERENCE_OUT"
