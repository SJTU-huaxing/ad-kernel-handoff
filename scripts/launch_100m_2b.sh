#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /cache/huaxing/miniforge3/etc/profile.d/conda.sh
conda activate nonlinear-qk
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-900}"
export TOKENIZERS_PARALLELISM=false
# Keep the platform's ASCEND_VISIBLE_DEVICES mapping (currently physical 3,2).
exec python scripts/run_100m_2b.py "$@"
