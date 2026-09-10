#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /cache/huaxing/miniforge3/etc/profile.d/conda.sh
conda activate nonlinear-qk
cd "$PROJECT_ROOT"
exec python scripts/tensorboard_service_100m.py "$@"
