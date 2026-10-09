#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REVIEW_PYTHON="${REVIEW_PYTHON:-$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python}"
export PYTHONPATH="$ROOT_DIR/apps/r1-dataset-review/backend${PYTHONPATH:+:$PYTHONPATH}"
if (($# == 0)); then
  set -- --dataset-root "$ROOT_DIR/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train" \
    --dataset-root "$ROOT_DIR/runtime/datasets/r1a7_cashier_raw/cashier_oreo_16d_v2_100" \
    --dataset-root "$ROOT_DIR/runtime/datasets/r1a7_cashier_raw/cashier_oreo_16d_v3_standard_80"
fi
exec "$REVIEW_PYTHON" -m dataset_review.server "$@"
