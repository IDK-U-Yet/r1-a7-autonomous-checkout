#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REVIEW_PYTHON="${REVIEW_PYTHON:-$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python}"
export PYTHONPATH="$ROOT_DIR/apps/r1-dataset-review/backend${PYTHONPATH:+:$PYTHONPATH}"
exec "$REVIEW_PYTHON" -m dataset_review.export "$@"
