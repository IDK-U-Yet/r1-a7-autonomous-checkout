#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly BACKEND_DIR="$ROOT_DIR/apps/r1-collection-console/backend"
readonly FRONTEND_DIST="$ROOT_DIR/apps/r1-collection-console/frontend/dist"

TASK_NAME="${TASK_NAME:-}"
SOURCE_HOST="127.0.0.1"
LISTEN_HOST="127.0.0.1"
PORT="8765"
PREVIEW_FPS="15"

usage() {
  cat <<'EOF'
Usage: start-collection-console.sh --task-name NAME [options]

Options:
  --task-name NAME       Collection task directory name (or set TASK_NAME)
  --source-host HOST     teleimager host (default: 127.0.0.1)
  --host HOST            Web listen address (default: 127.0.0.1)
  --port PORT            Web listen port (default: 8765)
  --preview-fps FPS      Browser preview rate (default: 15)
  -h, --help             Show this help
EOF
}

while (($#)); do
  case "$1" in
    --task-name)
      TASK_NAME="${2:?--task-name requires a value}"
      shift 2
      ;;
    --source-host)
      SOURCE_HOST="${2:?--source-host requires a value}"
      shift 2
      ;;
    --host)
      LISTEN_HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --preview-fps)
      PREVIEW_FPS="${2:?--preview-fps requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_NAME" ]]; then
  echo "TASK_NAME is required. Set it or pass --task-name NAME." >&2
  exit 2
fi
if [[ ! "$TASK_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TASK_NAME may contain only letters, numbers, dot, underscore, and dash." >&2
  exit 2
fi
if [[ ! -f "$FRONTEND_DIST/index.html" ]]; then
  echo "Collection console frontend is not built: $FRONTEND_DIST/index.html" >&2
  echo "Build apps/r1-collection-console/frontend before starting the console." >&2
  exit 2
fi

if [[ -z "${TV_PYTHON:-}" ]]; then
  TV_PYTHON="$(command -v python || command -v python3)"
fi

export R1A7_ROOT="$ROOT_DIR"
export PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec "$TV_PYTHON" -m collection_console.server \
  --task-dir "$ROOT_DIR/runtime/datasets/r1a7_cashier_raw/$TASK_NAME" \
  --scanner-log "$ROOT_DIR/runtime/logs/$TASK_NAME.scanner.jsonl" \
  --diagnostics-path "$ROOT_DIR/runtime/logs/$TASK_NAME.jsonl" \
  --source-host "$SOURCE_HOST" \
  --host "$LISTEN_HOST" \
  --port "$PORT" \
  --preview-fps "$PREVIEW_FPS"
