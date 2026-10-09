#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

APPLY=0
WITH_LEGACY_PI=0
CONDA_ROOT="${CONDA_ROOT:-/home/jieguo/anaconda3}"
OLD_LEROBOT_SOURCE="${R1A7_OLD_LEROBOT_SOURCE:-$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot}"
NEW_LEROBOT_SOURCE="${R1A7_NEW_LEROBOT_SOURCE:-/media/jieguo/disk/zyc/lerobot}"
PLUGIN_ROOT="${R1A7_LEROBOT_PLUGIN_ROOT:-}"
CYCLONEDDS_SOURCE="${R1A7_CYCLONEDDS_SOURCE:-}"
CYCLONEDDS_SPEC="${R1A7_CYCLONEDDS_SPEC:-cyclonedds==0.10.2}"
DRACCUS_SPEC="${R1A7_DRACCUS_SPEC:-draccus>=0.11.6,<0.12.0}"
LEGACY_TRANSFORMERS_SPEC="${R1A7_LEGACY_TRANSFORMERS_SPEC:-transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi}"

usage() {
  cat <<'EOF'
Usage: setup-r1a7-envs.sh [options]

By default this command is a dry run and only prints the commands it would run.
Pass --apply explicitly to change either environment.

Options:
  --apply                    Execute the displayed installation commands.
  --with-legacy-pi           Install the old fork's custom Transformers branch.
  --conda-root PATH          Conda installation root.
  --old-source PATH          Vendored LeRobot source for unitree_lerobot.
  --new-source PATH          Current upstream LeRobot source.
  --plugin-root PATH         Current R1-A7 plugin project (installed in both environments).
  --cyclonedds-source PATH   Local CycloneDDS Python source/wheel.
  --cyclonedds-spec SPEC     Pinned fallback when no local source is supplied.
  --draccus-spec SPEC        New LeRobot parser dependency range.
  --legacy-transformers SPEC Override the old fork's custom Transformers source.
  -h, --help                 Show this help.

This script never installs LeRobot or the R1 plugin with dependencies and never
uses pip --upgrade. It therefore does not intentionally change Torch.
EOF
}

while (($#)); do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --with-legacy-pi)
      WITH_LEGACY_PI=1
      shift
      ;;
    --conda-root|--old-source|--new-source|--plugin-root|--cyclonedds-source|--cyclonedds-spec|--draccus-spec|--legacy-transformers)
      (($# >= 2)) || {
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      }
      case "$1" in
        --conda-root) CONDA_ROOT="$2" ;;
        --old-source) OLD_LEROBOT_SOURCE="$2" ;;
        --new-source) NEW_LEROBOT_SOURCE="$2" ;;
        --plugin-root) PLUGIN_ROOT="$2" ;;
        --cyclonedds-source) CYCLONEDDS_SOURCE="$2" ;;
        --cyclonedds-spec) CYCLONEDDS_SPEC="$2" ;;
        --draccus-spec) DRACCUS_SPEC="$2" ;;
        --legacy-transformers) LEGACY_TRANSFORMERS_SPEC="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

readonly OLD_PYTHON="$CONDA_ROOT/envs/unitree_lerobot/bin/python"
readonly NEW_PYTHON="$CONDA_ROOT/envs/lerobot/bin/python"
readonly UNITREE_SDK_SOURCE="$ROOT_DIR/vendor/unitree-sdk2-python"
readonly UNITREE_LEROBOT_SOURCE="$ROOT_DIR/data/unitree-lerobot"

if [[ -z "$PLUGIN_ROOT" ]]; then
  plugin_candidates=(
    "$NEW_LEROBOT_SOURCE/integrations/lerobot_robot_unitree_r1a7"
    "$ROOT_DIR/integrations/lerobot-r1a7"
    "$ROOT_DIR/integrations/lerobot_r1a7"
    "$ROOT_DIR/plugins/lerobot-r1a7"
    "$ROOT_DIR/plugins/lerobot_r1a7"
  )
  for candidate in "${plugin_candidates[@]}"; do
    if [[ -f "$candidate/pyproject.toml" ]]; then
      PLUGIN_ROOT="$candidate"
      break
    fi
  done
fi

failed=0
for executable in "$OLD_PYTHON" "$NEW_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    printf 'MISSING environment Python: %s\n' "$executable" >&2
    failed=1
  fi
done
for project in \
  "$OLD_LEROBOT_SOURCE" \
  "$NEW_LEROBOT_SOURCE" \
  "$UNITREE_SDK_SOURCE" \
  "$UNITREE_LEROBOT_SOURCE"; do
  if [[ ! -f "$project/pyproject.toml" && ! -f "$project/setup.py" ]]; then
    printf 'MISSING Python project: %s\n' "$project" >&2
    failed=1
  fi
done
if [[ -z "$PLUGIN_ROOT" || ! -f "$PLUGIN_ROOT/pyproject.toml" ]]; then
  printf 'MISSING R1-A7 plugin; pass --plugin-root PATH\n' >&2
  failed=1
fi
if [[ -n "$CYCLONEDDS_SOURCE" && ! -e "$CYCLONEDDS_SOURCE" ]]; then
  printf 'MISSING local CycloneDDS source/wheel: %s\n' "$CYCLONEDDS_SOURCE" >&2
  failed=1
fi
((failed == 0)) || exit 1

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_command() {
  local label="$1"
  shift
  printf '\n[%s]\n' "$label"
  print_command "$@"
  if ((APPLY)); then
    "$@"
  fi
}

torch_version() {
  "$1" -c 'import torch; print(torch.__version__)'
}

pip_check_with_unitree_headless_exception() {
  local python_bin="$1"
  local output
  local status
  local unexpected=0
  local accepted_headless=0

  set +e
  output="$($python_bin -m pip check 2>&1)"
  status=$?
  set -e
  if ((status == 0)); then
    printf '%s\n' "$output"
    return 0
  fi

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    case "$line" in
      unitree-sdk2py*requires\ opencv-python*which\ is\ not\ installed.*)
        accepted_headless=1
        printf 'KNOWN METADATA EXCEPTION: %s\n' "$line" >&2
        ;;
      *)
        printf '%s\n' "$line" >&2
        unexpected=1
        ;;
    esac
  done <<<"$output"

  if ((accepted_headless)); then
    "$python_bin" -c '
import cv2
from importlib.metadata import version
headless = version("opencv-python-headless")
print(f"Using opencv-python-headless {headless}; cv2={cv2.__version__}")
'
  fi
  ((unexpected == 0 && accepted_headless == 1))
}

readonly OLD_TORCH_BEFORE="$(torch_version "$OLD_PYTHON")"
readonly NEW_TORCH_BEFORE="$(torch_version "$NEW_PYTHON")"

if [[ -n "$CYCLONEDDS_SOURCE" ]]; then
  dds_install=(
    -m pip install --disable-pip-version-check --no-deps --no-build-isolation
    "$CYCLONEDDS_SOURCE"
  )
else
  dds_install=(
    -m pip install --disable-pip-version-check --no-deps "$CYCLONEDDS_SPEC"
  )
fi

printf 'Mode: %s\n' "$([[ $APPLY -eq 1 ]] && printf APPLY || printf DRY-RUN)"
printf 'Old environment: %s (Torch %s)\n' "$OLD_PYTHON" "$OLD_TORCH_BEFORE"
printf 'New environment: %s (Torch %s)\n' "$NEW_PYTHON" "$NEW_TORCH_BEFORE"
printf 'R1-A7 plugin: %s\n' "$PLUGIN_ROOT"
printf 'New LeRobot parser dependency: %s\n' "$DRACCUS_SPEC"
if ((WITH_LEGACY_PI)); then
  printf 'Legacy PI0.5 Transformers: %s\n' "$LEGACY_TRANSFORMERS_SPEC"
else
  printf 'Legacy PI0.5 Transformers: SKIPPED (use --with-legacy-pi)\n'
fi
if [[ -z "$CYCLONEDDS_SOURCE" ]]; then
  printf 'CycloneDDS: %s (pip may need the configured package index/cache)\n' "$CYCLONEDDS_SPEC"
else
  printf 'CycloneDDS local source: %s\n' "$CYCLONEDDS_SOURCE"
fi

run_command "unitree_lerobot: CycloneDDS" "$OLD_PYTHON" "${dds_install[@]}"
run_command \
  "unitree_lerobot: local Unitree SDK" \
  "$OLD_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$UNITREE_SDK_SOURCE"
run_command \
  "unitree_lerobot: vendored LeRobot editable" \
  "$OLD_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$OLD_LEROBOT_SOURCE"
run_command \
  "unitree_lerobot: integration editable" \
  "$OLD_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$UNITREE_LEROBOT_SOURCE"
if ((WITH_LEGACY_PI)); then
  run_command \
    "unitree_lerobot: custom PI0.5 Transformers" \
    "$OLD_PYTHON" -m pip install --disable-pip-version-check --no-deps \
    "$LEGACY_TRANSFORMERS_SPEC"
fi
run_command \
  "unitree_lerobot: R1-A7 plugin editable" \
  "$OLD_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$PLUGIN_ROOT"

run_command "lerobot: CycloneDDS" "$NEW_PYTHON" "${dds_install[@]}"
run_command \
  "lerobot: local Unitree SDK" \
  "$NEW_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$UNITREE_SDK_SOURCE"
run_command \
  "lerobot: compatible draccus" \
  "$NEW_PYTHON" -m pip install --disable-pip-version-check --no-deps "$DRACCUS_SPEC"
run_command \
  "lerobot: current source editable" \
  "$NEW_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$NEW_LEROBOT_SOURCE"
run_command \
  "lerobot: R1-A7 plugin editable" \
  "$NEW_PYTHON" -m pip install --disable-pip-version-check --no-build-isolation --no-deps \
  -e "$PLUGIN_ROOT"

if ((APPLY)); then
  readonly OLD_TORCH_AFTER="$(torch_version "$OLD_PYTHON")"
  readonly NEW_TORCH_AFTER="$(torch_version "$NEW_PYTHON")"
  if [[ "$OLD_TORCH_AFTER" != "$OLD_TORCH_BEFORE" || "$NEW_TORCH_AFTER" != "$NEW_TORCH_BEFORE" ]]; then
    printf 'ERROR: Torch changed unexpectedly (old %s -> %s, new %s -> %s).\n' \
      "$OLD_TORCH_BEFORE" "$OLD_TORCH_AFTER" "$NEW_TORCH_BEFORE" "$NEW_TORCH_AFTER" >&2
    exit 1
  fi
  pip_check_with_unitree_headless_exception "$OLD_PYTHON"
  pip_check_with_unitree_headless_exception "$NEW_PYTHON"
  check_args=(--plugin-root "$PLUGIN_ROOT" --require-plugin --require-hardware)
  if ((WITH_LEGACY_PI)); then
    check_args+=(--require-legacy-pi)
  fi
  "$ROOT_DIR/scripts/lerobot/check_r1a7_setup.py" "${check_args[@]}"
  printf '\nEnvironment refresh completed without changing Torch.\n'
else
  printf '\nDry run only. Re-run with --apply to execute these commands.\n'
fi
