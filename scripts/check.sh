#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly REQUIRED_PATHS=(
  robot/xr-teleoperate
  robot/dex1-service
  data/unitree-lerobot
  vendor/unitree-sdk2
  vendor/unitree-sdk2-python
  versions.lock
)
readonly FORBIDDEN_NESTED_GIT_PATHS=(
  robot/xr-teleoperate/.git
  robot/xr-teleoperate/.gitmodules
  robot/xr-teleoperate/teleop/robot_control/dex-retargeting/.git
  robot/xr-teleoperate/teleop/teleimager/.git
  robot/xr-teleoperate/teleop/televuer/.git
  robot/dex1-service/.git
  data/unitree-lerobot/.git
  data/unitree-lerobot/.gitmodules
  data/unitree-lerobot/unitree_lerobot/lerobot/.git
  vendor/unitree-sdk2/.git
  vendor/unitree-sdk2-python/.git
)

failed=0
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$ROOT_DIR/$path" ]]; then
    printf 'MISSING  %s\n' "$path" >&2
    failed=1
  fi
done

for path in "${FORBIDDEN_NESTED_GIT_PATHS[@]}"; do
  if [[ -e "$ROOT_DIR/$path" || -L "$ROOT_DIR/$path" ]]; then
    printf 'NESTED_GIT  %s\n' "$path" >&2
    failed=1
  fi
done

if git -C "$ROOT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  printf '%-30s %s\n' "monorepo" "$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'no-commit')"
else
  printf 'NOT_GIT  main repository is not initialized\n' >&2
  failed=1
fi

if command -v rg >/dev/null 2>&1; then
  if rg -l -x -F 'version https://git-lfs.github.com/spec/v1' \
      --glob '!runtime/**' --glob '!**/.git/**' "$ROOT_DIR"; then
    printf 'Unresolved Git LFS pointer found\n' >&2
    failed=1
  fi

  legacy_root='/home/user/unitree-'R1A7
  if rg -n -F "$legacy_root" \
      --glob '!runtime/**' --glob '!**/.git/**' --glob '!**/.venv/**' \
      --glob '!**/node_modules/**' "$ROOT_DIR"; then
    printf 'Hard-coded legacy repository path found\n' >&2
    failed=1
  fi
fi

exit "$failed"
