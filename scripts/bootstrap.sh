#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export R1A7_ROOT="$ROOT_DIR"

command -v git >/dev/null 2>&1 || {
  printf 'git is required\n' >&2
  exit 1
}
command -v git-lfs >/dev/null 2>&1 || {
  printf 'git-lfs is required for the monorepo assets\n' >&2
  exit 1
}

if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'The monorepo has not been initialized yet: %s\n' "$ROOT_DIR" >&2
  printf 'Initialize the main repository before running bootstrap.\n' >&2
  exit 1
fi

git -C "$ROOT_DIR" lfs install --local

"$ROOT_DIR/scripts/check.sh"
printf 'Monorepo initialized under %s\n' "$ROOT_DIR"
