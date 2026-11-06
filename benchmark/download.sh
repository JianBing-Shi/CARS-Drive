#!/usr/bin/env bash
# Download reference implementations only; leave data and dependencies untouched.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-all}" in
  all) names=(navsim bench2drive nuscenes_planning) ;;
  navsim|bench2drive|nuscenes_planning) names=("$1") ;;
  *) echo "Usage: bash benchmark/download.sh [all|navsim|bench2drive|nuscenes_planning]" >&2; exit 2 ;;
esac

for name in "${names[@]}"; do
  case "$name" in
    navsim) url=https://github.com/autonomousvision/navsim.git; directory=navsim ;;
    bench2drive) url=https://github.com/Thinklab-SJTU/Bench2Drive.git; directory=Bench2Drive ;;
    nuscenes_planning) url=https://github.com/OpenDriveLab/UniAD.git; directory=UniAD ;;
  esac
  target="$root/$directory"
  if [[ ! -e "$target" ]]; then
    git clone --depth 1 "$url" "$target"
  fi
  echo "$name: $target"
  git -C "$target" rev-parse HEAD
done
