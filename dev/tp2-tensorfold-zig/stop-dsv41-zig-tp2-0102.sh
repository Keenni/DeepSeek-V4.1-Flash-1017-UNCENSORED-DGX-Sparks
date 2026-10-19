#!/usr/bin/env bash
# stop-dsv41-zig-tp2-0102.sh — 停 dsv41 zig TP2 组合 (worker-first rm)
set -euo pipefail
C=dsv41-zig-tp2-0102
H1=root@192.168.89.102; H2=root@192.168.89.101   # 0102HOSTS
for h in $H1 $H2; do
  echo "[stop] $h"
  ssh -o BatchMode=yes "$h" "docker rm -f $C >/dev/null 2>&1 || true"
done
echo "[stop] done"
