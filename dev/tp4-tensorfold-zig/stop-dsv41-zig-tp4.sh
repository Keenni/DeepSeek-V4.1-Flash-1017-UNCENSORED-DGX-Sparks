#!/usr/bin/env bash
# stop-dsv41-zig-tp4.sh — master-first 停 dsv41 zig TP4 (宪法同款)
set -euo pipefail
C=dsv41-zig-tp4
for h in root@192.168.89.103 root@192.168.89.104 root@192.168.89.101 root@192.168.89.102; do
  echo "[stop] $h"
  ssh -o BatchMode=yes "$h" "docker rm -f $C >/dev/null 2>&1 || true"
done
echo "[stop] done (master-first: 03→04→01→02)"
