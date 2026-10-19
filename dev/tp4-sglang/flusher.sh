#!/usr/bin/env bash
# flusher.sh — 无条件每 60s drop_caches（tonyd2wild "整个诀窍"）
# GB10 无独立 VRAM：NVIDIA 驱动要真正的 MemFree 而非 MemAvailable，
# 阈值触发的 flusher 会长期低于阈值 → 同一命令时好时坏。必须无条件。
while true; do
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
  sleep 60
done
