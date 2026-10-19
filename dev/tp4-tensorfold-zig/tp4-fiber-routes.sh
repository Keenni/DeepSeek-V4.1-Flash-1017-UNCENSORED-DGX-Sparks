#!/usr/bin/env bash
# 对角 /32 路由补全矩阵（每个 rank 到其余 3 个 peer 的 f0/f1 两个地址）
# 节点端口地址：01=10.0.0.101/10.0.3.101  02=10.0.0.102/10.0.1.102
#              03=10.0.1.103/10.0.2.103  04=10.0.2.104/10.0.3.104
set -uo pipefail
log(){ echo "[$(date +%T)] $*"; }
R(){ ssh -o BatchMode=yes "$@"; }

log "== 03: 补 10.0.0.102(f0 via 02) + 10.0.3.104(04f1 via 04f0)"
R dgx03 'ip route replace 10.0.0.102/32 via 10.0.1.102 dev enp1s0f0np0; ip route replace 10.0.3.104/32 via 10.0.2.104 dev enp1s0f1np1'
log "== 02: 补 10.0.2.103(03f1 via 03f0)"
R dgx02 'ip route replace 10.0.2.103/32 via 10.0.1.103 dev enp1s0f1np1'
log "== 01: 补 10.0.2.103 已有? 校验并补齐 10.0.3.104/10.0.2.104"
R dgx01 'ip route replace 10.0.1.103/32 via 10.0.3.104 dev enp1s0f1np1; ip route replace 10.0.2.103/32 via 10.0.3.104 dev enp1s0f1np1; ip route replace 10.0.2.104/32 via 10.0.3.104 dev enp1s0f1np1; ip route replace 10.0.1.102/32 via 10.0.0.102 dev enp1s0f0np0'
log "== 04: 校验 10.0.2.103/10.0.1.103/10.0.0.101/10.0.0.102/10.0.1.102"
R dgx04 'ip route replace 10.0.1.103/32 via 10.0.2.103 dev enp1s0f0np0; ip route replace 10.0.0.101/32 via 10.0.3.101 dev enp1s0f1np1; ip route replace 10.0.1.102/32 via 10.0.3.101 dev enp1s0f1np1; ip route replace 10.0.0.102/32 via 10.0.3.101 dev enp1s0f1np1'

log "== 连通性：每 rank → 其余 3 个 peer 的 2 个地址，共 24 条"
fail=0
chk(){ if R "$1" "ping -c1 -W1 $2 >/dev/null 2>&1"; then :; else log "  FAIL $1 -> $2"; fail=1; fi; }
for a in 10.0.0.102 10.0.1.102 10.0.2.104 10.0.3.104; do chk dgx03 $a; done
for a in 10.0.0.101 10.0.3.101 10.0.2.104 10.0.3.104; do chk dgx01 $a; done
for a in 10.0.0.101 10.0.3.101 10.0.1.103 10.0.2.103; do chk dgx02 $a; done
for a in 10.0.0.101 10.0.3.101 10.0.0.102 10.0.1.102; do chk dgx04 $a; done
[ $fail = 0 ] && log "MATRIX OK" || log "MATRIX HAS FAILURES"
