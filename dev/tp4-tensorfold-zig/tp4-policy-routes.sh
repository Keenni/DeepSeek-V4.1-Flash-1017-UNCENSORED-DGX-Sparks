#!/usr/bin/env bash
# 对角 RoCE 策略路由（按源地址分流，保证无论 NCCL 用哪个口的 GID，出向都走同一 dev）
#   每台：rule from <f0地址> → table 100；rule from <f1地址> → table 200
#   表内只放对角 peer 的两个地址，出向下一跳 = 该 dev 上的邻居（邻居再一跳送到对角）
#   非对角流量不受影响（表内无匹配即回落到 main 表）
set -uo pipefail
log(){ echo "[$(date +%T)] $*"; }
R(){ ssh -o BatchMode=yes "$@"; }

# rank: f0addr f1addr f0dev f1dev f0neigh f1neigh diag_peer_addrs...
setup(){ # host f0addr f1addr f0neigh f1neigh p1 p2
  local h=$1 f0=$2 f1=$3 n0=$4 n1=$5 p1=$6 p2=$7
  R $h "ip route flush table 100 2>/dev/null; ip route flush table 200 2>/dev/null
ip route add $p1/32 via $n0 dev enp1s0f0np0 table 100
ip route add $p2/32 via $n0 dev enp1s0f0np0 table 100
ip route add $p1/32 via $n1 dev enp1s0f1np1 table 200
ip route add $p2/32 via $n1 dev enp1s0f1np1 table 200
ip rule del from $f0 lookup 100 2>/dev/null; ip rule del from $f1 lookup 200 2>/dev/null
ip rule add from $f0 lookup 100 prio 1000; ip rule add from $f1 lookup 200 prio 1010
echo \"  [$h] rules:\"; ip rule | grep -E \"1000|1010\"" >/dev/null
}

log "== 策略路由"
setup dgx03 10.0.1.103 10.0.2.103 10.0.1.102 10.0.2.104 10.0.0.101 10.0.3.101
setup dgx01 10.0.0.101 10.0.3.101 10.0.0.102 10.0.3.104 10.0.1.103 10.0.2.103
setup dgx02 10.0.0.102 10.0.1.102 10.0.0.101 10.0.1.103 10.0.2.104 10.0.3.104
setup dgx04 10.0.2.104 10.0.3.104 10.0.2.103 10.0.3.101 10.0.0.102 10.0.1.102

log "== 用源地址绑定验证（-I 指定出口源）"
chk(){ if R "$1" "ping -c1 -W1 -I $2 $3 >/dev/null 2>&1"; then log "  $1 src $2 -> $3 OK"; else log "  $1 src $2 -> $3 FAIL"; fi; }
chk dgx03 10.0.1.103 10.0.0.101
chk dgx03 10.0.2.103 10.0.0.101
chk dgx01 10.0.0.101 10.0.1.103
chk dgx01 10.0.3.101 10.0.1.103
chk dgx02 10.0.0.102 10.0.2.104
chk dgx02 10.0.1.102 10.0.2.104
chk dgx04 10.0.2.104 10.0.0.102
chk dgx04 10.0.3.104 10.0.0.102
