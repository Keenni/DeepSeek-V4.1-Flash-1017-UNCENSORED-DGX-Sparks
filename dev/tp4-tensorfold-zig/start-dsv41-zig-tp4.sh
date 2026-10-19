#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# start-dsv41-zig-tp4.sh — DSV4.1-Flash-1017 zig native TP4 · 纯环 switchless NCCL（2026-10-20）
#
# 形态（对齐 glm53 TF TP4 的成功做法）：光纤环 + 我们打过补丁的 switchless NCCL（2.30.7，
# /root/nccl-switchless，LD_PRELOAD 进容器）；**不开 RDMA rings**（TF_RDMA_DEVICES 不设）——
# world=4 的 2D 交换全部走 NCCL：all-gather 按环中继，对角交换走 ncclSend/Recv。
# NCCL 选腿 = 每机全 4 口白名单 + SUBNET_AWARE_ROUTING(/24)，环上邻居按链路 /24 自动选口。
# 控制面 bootstrap（TCPStore）走 89 管理网（NCCL_SOCKET_IFNAME=enP7s7），数据面 IB/RoCE。
#
# 用法: 只在 dgx03 跑一次（worker-first 3→2→1→0，自带 gate 前自清）；停: stop-dsv41-zig-tp4.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CONTAINER="dsv41-zig-tp4"
MODEL_DIR="/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-v2"
ENGRAM_DIR="/models/DeepSeek-V4.1-Flash-1017-heretic"
ENGINE_DIR="/root/tf1017/engine"
KIT_DIR="/root/tf1017/kit"
CACHE_DIR="/root/tf1017/cache"
NCCL_DIR="/root/nccl-switchless"
IMAGE="nvcr.io/nvidia/pytorch:26.07-py3"
PORT=8000
MASTER_PORT=29571
MASTER_ADDR="192.168.89.103"
SERVED="deepseek-v4.1-flash-uncensored"
CONTEXT=1048576
POOL=8388608
PARALLEL=16
MEM_GATE_GIB=85
WAIT_TIMEOUT=1800

HOSTS=(root@192.168.89.103 root@192.168.89.104 root@192.168.89.101 root@192.168.89.102)
NAMES=(dgx03 dgx04 dgx01 dgx02)

log(){ printf '[tp4] %s\n' "$*"; }
die(){ printf '[tp4] ERROR: %s\n' "$*" >&2; exit 1; }
r(){ ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

# 每机 RoCEv2/IPv4 的 GID index（glm53 TP4 同款：钉错=QP 失败或 hang）
gid_of(){ r "$1" 'for dev in /sys/class/infiniband/*/ports/1; do
  [ -e "$dev/gids" ] || continue
  for idx in 0 1 2 3 4 5 6 7; do
    gid=$(cat "$dev/gids/$idx" 2>/dev/null) || continue
    case "$gid" in ""|0000:0000:0000:0000:0000:0000:0000:0000) continue;; esac
    t=$(cat "$dev/gid_attrs/types/$idx" 2>/dev/null) || continue
    case "$t" in *v2*) ;; *) continue;; esac
    case "$gid" in *ffff:*) echo "$idx"; exit 0;; esac
  done
done; echo 3'; }

log "前置校验"
for g in 0 1 2 3; do
  host=${HOSTS[$g]}; name=${NAMES[$g]}
  r "$host" "test -f $MODEL_DIR/config.json" || die "$name 缺 config.json"
  n=$(r "$host" "ls $MODEL_DIR | grep -c 'of-00043.safetensors' || true")
  [[ "$n" == 43 ]] || die "$name EXL3 片 $n != 43"
  e=$(r "$host" "ls $MODEL_DIR | grep -c '^model-00004[78]-of-00048.safetensors' || true")
  [[ "$e" == 2 ]] || log "WARN $name engram 计数 $e/2（lookup 抖动，交给引擎判定）"
  r "$host" "test -f $MODEL_DIR/chat_template.jinja" || die "$name 缺 chat_template.jinja"
  r "$host" "test -x $ENGINE_DIR/tensorfold-native && test -x $ENGINE_DIR/tf-dsv41-lanes" || die "$name 缺引擎"
  r "$host" "test -f $KIT_DIR/engram.json && test -f $KIT_DIR/aot/aot.json" || die "$name 缺 kit"
  r "$host" "test -f $NCCL_DIR/libnccl.so.2" || die "$name 缺 switchless NCCL($NCCL_DIR)"
  r "$host" "docker image inspect $IMAGE >/dev/null 2>&1" || die "$name 缺镜像 $IMAGE"
  c=$(r "$host" "ls $CACHE_DIR/rank-cache/rank${g}of4-*.bin 2>/dev/null | wc -l")
  [[ "$c" == 1 ]] || die "$name 缺 rank${g}of4 rank-cache"
  r "$host" "test -f $CACHE_DIR/dsv41_token_map.json" || die "$name 缺 token map"
  r "$host" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
  free=$(r "$host" "LC_ALL=C free -g | awk '/^Mem:/{print \$4}'")
  log "  $name rank$g MemFree=${free}GiB"
  (( free >= MEM_GATE_GIB )) || die "$name MemFree ${free}GiB < ${MEM_GATE_GIB}GiB（先停旧栈/trim）"
done
if r "${HOSTS[0]}" "curl -sf --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; then
  die "dgx03 :$PORT 被占用：先停现役 TP2 两对"
fi
log "preflight OK"

# 数据面 = switchless 补丁 NCCL（LD_PRELOAD）；选腿 = 全 4 口 + subnet-aware(/24)；环邻接交换走光纤
GIDS=()
for g in 0 1 2 3; do GIDS+=("$(gid_of "${HOSTS[$g]}")"); done
log "GID index: dgx03=${GIDS[0]} dgx04=${GIDS[1]} dgx01=${GIDS[2]} dgx02=${GIDS[3]}"

ENVS="-e TF_TP_WORLD=4 -e TF_TP_MASTER=$MASTER_ADDR -e TF_TP_PORT=$MASTER_PORT \
 -e TF_DS_RANK_CACHE=/cache/rank-cache -e TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json -e TF_DS_ENGRAM=/engram \
 -e TF_DS_KIT=/kit -e TENSORFOLD_CUDA_KERNELS=/kit/aot \
 -e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=1 -e TF_DS_ENGRAM_AIO=1 -e TF_DS_WINDOW=8388608 -e TF_DS_ROUND_ROWS=48 -e TF_DS_STREAMS=16 -e TF_DS_SERVED_K=solo -e TENSORFOLD_MEMORY_RESERVE_GIB=8 -e TF_DS_2D_GU=parity \
 -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_ALGO=Ring -e NCCL_CROSS_NIC=1 \
 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_IB_MERGE_NICS=0 \
 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_ADDR_RANGE=10.0.0.0/22 \
 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_SOCKET_IFNAME=enP7s7 \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e HF_HUB_OFFLINE=1"

RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --restart no)
MOUNTS="-v $ENGINE_DIR:/engine:ro -v $KIT_DIR:/kit:ro -v $MODEL_DIR:/model:ro -v $ENGRAM_DIR:/engram:ro -v $CACHE_DIR:/cache -v $NCCL_DIR:/opt/patched-nccl:ro"

launch(){
  local g=$1 host=${HOSTS[$1]} name=${NAMES[$1]} gid=${GIDS[$1]}
  if [[ $g == 0 ]]; then
    r "$host" "docker run -d --name $CONTAINER ${RUN_ARGS[*]} $ENVS -e NCCL_IB_GID_INDEX=$gid $MOUNTS \
      $IMAGE /engine/tensorfold-native serve /model --host 0.0.0.0 --port $PORT --name $SERVED \
        --backend cuda --context $CONTEXT --parallel $PARALLEL" >/dev/null || die "rank0($name) 启动失败"
  else
    r "$host" "docker run -d --name $CONTAINER ${RUN_ARGS[*]} $ENVS -e NCCL_IB_GID_INDEX=$gid $MOUNTS \
      $IMAGE /engine/tf-dsv41-lanes /model /cache/rank-cache $g 4 $MASTER_ADDR $MASTER_PORT /kit \
        --pool $POOL --engram /engram --token-map /cache/dsv41_token_map.json \
        --graphs 1 --side 1 --prefetch 1 --aio 1 --round-rows 48 --streams 16" >/dev/null || die "rank$g($name) 启动失败"
  fi
  log "rank$g ($name) 已启动"
}

launch 3; launch 2; launch 1; launch 0
log "四 rank 已启动 (3→2→1→0)，等待就绪（rank<g>of4 装载 + switchless NCCL 会合，最长 ${WAIT_TIMEOUT}s）"

start=$SECONDS
until r "${HOSTS[0]}" "curl -sf --max-time 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; do
  for g in 0 1 2 3; do
    st=$(r "${HOSTS[$g]}" "docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null || echo missing")
    if [[ "$st" != true ]]; then
      log "rank$g (${NAMES[$g]}) 已退出，日志尾:"
      r "${HOSTS[$g]}" "docker logs --tail 40 $CONTAINER 2>&1" | sed 's/^/  | /'
      die "rank$g 在就绪前退出"
    fi
  done
  (( SECONDS - start < WAIT_TIMEOUT )) || die "${WAIT_TIMEOUT}s 未就绪（容器仍在跑）"
  (( (SECONDS - start) % 120 < 6 )) && log "⋯ $((SECONDS-start))s"
  sleep 5
done
log "API 就绪 (耗时 $((SECONDS-start))s)"

smoke=$(r "${HOSTS[0]}" "curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$SERVED\",\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}]}'" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); c=(r["choices"][0]["message"].get("content") or "").strip(); assert c, r; print(repr(c[:40]), r["usage"]["completion_tokens"], "tok")') \
  || die "烟测失败（查四机 docker logs $CONTAINER）"
log "烟测 OK: $smoke"
log "✔ dsv41-zig-tp4 LIVE: http://$MASTER_ADDR:$PORT/v1 · model=$SERVED · ctx=$CONTEXT · par=$PARALLEL · tp=4 · switchless-NCCL"
