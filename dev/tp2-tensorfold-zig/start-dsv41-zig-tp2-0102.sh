#!/bin/bash
# =====================================================================
# start-dsv41-zig-tp2-0102.sh  — DSV4.1-Flash-1017 EXL3 2.9bpw UNCENSORED, TensorFold zig TP2
#
# 上游（每次改动/排障先看这些，不要凭记忆）:
#   配方 repo   : https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10
#   引擎(现役)  : https://github.com/bertholomus/TensorFold  分支 deepseek-v41-zig   (TensorFold 1.0.x native Zig, v0.6.0+)
#   引擎(旧Python): https://github.com/bertholomus/TensorFold  分支 deepseek-v41-tp2   (v0.1.0–v0.5.1)
#   权重(母包)  : https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw
#   权重(去审查): https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw
#   kernel kit  : 上述配方 repo 的 kit/ （v0.6.1 起已入库，含 MANIFEST + verify_kit.sh）
#   Engram 数据 : DeepSeek 原始 1017 片 47/48（本脚本用 /models/DeepSeek-V4.1-Flash-1017-heretic）
#   同类参考    : https://github.com/jayleaton/deepseek-v41-tensorfold-spark
#                 https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-TensorFold-2x-DGX-Spark
#                 https://github.com/ZackO2o/DeepSeek-V4.1-Flash-TensorFold-DGXSpark-TP2
#
# 与 TP4 脚本的区别: TF_TP_WORLD=2; rank-cache 走 node_inputs 预生成复用(引擎 find 只读不自建; 69158a0+7def4b4 修复版二进制)
# 集群铁律: 只在 dgx01 跑一次 / worker-first / master 恒 dgx01 / :8000 / restart=no / 不碰在跑的生产容器
# =====================================================================
set -euo pipefail

CONTAINER="dsv41-zig-tp2-0102"
MODEL_DIR="/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-v2"
ENGRAM_DIR="/models/DeepSeek-V4.1-Flash-1017-heretic"
ENGINE_DIR="/root/tf1017/engine"
KIT_DIR="/root/tf1017/kit"
CACHE_DIR="/root/tf1017/cache"
LOGDIR="/root/tf1017/logs"
IMAGE="nvcr.io/nvidia/pytorch:26.07-py3"
PORT=8000
MASTER_PORT=29620
MASTER_ADDR="192.168.89.101"      # rank0 = dgx01
SERVED="deepseek-v4.1-flash-uncensored"
CONTEXT=1048576
PARALLEL=4
POOL=1048576
MEM_GATE_GIB=85

H="root@192.168.89.101"           # dgx01 rank0
W="root@192.168.89.102"           # dgx02 rank1

log(){ printf '[tp2] %s\n' "$*"; }
die(){ printf '[tp2] ERROR: %s\n' "$*" >&2; exit 1; }
r(){ ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

log "前置校验"
for host in "$H" "$W"; do
  r "$host" "test -f $MODEL_DIR/config.json" || die "$host 缺 config.json"
  n=$(r "$host" "ls $MODEL_DIR | grep -c 'of-00043.safetensors'")
  [[ "$n" == 43 ]] || die "$host EXL3 片 $n != 43"
  e=$(r "$host" "ls $MODEL_DIR | grep -c '^model-00004[78]-of-00048.safetensors'")
  [[ "$e" == 2 ]] || log "WARN $host engram 计数 $e/2（lookup 抖动，继续交给引擎判定）"
  r "$host" "test -x $ENGINE_DIR/tensorfold-native" || die "$host 缺 tensorfold-native"
  r "$host" "test -x $ENGINE_DIR/tf-dsv41-lanes" || die "$host 缺 tf-dsv41-lanes"
  r "$host" "test -f $KIT_DIR/engram.json" || die "$host 缺 kit/engram.json"
  r "$host" "docker image inspect $IMAGE >/dev/null 2>&1" || die "$host 缺镜像 $IMAGE"
  free=$(r "$host" "LC_ALL=C free -g | awk '/^Mem:/{print \$4}'")
  log "$host MemFree=${free}GiB"
  (( free >= MEM_GATE_GIB )) || die "$host MemFree ${free}GiB < ${MEM_GATE_GIB}GiB"
done
if r "$H" "curl -sf --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; then
  die "dgx01 :$PORT 被占用"
fi

# token map (独立于 rank-cache, 秒级)
if ! r "$H" "test -f $CACHE_DIR/dsv41_token_map.json"; then
  log "生成 token map (dgx01)"
  r "$H" "cd $ENGINE_DIR && python3 make_token_map.py $MODEL_DIR $CACHE_DIR/dsv41_token_map.json" || die "token map 生成失败"
fi
if ! r "$W" "test -f $CACHE_DIR/dsv41_token_map.json"; then
  log "同步 token map 到 dgx02"
  r "$H" "scp -q $CACHE_DIR/dsv41_token_map.json root@192.168.89.102:$CACHE_DIR/" || die "token map 同步失败"
fi

# rank-cache 铁律: 引擎不自建(model.zig find 只读, 缺=NoRankCache 直接死)——在则复用, 缺则预生成, 绝不清
for host in "$H" "$W"; do
  if [[ "$host" == "$H" ]]; then rk=0; else rk=1; fi
  r "$host" "mkdir -p $CACHE_DIR/rank-cache"
  n=$(r "$host" "ls $CACHE_DIR/rank-cache/rank${rk}of2-*.bin 2>/dev/null | wc -l")
  if [[ "$n" == 0 ]]; then
    log "$host 缺 rank${rk}of2, node_inputs 预生成(约10min, 只读权重)"
    r "$host" "bash $ENGINE_DIR/node_inputs.sh $MODEL_DIR $CACHE_DIR $rk 2" || die "$host rank${rk}of2 预生成失败"
  fi
  r "$host" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
done

RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --restart no)

launch(){
  local rank=$1 host
  [[ $rank == 0 ]] && host=$H || host=$W
  local envs="-e TF_TP_WORLD=2 -e TF_TP_MASTER=$MASTER_ADDR -e TF_TP_PORT=$MASTER_PORT -e TF_DS_RANK_CACHE=$CACHE_DIR/rank-cache -e TF_DS_TOKEN_MAP=$CACHE_DIR/dsv41_token_map.json -e TF_DS_ENGRAM=$ENGRAM_DIR"
  r "$host" "mkdir -p $LOGDIR"
  if [[ $rank == 0 ]]; then
    r "$host" "docker run -d --name $CONTAINER ${RUN_ARGS[*]} \
      -e $envs -e TENSORFOLD_NO_UPDATE_CHECK=1 -e HF_HUB_OFFLINE=1 \
      -v $ENGINE_DIR:/engine:ro -v $KIT_DIR:/kit:ro -v $MODEL_DIR:/model:ro -v $ENGRAM_DIR:/engram:ro -v $CACHE_DIR:/cache \
      $IMAGE /engine/tensorfold-native serve /model --host 0.0.0.0 --port $PORT --name $SERVED \
        --backend cuda --context $CONTEXT --parallel $PARALLEL \
        --engram /engram --token-map /cache/dsv41_token_map.json \
        --rdma rocep1s0f0,rocep1s0f1 --rdma-kernels /engine/rdma_gather.fatbin \
        --graphs 1 --side 1 --prefetch 1 --aio 1" >/dev/null || die "rank0 启动失败"
  else
    r "$host" "docker run -d --name $CONTAINER ${RUN_ARGS[*]} \
      -e $envs -e TENSORFOLD_NO_UPDATE_CHECK=1 -e HF_HUB_OFFLINE=1 \
      -v $ENGINE_DIR:/engine:ro -v $KIT_DIR:/kit:ro -v $MODEL_DIR:/model:ro -v $ENGRAM_DIR:/engram:ro -v $CACHE_DIR:/cache \
      $IMAGE /engine/tf-dsv41-lanes /model /cache/rank-cache 1 2 $MASTER_ADDR $MASTER_PORT /kit \
        --engram /engram --token-map /cache/dsv41_token_map.json \
        --pool $POOL --rdma rocep1s0f0,rocep1s0f1 --rdma-kernels /engine/rdma_gather.fatbin \
        --graphs 1 --side 1 --prefetch 1 --aio 1" >/dev/null || die "rank1 启动失败"
  fi
  log "rank$rank ($host) 已启动"
}

launch 1; launch 0
log "等待就绪 (首启含 rank-cache 写入 ~106GB/rank, 最长 1800s)"
start=$SECONDS
until r "$H" "curl -sf --max-time 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; do
  for spec in "0:$H" "1:$W"; do
    rank=${spec%%:*}; host=${spec#*:}
    st=$(r "$host" "docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null || echo missing")
    if [[ "$st" != "true" ]]; then
      log "rank$rank ($host) 已退出, 日志尾:"; r "$host" "docker logs --tail 30 $CONTAINER 2>&1" | sed 's/^/  | /'
      die "rank$rank 在就绪前退出"
    fi
  done
  (( SECONDS - start < 1800 )) || die "1800s 未就绪 (容器仍在跑)"
  (( (SECONDS - start) % 120 < 6 )) && log "⋯ $((SECONDS-start))s"
  sleep 5
done
log "API 就绪 (耗时 $((SECONDS-start))s)"
r "$H" "curl -s --max-time 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$SERVED\",\"max_tokens\":32,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}]}'" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"].get("content") or ""; print("[tp2] smoke:", repr(c.strip()[:60]), r["usage"]["completion_tokens"], "tok")' \
  || die "烟测失败"
log "✔ dsv41-zig-tp2 LIVE: http://$MASTER_ADDR:$PORT/v1  model=$SERVED ctx=$CONTEXT par=$PARALLEL"
