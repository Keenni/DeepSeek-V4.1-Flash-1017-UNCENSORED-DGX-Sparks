#!/usr/bin/env bash
# ============================================================================
# start-dsv41-tp4.sh — DeepSeek-V4.1-Flash-UNCENSORED on 4x DGX Spark (TP4, v2 = knapcio production line)
# ============================================================================
# 上游 repo（recipe/adapter/boot.py 的唯一来源，更新后跑 `update` 子命令跟随）:
#   https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks
#   （TP4 专用入口 start-tp4.sh + .env.tp4.example；本脚本只接管编排与本地化参数）
#
# 调优下游 profile（EP2/并发16/chunk4096/shared_pad_k/indexer_chunked 等，含实测数字）:
#   https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4
#   （MiaAI-Lab 上游的下游调优档，adapters 更激进；采纳前按 §7.3 先评估再选择性吸收）
#
# 本集群改造（vs 上游，全部有意为之）：
#   - 编排自写（不套用上游 start.sh 的 NFS/共享目录机制）：权重 476G 已平铺在
#     四台本地 /models/DeepSeek-V4.1-Flash-UNCENSORED-FP8，各机本地直挂只读，零 NFS。
#   - 端口 8000（上游 8888）；served 名 deepseek-v4.1-flash-uncensored。
#   - Engram 打包目录 /models/dsv41-uncensored-engram（上游 ~/dsv41-engram）。
#   - NCCL 用本集群 patched 2.30.7 skip-tree（/root/nccl-switchless，四台已就位），
#     与 GLM TP4 同一套 switchless 环参数（NCCL_SKIP_TREE_CONNECT=1 等）。
#   - 编译/内核缓存持久化：宿主 /root/.cache/huggingface → 容器 /cache/huggingface，
#     容器重建不丢（TRITON/FLASHINFER/CUDA cache 全指过去）。
#   - 内置 tonyd2wild 四步 ritual（swappiness=0 持久化 + swap cycle + drop_caches
#     + 无条件 flusher）与 cgroup --memory 112g 护栏，master 就绪后自动停 flusher。
#   - 互斥清理：主入口起栈前自动强删四台所有 LLM 容器（GLM TP4 会被停）。
#
# 集群拓扑（2026-09-15 实测）：
#   光纤环 dgx01→dgx02→dgx03→dgx04→dgx01，网段 10.0.0/1/2/3，MTU 9000。
#   rank 环序 0→1→2→3→0 = dgx03→dgx04→dgx01→dgx02→dgx03，四条 Ring 边全直连。
#   rank0=dgx03(master/API)  rank1=dgx04  rank2=dgx01  rank3=dgx02
#   dgx00 不入环（无光纤口，跑 ComfyUI）。
#
# 用法（在任意一台发起，默认 dgx02）：
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh            # 全流程起栈（含清理/打包/ritual）
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh doctor     # 严格预检
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh update     # 拉上游最新 repo → 四台重 build 镜像
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh build      # 仅同步 repo + 四台 build 镜像
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh pack       # 仅打包 Engram 分片（幂等，缺才打）
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh stop       # 停栈（master-first，四台）
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh sync       # 仅同步部署件 + adapter-override 到四台
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh status     # 四台容器 + API
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh logs [head|w1|w2|w3] [N]
#   bash /root/dsv41-tp4/start-dsv41-tp4.sh smoke      # 19+23 算术验收
# 内部子模式（跨机 ssh 调用，勿手动用）：worker-only <rank> | master-only | pack-only <rank>
# ============================================================================
set -uo pipefail

DEPLOY_DIR="/root/dsv41-tp4"
ENV_FILE="$DEPLOY_DIR/.env.tp4"
UPSTREAM_REPO="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks"
REPO_DIR="$DEPLOY_DIR/upstream"
# 必须是本地版、绝不能被上游 reset 冲掉的文件（相对 $REPO_DIR）。
# 加新项后跑 `bash $0 vendor-save` 建备份。
# 2026-09-26：原 pin 的 adapter/sitecustomize.py 已取消 —— 我们的 house patch
# （shared_pad_k 挂点）上游 225ad3c 原样合入，继续钉旧版会覆盖掉新的
# deepseek_v4_dspark 分支，导致镜像内自测 test_draft_head_fp8_tp4 直接 AttributeError。
VENDORED_FILES=(
)
FLUSHER="$DEPLOY_DIR/flusher.sh"
ADAPTER_OVERRIDE_DIR="$DEPLOY_DIR/adapter-override"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YEL=$'\033[1;33m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YEL}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }
die()  { err "$*"; exit 1; }

[[ -f "$ENV_FILE" ]] || die "缺 $ENV_FILE"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

HEAD_CTN="dsv41-head"
WORKER_CTN="dsv41-worker"
LLM_CTN_RE='glm|vllm|sglang|dsv4|qwen|exl3|dflash|mtp|deepseek|flash|spark-vllm'

# rank 表：环序必须匹配物理环 03→04→01→02→03
#   rank0=dgx03(88.103) rank1=dgx04(88.104) rank2=dgx01(88.101) rank3=dgx02(88.102)
rank_of() {
  case "$1" in
    dgx03) echo 0 ;; dgx04) echo 1 ;; dgx01) echo 2 ;; dgx02) echo 3 ;;
    *) echo -1 ;;
  esac
}
ME="$(hostname)"
ME_RANK="$(rank_of "$ME")"
[[ "$ME_RANK" -ge 0 ]] || die "本机 $ME 不在 TP4 四机清单 (dgx01/02/03/04)"
if [[ "$ME_RANK" == "0" ]]; then
  REMOTE_RANKS=(1 2 3)
else
  REMOTE_RANKS=()  # 本机是 worker：worker 们由本机自己起，master 由 ssh 触发
fi
host_of_rank() {
  case "$1" in
    0) echo dgx03 ;; 1) echo dgx04 ;; 2) echo dgx01 ;; 3) echo dgx02 ;;
  esac
}
ip_of_host() {
  case "$1" in
    dgx01) echo 192.168.89.101 ;; dgx02) echo 192.168.89.102 ;;
    dgx03) echo 192.168.89.103 ;; dgx04) echo 192.168.89.104 ;;
  esac
}
host_ip_of_rank() { ip_of_host "$(host_of_rank "$1")"; }

remote() { # remote <host> <cmd...>
  local h="$1"; shift
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "root@$(ip_of_host "$h")" "$*"
}
remote_bg() { # 免阻塞触发（远端 nohup，不等返回）
  local h="$1"; shift
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "root@$(ip_of_host "$h")" \
    "nohup $* >/dev/null 2>&1 &"
}

# ── 互斥清理：四台强删所有 LLM 容器（只随主入口发生，子模式不含清理） ──────────
cleanup_node() {
  local h="$1" names
  if [[ "$h" == "$ME" ]]; then
    names=$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E "$CTN_RE" || true)
  else
    names=$(remote "$h" "docker ps -a --format '{{.Names}}' | grep -E '$CTN_RE'" 2>/dev/null || true)
  fi
  if [[ -n "$names" ]]; then
    warn "$h 清理: $(echo "$names" | tr '\n' ' ')"
    if [[ "$h" == "$ME" ]]; then
      echo "$names" | xargs -r docker rm -f >/dev/null 2>&1
    else
      remote "$h" "echo '$names' | xargs -r docker rm -f" >/dev/null
    fi
  fi
}
CTN_RE='^.*(glm|vllm|sglang|dsv4|qwen|exl3|dflash|mtp|deepseek|flash).*$'
cleanup_llm_all() {
  info "互斥清理：四台强删 LLM 旧容器（含在跑的 GLM TP4）"
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do cleanup_node "$h"; done
}

# ── HiCache L3 的 NFS 共享（幂等）：rank0(dgx03) 为服务端，其余三台挂同名路径 ──
# 为什么必须在脚本里：v1 时代是手工落的一次性操作，9-24 宿主重启时三台挂载失败
# （Network is unreachable for 192.168.89.103，88→89 迁移所致）后 fstab 行丢失，
# 之后 L3 全白建 —— 而四 rank 的 all_reduce(MIN) 会把命中数抹成 0，表象与"没开"一致。
HICACHE_HOST_DIR_DEFAULT=/root/dsv41-hicache
ensure_hicache_nfs() {
  # HiCache 关闭时不该碰 NFS（HICACHE_ENABLE=1 才执行）
  if [[ "${HICACHE_ENABLE:-0}" != "1" ]]; then info "HiCache 关闭（HICACHE_ENABLE!=1），跳过 hicache NFS 共享"; return 0; fi
  local server; server=$(ip_of_host dgx03)
  local server_host=dgx03
  local line="$server:$HICACHE_HOST_DIR_DEFAULT $HICACHE_HOST_DIR_DEFAULT nfs vers=4.2,hard,timeo=600,retrans=2,_netdev,x-systemd.after=network-online.target,x-systemd.mount-timeout=60 0 0"
  local h
  # 服务端：确保导出（幂等）
  remote "$server_host" "grep -qF '$HICACHE_HOST_DIR_DEFAULT ' /etc/exports || echo '$HICACHE_HOST_DIR_DEFAULT 192.168.89.0/24(rw,sync,no_subtree_check,no_root_squash)' >> /etc/exports; exportfs -ra" \
    >/dev/null 2>&1 || warn "dgx03 hicache export 配置失败"
  for h in dgx04 dgx01 dgx02; do
    remote "$h" "set -e
      mkdir -p $HICACHE_HOST_DIR_DEFAULT
      if grep -qF 'dsv41-hicache' /etc/fstab; then
        sed -i '\\#dsv41-hicache#d' /etc/fstab
      else
        cp -a /etc/fstab /etc/fstab.bak-hicache
      fi
      echo '$line' >> /etc/fstab
      systemctl daemon-reload
      mountpoint -q $HICACHE_HOST_DIR_DEFAULT || systemctl start 'root-dsv41\\x2dhicache.mount' 2>/dev/null || mount $HICACHE_HOST_DIR_DEFAULT
      n=\$(find $HICACHE_HOST_DIR_DEFAULT -type f 2>/dev/null | wc -l); echo "hicache-nfs \$n"" \
      >/dev/null 2>&1 || warn "$h hicache NFS 挂载失败"
    local n; n=$(remote "$h" "find $HICACHE_HOST_DIR_DEFAULT -type f 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r')
    [[ -n "$n" && "$n" != "0" ]] && info "hicache NFS: $h 已挂载（$n 文件）" || warn "hicache NFS: $h 无内容（$n）"
  done
  info "hicache NFS 就绪（服务端 $server_host，L3 共享可生效）"
}

# ── tonyd2wild 四步 ritual（每台）：GB10 统一内存保命 ─────────────────────────
ritual_node() {
  local h="$1"
  local script='
    grep -qx "vm.swappiness=0" /etc/sysctl.d/99-dsv41-tp4.conf 2>/dev/null || \
      echo "vm.swappiness=0" > /etc/sysctl.d/99-dsv41-tp4.conf
    sysctl -w vm.swappiness=0 >/dev/null
    swapoff -a && swapon -a
    sync; echo 3 > /proc/sys/vm/drop_caches
    pkill -f "dsv41-tp4/flusher.sh" 2>/dev/null
    setsid nohup bash /root/dsv41-tp4/flusher.sh >/dev/null 2>&1 &
    echo ritual-ok'
  if [[ "$h" == "$ME" ]]; then bash -c "$script" >/dev/null
  else remote "$h" "$script" >/dev/null; fi
  info "$h ritual 完成（swappiness=0 + swap cycle + drop_caches + flusher 已起）"
}
ritual_all() {
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do ritual_node "$h"; done
}
stop_flusher_all() {
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do
    if [[ "$h" == "$ME" ]]; then pkill -f "dsv41-tp4/flusher.sh" 2>/dev/null || true
    else remote "$h" "pkill -f dsv41-tp4/flusher.sh" 2>/dev/null || true; fi
  done
  info "四台 flusher 已停"
}

# ── 上游 repo + overlay 镜像 ──────────────────────────────────────────────────
ensure_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    local before after
    before=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo '?')
    git -C "$REPO_DIR" fetch origin >/dev/null 2>&1 || true
    git -C "$REPO_DIR" reset --hard origin/HEAD >/dev/null 2>&1 \
      || git -C "$REPO_DIR" pull --ff-only >/dev/null 2>&1 || true
    after=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo '?')
    if [[ "$before" != "$after" && "$before" != '?' ]]; then
      warn "上游 HEAD 已变: ${before:0:7} → ${after:0:7}（本次 serve 将用新代码，先看改动面）"
      git -C "$REPO_DIR" diff --name-only "$before" "$after" 2>/dev/null | sed 's/^/    /' || true
    fi
    restore_vendored
  else
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$UPSTREAM_REPO" "$REPO_DIR" || die "clone 上游 repo 失败"
  fi
  local head_sha
  head_sha=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')
  info "上游 repo 就绪 @ $head_sha ($UPSTREAM_REPO)"
}
# ── 上游 repo 的本地 patch 保护 ────────────────────────────────────────────────
# ensure_repo 里 `git reset --hard origin/HEAD` 会冲掉 tracked 文件的手工改动。
# 这里给"必须是本地版"的文件做影子备份 + 自动恢复，并让上游 HEAD 变化显式可见。
# 备份放 $REPO_DIR/.vendor（未跟踪，reset --hard 不删）。
VENDOR_DIR="$REPO_DIR/.vendor"
VENDOR_LIST="$DEPLOY_DIR/VENDORED"

restore_vendored() {
  [[ -f "$VENDOR_LIST" ]] || return 0
  local rel restored=0
  while read -r rel; do
    [[ -z "$rel" || "$rel" == \#* ]] && continue
    if [[ ! -f "$VENDOR_DIR/$rel" ]]; then
      warn "vendored 备份缺失: $rel（跑 vendor-save 重建）"; continue
    fi
    if ! cmp -s "$VENDOR_DIR/$rel" "$REPO_DIR/$rel"; then
      mkdir -p "$(dirname "$REPO_DIR/$rel")"
      cp -a "$VENDOR_DIR/$rel" "$REPO_DIR/$rel"
      warn "已恢复被 reset 冲掉的本地 patch: $rel"
      restored=1
    fi
  done < "$VENDOR_LIST"
  return 0
}

vendor_save() {
  mkdir -p "$VENDOR_DIR"
  local rel n=0
  for rel in "${VENDORED_FILES[@]}"; do
    [[ -f "$REPO_DIR/$rel" ]] || die "vendor 源不存在: $REPO_DIR/$rel"
    mkdir -p "$VENDOR_DIR/$(dirname "$rel")"
    cp -a "$REPO_DIR/$rel" "$VENDOR_DIR/$rel"
    n=$((n+1))
  done
  printf '%s\n' "${VENDORED_FILES[@]}" > "$VENDOR_LIST"
  info "已登记 $n 个 vendored 文件（上游 reset 后自动恢复）"
}

cmd_vendor_diff() {
  [[ -f "$VENDOR_LIST" ]] || die "还没登记 vendored 文件"
  local rel d=0
  while read -r rel; do
    [[ -z "$rel" || "$rel" == \#* ]] && continue
    if ! cmp -s "$VENDOR_DIR/$rel" "$REPO_DIR/$rel"; then
      echo "=== $rel （repo 版本 vs 我们的版本）==="
      diff -u "$REPO_DIR/$rel" "$VENDOR_DIR/$rel" | head -80 || true
      d=$((d+1))
    fi
  done < "$VENDOR_LIST"
  [[ $d == 0 ]] && info "vendored 文件与 repo 一致（上游未改动它们）"
}

sync_repo_all() {
  # 参考件用途（集中式 build 只需本机 repo）；目标就是远端 $REPO_DIR 本身
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do
    [[ "$h" == "$ME" ]] && continue
    info "同步上游 repo → $h:$REPO_DIR"
    ssh "root@$(ip_of_host "$h")" "mkdir -p $DEPLOY_DIR && rm -rf $REPO_DIR" 2>/dev/null
    rsync -a -e "ssh -o StrictHostKeyChecking=no" "$REPO_DIR/" "root@$(ip_of_host "$h"):$REPO_DIR/"
  done
}

sync_deploy_all() {
  # 部署件（脚本/.env/flusher）必须四台一致——worker-only/master-only 由远端执行本脚本，
  # 缺文件就静默起不来（2026-09-15 实踩：只同步了 repo，worker 一台没起）
  # adapter-override 同样必须字节一致：各 rank hook 行为不一致 = TP 步进 hang。
  # v2（2026-09-26）：repo 拉到 225ad3c（knapcio TP4 production line），新 sitecustomize
  # 依赖全部 25+ 个 adapter 模块，而旧 overlay 镜像 /opt/dsv41/adapter 是 fba4470
  # 快照 —— override 白名单改为「repo 里全部 .py + row_store.cpp/.so 除外」，
  # 上游新增模块自动跟上（对应 v1 时代 loop_abort 缺失的教训）。
  # canary-roce 镜像 build 时已 COPY 全量 adapter，此 override 仍安全（同 md5 覆盖）。
  local OVERRIDE_SRCS=()
  local _py
  for _py in "$REPO_DIR"/adapter/*.py; do
    OVERRIDE_SRCS+=("$_py")
  done
  # house 文件不在 repo 跟踪里（reset 不动，但要进 override）
  [[ -f "$REPO_DIR/adapter/dynamic_chunk.py" ]] || OVERRIDE_SRCS+=("$REPO_DIR/adapter/dynamic_chunk.py")
  mkdir -p "$ADAPTER_OVERRIDE_DIR"
  cp "${OVERRIDE_SRCS[@]}" "$ADAPTER_OVERRIDE_DIR/" || die "adapter-override 拷贝失败（$REPO_DIR 缺文件？）"
  # row_store 的 .so 必须同在 override 目录：PYTHONPATH 首选 override，
  # 缺 .so 会让 sglang.srt.multimodal.processors.deepseek_v41（原生多模态）导入失败，
  # 回落到 transformers_auto → TokenizersBackend 无 .tokenizer → head 起不来。
  # .so 只在镜像里编译产物，从镜像按需提取一次。
  if [[ ! -f "$ADAPTER_OVERRIDE_DIR/librow_store.so" && ! -f "$REPO_DIR/adapter/librow_store.so" ]]; then
    local _cid
    _cid=$(docker create "$IMAGE") || die "docker create 失败（取 librow_store.so）"
    docker cp "$_cid:/opt/dsv41/adapter/librow_store.so" "$ADAPTER_OVERRIDE_DIR/librow_store.so" \
      || { docker rm "$_cid" >/dev/null; die "从镜像提取 librow_store.so 失败"; }
    docker rm "$_cid" >/dev/null
    info "已从镜像提取 librow_store.so → adapter-override"
  fi
  local h _ov
  for h in dgx01 dgx02 dgx03 dgx04; do
    [[ "$h" == "$ME" ]] && continue
    ssh "root@$(ip_of_host "$h")" "mkdir -p $DEPLOY_DIR $ADAPTER_OVERRIDE_DIR" 2>/dev/null
    rsync -a -e "ssh -o StrictHostKeyChecking=no" \
      "$DEPLOY_DIR/start-dsv41-tp4.sh" "$DEPLOY_DIR/.env.tp4" "$DEPLOY_DIR/flusher.sh" \
      "$DEPLOY_DIR/VENDORED" \
      "root@$(ip_of_host "$h"):$DEPLOY_DIR/" || die "部署件同步到 $h 失败"
    rsync -a --delete -e "ssh -o StrictHostKeyChecking=no" \
      "$ADAPTER_OVERRIDE_DIR/" "root@$(ip_of_host "$h"):$ADAPTER_OVERRIDE_DIR/" || die "adapter-override 同步到 $h 失败"
    info "部署件 + adapter-override（$(ls "$ADAPTER_OVERRIDE_DIR" | wc -l | tr -d ' ') 文件）→ $h"
  done
}

# 部署习惯（用户固化）：四台绝不各自去外网拉 45G base 镜像。
# 集中式：本机一次 pull base + build overlay → **ctr export/import 流式内网分发**
# （2026-09-26 起：本集群 docker save 已坏——containerd content store 里 layer 被 GC，
#  产出「层数=0」空壳，install 看着正常、起容器才炸。绝不用 docker save/load。）
# 分发后**必须 docker create 实测**，不能只看 docker inspect。
IMAGE_TAR_DIR="$DEPLOY_DIR/image"
distribute_image() {
  local img_id="$1" h ok=1
  local -a DIST_PIDS=()
  img_id="${img_id#sha256:}"
  # containerd 里的名字是 docker.io/library/<tag>，ctr 用简名会 "image not found"
  local cref
  cref=$(ctr -n moby images ls -q 2>/dev/null | grep -E "(^|/)$IMAGE\$" | head -1)
  [[ -n "$cref" ]] || cref="$IMAGE"
  info "ctr 引用名: $cref"
  for h in dgx01 dgx02 dgx03 dgx04; do
    [[ "$h" == "$ME" ]] && continue
    local ip; ip=$(ip_of_host "$h")
    # 幂等门：远端已是同一镜像 ID 就直接跳过（ctr 流不支持断点续传，
    # 所以「已在位不重传」是唯一能省钱的做法；重跑 build 因此接近零成本）
    local pre
    pre=$(ssh -o ConnectTimeout=6 "root@$ip" "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" | tr -d '\r')
    if [[ "${pre#sha256:}" == "$img_id" ]]; then
      info "$h: 已是同 ID 镜像（${img_id:0:12}），跳过分发"
      continue
    fi
    # 并发分发：每台一个后台流（三路同时，都吃同一份本地 blob 读 + 各自网口）
    (
      info "$h: ctr export → ssh import（内网流式，并发）..."
      ctr -n moby images export - "$cref" \
        | ssh -o ConnectTimeout=10 "root@$ip" "ctr -n moby images import -" >/dev/null \
        || { err "$h: ctr 流式分发失败"; exit 1; }
      local rid
      rid=$(ssh "root@$ip" "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" | tr -d '\r')
      [[ "${rid#sha256:}" == "$img_id" ]] \
        || { err "$h: 镜像 ID 不符（远端 ${rid:-无} vs 本地 $img_id）"; exit 1; }
      # 实测门（铁律）：docker create 过一遍，空壳镜像在这里必炸
      ssh "root@$ip" "docker create $IMAGE >/dev/null && cid=\$(docker create $IMAGE) && docker rm \$cid >/dev/null" >/dev/null 2>&1 \
        || { err "$h: docker create 实测失败（镜像可能是空壳）"; exit 1; }
      info "$h: 镜像 ID 对账一致 + docker create 实测 ✓"
    ) &
    DIST_PIDS+=($!)
  done
  local p
  for p in "${DIST_PIDS[@]}"; do wait "$p" || ok=0; done
  [[ "$ok" == 1 ]] || die "镜像分发有失败（查上方输出）"
  info "overlay 镜像四台就绪（ID 一致: $img_id）"
}
build_all() {
  ensure_repo
  info "拉 base 镜像（仅本机一次外网）: $BASE_IMAGE"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || \
    docker pull --platform linux/arm64 "$BASE_IMAGE" || die "base 镜像 pull 失败"
  info "build overlay: $IMAGE ← $REPO_DIR/upstream (Dockerfile: ${BUILD_DOCKERFILE:-Dockerfile})"
  # 构建上下文 = repo 根（$REPO_DIR 本身就是 upstream checkout）；
  # -f 必须给**绝对路径**：BuildKit 按 CLI CWD 解析相对 -f，跟 context 不一致会报
  # "failed to read dockerfile"（v1 还多写一层 upstream/ → context 直接不存在）。
  docker build -f "$REPO_DIR/${BUILD_DOCKERFILE:-Dockerfile}" ${BUILD_ARGS:+$BUILD_ARGS} \
    -t "$IMAGE" "$REPO_DIR" >/tmp/dsv41-build-$ME.log 2>&1 \
    || { tail -30 /tmp/dsv41-build-$ME.log; die "overlay build 失败"; }
  local img_id
  img_id=$(docker image inspect -f '{{.Id}}' "$IMAGE") || die "overlay 镜像 inspect 失败"
  info "本机镜像 ID: ${img_id#sha256:}"
  # 本地自测：docker create 过一遍
  docker create "$IMAGE" >/dev/null || die "本机 docker create 失败（空壳镜像？）"
  ld=$(docker create "$IMAGE") && docker rm "$ld" >/dev/null
  distribute_image "$img_id"
}
cmd_update() {
  ensure_repo
  sync_repo_all
  build_all
  warn "repo/镜像已更新。生效需重启栈（权重/Engram 不受影响）。"
}

# ── Engram 打包（幂等：各机缺自己的 rank 分片才打） ────────────────────────────
pack_node() {
  local r="$1" h
  h=$(host_of_rank "$r")
  local marker="$ENGRAM_DIR/engram-l1-r${r}of4.bin"
  local have
  if [[ "$h" == "$ME" ]]; then
    have=$( [[ -f "$marker" ]] && echo 1 || echo 0 )
  else
    have=$(remote "$h" "[[ -f $marker ]] && echo 1 || echo 0" 2>/dev/null | tr -d '\r')
  fi
  if [[ "$have" == "1" ]]; then info "$h: Engram rank$r 分片已存在，跳过"; return 0; fi
  info "$h: 打包 Engram rank$r（~48GiB，约 10 分钟）..."
  local cmd="mkdir -p $ENGRAM_DIR && docker run --rm --network host \
    -v $MODEL_DIR:$MODEL_IN_CONTAINER:ro \
    -v $ENGRAM_DIR:/engram \
    -e DSV41_SOURCE=$MODEL_IN_CONTAINER \
    --entrypoint python3 $IMAGE \
    /opt/dsv41/scripts/pack_engram.py --rank $r --tp $TP_SIZE --out /engram"
  if [[ "$h" == "$ME" ]]; then bash -c "$cmd" || die "pack rank$r 失败（本机）"
  else remote "$h" "$cmd" || die "pack rank$r 失败（$h）"; fi
}
cmd_pack() {
  local r
  for r in 0 1 2 3; do pack_node "$r"; done
  info "Engram 四台就绪（$ENGRAM_DIR）"
}

# ── 容器启动（本机执行；env 全字面量展开，无跨机转义问题） ─────────────────────
run_rank_local() {
  # bash 数组构造 docker run 参数：值全部引号包裹，无 eval 二次展开，
  # 含空格的值（EXTRA_SGLANG_ARGS 等）不会被 word-split 撕碎。
  local rank="$1" ctn frac
  if [[ "$rank" == "0" ]]; then
    ctn="$HEAD_CTN"; frac="$HEAD_MEM_FRACTION_STATIC"
  else
    ctn="$WORKER_CTN"; frac="$MEM_FRACTION_STATIC"
  fi
  local -a a=()
  a+=(
    --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all
    --shm-size "$SHM_SIZE" --ulimit memlock=-1:-1 --ulimit stack=67108864
    --memory "$CONTAINER_MEMORY" --memory-swap "$CONTAINER_MEMORY" --restart no
    --device /dev/infiniband:/dev/infiniband
    -v "$MODEL_DIR:$MODEL_IN_CONTAINER:ro"
    -v "$ENGRAM_DIR:/engram"
    -e "DSV41_PACKED_DIR=$DSV41_PACKED_DIR"
    -e "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=$SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"
    -e "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
    -v "$CACHE_HOST_DIR:$CACHE_CONTAINER_DIR"
    -v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"
    -v "$ADAPTER_OVERRIDE_DIR:/opt/dsv41/adapter-override:ro"
    # [HiCache OFF] -v "$HICACHE_HOST_DIR:/hicache" 已摘（worker 上残留 NFS 挂载点会让 docker 起容器失败）
    -v "$DEPLOY_DIR/state-r$rank:/state"
    -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR"
    -e "LD_PRELOAD=$NCCL_CONTAINER_DIR/libnccl.so.2"
    -e "EP_SUPPRESS_NCCL_CHECK=1"
    -e "NODE_RANK=$rank" -e "NNODES=$NNODES"
    -e "TP_SIZE=$TP_SIZE" -e "EP_SIZE=$EP_SIZE"
    -e "DIST_INIT_ADDR=$HEAD_IP:$DIST_PORT"
    -e "MODEL_PATH=$MODEL_IN_CONTAINER"
    -e "DSV41_SOURCE=$MODEL_IN_CONTAINER"
    -e "STATE_PATH=/state" -e "SERVER_PORT=$PORT" -e "HOST=0.0.0.0"
    -e "OFFLOAD_MODE=$OFFLOAD_MODE" -e "DSV41_CACHE_GIB=$DSV41_CACHE_GIB"
    -e "DSV41_IO_THREADS=$DSV41_IO_THREADS" -e "DSV41_RESIDENT_SCALES=$DSV41_RESIDENT_SCALES"
    -e "DSV41_CACHE_WAYS=$DSV41_CACHE_WAYS" -e "DSV41_STATS_SECONDS=$DSV41_STATS_SECONDS"
    -e "CONTEXT_LENGTH=$CONTEXT_LENGTH" -e "MEM_FRACTION_STATIC=$frac"
    -e "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
    -e "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
    -e "MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS"
    -e "CUDA_GRAPH_MAX_BS_DECODE=$MAX_RUNNING_REQUESTS"
    -e "SPEC_ALGO=$SPEC_ALGO" -e "DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE"
    -e "DSPARK_SPS_TABLE=$DSPARK_SPS_TABLE" -e "DSPARK_STS_TABLE=$DSPARK_STS_TABLE"
    -e "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
    -e "SKIP_PREPARE=1" -e "SKIP_VERIFY=1" -e "WARMUP=$WARMUP"
    -e "NCCL_NET=$NCCL_NET" -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    -e "NCCL_IB_HCA=$IB_HCA"
    -e "NCCL_IB_ADDR_FAMILY=$NCCL_IB_ADDR_FAMILY"
    -e "NCCL_IB_ROCE_VERSION_NUM=$NCCL_IB_ROCE_VERSION_NUM"
    -e "NCCL_IGNORE_CPU_AFFINITY=$NCCL_IGNORE_CPU_AFFINITY"
    -e "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    -e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE" -e "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
    -e "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE"
    -e "NCCL_IB_SUBNET_PREFIX_LEN=$NCCL_IB_SUBNET_PREFIX_LEN"
    -e "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING"
    -e "NCCL_IB_MERGE_NICS=$NCCL_IB_MERGE_NICS" -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
    -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
    -e "NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE"
    -e "NCCL_ALGO=$NCCL_ALGO" -e "NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
    -e "NCCL_LL128_BUFFSIZE=$NCCL_LL128_BUFFSIZE" -e "NCCL_PROTO=$NCCL_PROTO"
    -e "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
    -e "NCCL_SKIP_TREE_CONNECT=$NCCL_SKIP_TREE_CONNECT"
    -e "NCCL_SWITCHLESS_RING_ONLY=$NCCL_SWITCHLESS_RING_ONLY"
    -e "NCCL_DEBUG=$NCCL_DEBUG" -e "NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
    -e "DSV41_TP_PAD=$DSV41_TP_PAD"
    -e "DSV41_MXFP8_BACKEND=$DSV41_MXFP8_BACKEND"
    -e "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=$SGLANG_FLASHINFER_MOE_FUSED_FINALIZE"
    -e "DSV41_PREFILL_EMPTY_CACHE_TOKENS=$DSV41_PREFILL_EMPTY_CACHE_TOKENS"
    -e "DSV41_SHARED_PAD_K=$DSV41_SHARED_PAD_K"
    -e "DSV41_SHARED_PAD_BUF_ROWS=$DSV41_SHARED_PAD_BUF_ROWS"
    -e "PYTHONPATH=/opt/dsv41/adapter-override:/opt/dsv41/adapter:/opt/b12x:/opt/b12x_next"
    # rustup shim 探针（sglang srt/rust_extensions/loader.py 会跑 cargo --version --verbose）：
    # 镜像 PATH 首位是 /root/.cargo/bin，而那是 rustup 符号链接——在部分机器上会卡在
    # futex 等 toolchain/锁（dgx04 实测挂死 3min+，其余三台秒失败），导致四台启动进度不一致。
    # 我们不用 Rust 扩展（SGLANG_RUST_BUILD_MODE=never），直接从 PATH 里摘掉它 → 四台统一秒失败走 PIL 兜底。
    -e "PATH=/opt/sglang/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    # [HiCache OFF] SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR 已摘
    # [HiCache OFF] SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE 已摘
    -e "SGLANG_DSPARK_ENABLE_SPS_RECORD=$SGLANG_DSPARK_ENABLE_SPS_RECORD"
    -e "SGLANG_RAGGED_VERIFY_MODE=$SGLANG_RAGGED_VERIFY_MODE"
    -e "DSV41_CHUNK_BUDGET=$DSV41_CHUNK_BUDGET"
    -e "DSV41_CHUNK_MAX=$DSV41_CHUNK_MAX"
    -e "DSV41_CHUNK_MIN=$DSV41_CHUNK_MIN"
    -e "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    -e "EXTRA_SGLANG_ARGS=$EXTRA_SGLANG_ARGS"
    -e "DSV41_LAUNCHER=tp4"
    -e "DSV41_SHARED_PAD_K=${DSV41_SHARED_PAD_K:-0}"
    -e "DSV41_SHARED_PAD_BUF_ROWS=${DSV41_SHARED_PAD_BUF_ROWS:-256}"
    -e "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    -e "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
    -e "HF_HOME=$HF_HOME"
    -e "VLLM_HOST_IP=$(host_ip_of_rank "$rank")"
    -e "HOST_IP=$(host_ip_of_rank "$rank")"
  )
  # TP4 production：EXTRA_CONTAINER_ENV 逐项展开为 -e（非法项直接 die）；
  # 占位符 <PLAN_PY_OUTPUT> 未替换会在 launch 前的干跑校验里被抓住。
  if [[ -n "${EXTRA_CONTAINER_ENV:-}" ]]; then
    local _kv
    for _kv in $EXTRA_CONTAINER_ENV; do
      [[ "$_kv" == *"=<PLAN_PY_OUTPUT>"* ]] && die "EXTRA_CONTAINER_ENV: B12X_ROCE_PEER_HCA_MAPS 占位符未替换 —— 先跑 scripts/ring_mesh/plan.py 生成 peer maps"
      [[ "$_kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "EXTRA_CONTAINER_ENV: '$_kv' 不是 KEY=VALUE"
      a+=(-e "$_kv")
    done
  fi
  [[ -n "${API_KEY:-}" ]] && a+=(-e "API_KEY=$API_KEY")
  docker rm -f "$ctn" >/dev/null 2>&1 || true
  mkdir -p "$DEPLOY_DIR/state-r$rank"
  mkdir -p "$HICACHE_HOST_DIR"
  if [[ "$rank" == "0" ]]; then
    docker run -d --name "$ctn" "${a[@]}" \
      -e "SKIP_SMOKE=$SKIP_SMOKE" -e "SMOKE_QUICK=$SMOKE_QUICK" \
      "$IMAGE" run >/dev/null || die "master 容器启动失败"
    info "master (rank0) 容器已起: $ctn"
  else
    docker run -d --name "$ctn" "${a[@]}" \
      -e "SKIP_SMOKE=1" -e "SMOKE_QUICK=1" \
      "$IMAGE" run >/dev/null || die "worker rank$rank 容器启动失败"
    info "worker (rank$rank) 容器已起: $ctn"
  fi
}
wait_ctn_running() {
  local ctn="$1" tries="${2:-24}"
  local i=0
  while (( i < tries )); do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$ctn" 2>/dev/null)" == "true" ]]; then return 0; fi
    sleep 5; i=$((i+1))
  done
  return 1
}

# ── 主流程 ────────────────────────────────────────────────────────────────────
preflight() {
  info "预检: 四机连通 / 权重 / 镜像 / GPU 占用"
  local h bad=0
  for h in dgx01 dgx02 dgx03 dgx04; do
    remote "$h" "test -f $MODEL_DIR/config.json && docker image inspect $IMAGE >/dev/null 2>&1 && echo ok" \
      >/dev/null 2>&1 || { err "$h: 权重或镜像缺失（build/pack 未做？）"; bad=1; }
  done
  [[ "$bad" == 0 ]] || die "预检不过：先跑 build / pack"
  info "预检通过"
}

cmd_serve() {
  sync_deploy_all
  ensure_hicache_nfs
  cleanup_llm_all
  ensure_repo
  sync_repo_all
  # 镜像四台都在则跳过 build（首次/更新后自动 build）
  local h missing=0
  for h in dgx01 dgx02 dgx03 dgx04; do
    remote "$h" "docker image inspect $IMAGE >/dev/null 2>&1" 2>/dev/null || missing=1
  done
  if [[ "$missing" == 1 ]]; then build_all; fi
  cmd_pack
  preflight
  ritual_all

  # worker-first（Mia 编排序 rank1..N 先起，master 最后）
  local r
  for r in 1 2 3; do
    local hh; hh=$(host_of_rank "$r")
    if [[ "$hh" == "$ME" ]]; then
      run_rank_local "$r"
      wait_ctn_running "$WORKER_CTN" 24 || die "worker rank$r 容器未进入 running"
    else
      info "触发 $hh worker rank$r ..."
      remote_bg "$hh" "bash $DEPLOY_DIR/start-dsv41-tp4.sh worker-only $r"
      sleep 2
    fi
  done
  # 等 worker 容器都 running（跨机）
  for r in 1 2 3; do
    local hh; hh=$(host_of_rank "$r")
    [[ "$hh" == "$ME" ]] && continue
    local i=0 ok=0
    while (( i < 24 )); do
      if remote "$hh" "[[ \$(docker inspect -f '{{.State.Running}}' $WORKER_CTN 2>/dev/null) == true ]]" 2>/dev/null; then ok=1; break; fi
      sleep 5; i=$((i+1))
    done
    [[ "$ok" == 1 ]] || die "worker rank$r ($hh) 未进入 running，查: bash $0 logs w$r"
  done

  info "起 master (rank0, API :$PORT)..."
  local mh; mh=$(host_of_rank 0)
  if [[ "$mh" == "$ME" ]]; then
    run_rank_local 0
  else
    remote_bg "$mh" "bash $DEPLOY_DIR/start-dsv41-tp4.sh master-only"
  fi

  info "等待 master 就绪（首次启动含 JIT 编译，~20 分钟属正常；权重加载+TP 握手勿中断）..."
  local i=0
  while (( i < 360 )); do
    if remote "$mh" "docker logs $HEAD_CTN 2>&1 | grep -q '^Ready: API on port $PORT'" 2>/dev/null; then
      stop_flusher_all
      info "API 就绪: http://$(ip_of_host dgx03):$PORT  (served: $SERVED_MODEL_NAME)"
      cmd_smoke || warn "smoke 未过，看日志: bash $0 logs"
      return 0
    fi
    local st
    st=$(remote "$mh" "docker inspect -f '{{.State.Status}}' $HEAD_CTN 2>/dev/null" 2>/dev/null | tr -d '\r')
    case "$st" in
      running|created|restarting) ;;
      *) err "master 容器状态: $st —— 失败，日志: bash $0 logs"; exit 1 ;;
    esac
    sleep 10; i=$((i+1))
    (( i % 18 == 0 )) && info "仍在等待... $(( i * 10 / 60 ))min"
  done
  die "等待 master 就绪超时（60min）。日志: bash $0 logs"
}

cmd_stop() {
  info "停栈（master-first）"
  local mh; mh=$(host_of_rank 0)
  remote "$mh" "docker rm -f $HEAD_CTN" >/dev/null 2>&1 || true
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do
    [[ "$h" == "$mh" ]] && continue
    remote "$h" "docker rm -f $WORKER_CTN" >/dev/null 2>&1 || true
  done
  stop_flusher_all
  info "四台容器已清（master-first），flusher 已停"
}

cmd_status() {
  local h
  for h in dgx01 dgx02 dgx03 dgx04; do
    echo "== $h =="
    remote "$h" "docker ps -a --filter name=dsv41- --format '{{.Names}}  {{.Status}}'" 2>/dev/null || warn "$h ssh 失败"
  done
  echo "== API :$PORT =="
  curl -fsS --max-time 5 "http://$(ip_of_host dgx03):$PORT/v1/models" 2>/dev/null \
    | head -c 300 && echo || echo "(未响应)"
}

cmd_logs() {
  local who="${1:-head}" lines="${2:-120}"
  case "$who" in
    head) remote "$(host_of_rank 0)" "docker logs --tail $lines $HEAD_CTN" 2>/dev/null || true ;;
    w[1-3]) local r="${who#w}"
      remote "$(host_of_rank "$r")" "docker logs --tail $lines $WORKER_CTN" 2>/dev/null || true ;;
    *) remote "$(host_of_rank 0)" "docker logs --tail $lines $HEAD_CTN" 2>/dev/null || true ;;
  esac
}

cmd_smoke() {
  curl -fsS --max-time 180 "http://$(ip_of_host dgx03):$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 + 23? Reply only with the number.\"}]}" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"].strip(); print("smoke:", c); assert "42" in c' \
    && info "smoke 通过" 
}

cmd_doctor() {
  info "doctor: 严格预检"
  local h bad=0
  for h in dgx01 dgx02 dgx03 dgx04; do
    echo "--- $h ---"
    remote "$h" "hostname; test -d /dev/infiniband && echo IB_OK || echo IB_MISSING; \
      test -f $MODEL_DIR/config.json && echo weights_OK || echo weights_MISSING; \
      docker image inspect $IMAGE >/dev/null 2>&1 && echo image_OK || echo image_MISSING; \
      test -f $NCCL_HOST_DIR/libnccl.so.2.30.7 && echo nccl_OK || echo nccl_MISSING; \
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -2; \
      ip -br link show | grep -E 'enp1s0f[01]' " 2>/dev/null || { err "$h ssh 失败"; bad=1; }
  done
  echo "--- 环四条边 MTU9000 ping ---"
  for pair in "10.0.0.101 10.0.0.102" "10.0.1.102 10.0.1.103" "10.0.2.103 10.0.2.104" "10.0.3.104 10.0.3.101"; do
    set -- $pair
    ping -M do -s 8900 -c 2 -W 2 -I enp1s0f0np0 "$1" >/dev/null 2>&1 || ping -M do -s 8900 -c 2 -W 2 "$1" >/dev/null 2>&1 \
      || { warn "ping $1 失败（或本机无该口）"; }
  done
  [[ "$bad" == 0 ]] || die "doctor 有阻塞项"
  info "doctor 完成"
}

# ── 跨机子模式分派（必须在主 case 之前） ───────────────────────────────────────
if [[ "${1:-}" == "worker-only" ]]; then
  rank="${2:-}"; [[ -n "$rank" ]] || die "worker-only 需要 rank 参数"
  mkdir -p "$DEPLOY_DIR/state-r$rank"
  run_rank_local "$rank"
  wait_ctn_running "$WORKER_CTN" 24 || die "worker rank$rank 容器未 running"
  exit $?
fi
if [[ "${1:-}" == "master-only" ]]; then
  run_rank_local 0
  exit $?
fi
if [[ "${1:-}" == "pack-only" ]]; then
  pack_node "${2:-0}"
  exit $?
fi

CMD="${1:-serve}"
case "$CMD" in
  serve|start|"") cmd_serve ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  sync) sync_deploy_all; ensure_hicache_nfs ;;
  logs) shift; cmd_logs "${1:-head}" "${2:-120}" ;;
  smoke) cmd_smoke ;;
  doctor) cmd_doctor ;;
  update) cmd_update ;;
  vendor-save) vendor_save ;;
  vendor-diff) cmd_vendor_diff ;;
  build) ensure_repo; sync_repo_all; sync_deploy_all; build_all ;;
  dist) distribute_image "$(docker image inspect -f '{{.Id}}' "$IMAGE")" ;;
  pack) cmd_pack ;;
  -h|--help|help) sed -n '2,32p' "$0" | tr -d '#' ;;
  *) die "未知命令: $CMD (serve|stop|status|logs|smoke|doctor|update|vendor-save|vendor-diff|build|pack)" ;;
esac
