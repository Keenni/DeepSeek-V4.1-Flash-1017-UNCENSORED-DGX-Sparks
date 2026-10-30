# DeepSeek-V4.1-Flash-1017（去审查）在 DGX Spark 上的四套可用栈

把 **DeepSeek-V4.1-Flash-1017** 的去审查版跑在 NVIDIA DGX Spark（GB10）上的部署件、引擎补丁与实测数据：
两机 **TP2** 走 TensorFold 的 Zig 引擎 + EXL3 2.9bpw 重打包；四机 **TP4** 走同一个 Zig 引擎，或者走
SGLang + 原生 FP8/FP4 权重。

这里的脚本是我们在跑的脚本，数字是我们实测的数字。测试环境是四台 GB10 组成的 **无交换机（switchless）200G 光纤环**
（不是全互联），所有请求都是真实推理。[English →](README.md)

**关键词：** switchless TP4 · 无交换机光纤环 · DGX Spark TP4 · TensorFold Zig（全 AOT、CUDA 侧无 JIT——换权重
布局就得改引擎源码）· SGLang TP4 · EXL3 2.9bpw · 原生 FP8/FP4 · DeepSeek-V4.1-Flash-1017 去审查。

读之前先知道两件事：

* **这里两套 TP4 都是 switchless-TP4**：四机、四条直连光纤成环、fabric 里没有交换机。对 **Zig 线**影响最大——
  它的 2D 切分需要对角对，环上没有对角链路，所以对角流量走「四机 all-gather」（我们的 patch）。
* **Zig 引擎在 CUDA 侧没有运行期编译，所以引擎不认识的权重布局 = 改源码。** SGLang / vLLM 是 Python 引擎：
  新量化、新布局通常只是加一个类，kernel 由 Triton JIT / FlashInfer autotune 在首次调用时编译。TensorFold
  则是全 AOT——CUDA kernel 是预编 cubin（`cuModuleLoadData` 加载，运行期编译只存在于 macOS 的 Metal 路径），
  权重加载的 dtype 表与按 key 的调度都编译进二进制。我们的去审查 EXL3 包正是引擎没有代码的布局（`wo_b` 是
  plain `.weight` 而不是 `.trellis`、engram 片带 F8 dtype、bf16 副本遮蔽 f32 主体），运行时没有任何兜底：
  只能写 Zig 代码、重编 fatbin 与三个二进制。下一个变体请按同样量级做计划。

## 选哪套，为什么

| | TP2（两机） | TP4（四机） |
|---|---|---|
| 引擎 | **TensorFold Zig**（`tensorfold-native`） | **SGLang**（生产）或 TensorFold Zig |
| 权重 | EXL3 2.9bpw 去审查重打包 | 原生 FP8 + FP4 routed experts（heretic） |
| 为什么 | **唯一装得下的组合**：477 GB 的 FP8 摊到 2 个 rank = 238 GB/rank，超过 GB10 的 128 GB 统一内存；EXL3 2.9bpw 约 99 GB/rank 才进得去，而能吃 EXL3 的引擎只有 Zig（Python 线已冻结）。 | FP8 四路切得下，SGLang 才成立。Zig TP4 能跑，但要付环上通信税（见下）。 |
| KV 池 | 1M positions（`POOL=1048576`、`PARALLEL=4`） | SGLang：5,382,912 token（`mem-fraction-static 0.85` + RoCEnante mesh）。Zig TP4：8M positions（`TF_DS_WINDOW=8388608`、`PARALLEL=16`） |
| decode c1（sparkDash，prose 400 tok） | 66.8 tok/s | SGLang 73.0 tok/s |
| prefill（sparkDash，4k/8k/16k/32k） | 1,779 / 1,912 / 1,965 / 1,994 tok/s | SGLang **5,710 / 5,212 / 5,266 / 4,685 tok/s** |
| 定位 | TP2 的生产栈 | 2026-10-26 起 TP4 的生产栈是 MiaAI-Lab 四机 TensorFold 原生 FP8 栈；SGLang 是等价备选；Zig TP4 作为高 decode 的实验栈保留 |

完整推理与每个判断背后的数字：[docs/selection.md](docs/selection.md)。

## 仓库结构

```
dev/tp2-tensorfold-zig/    TP2 Zig 启动/停止脚本、rank-cache 与 token-map 生成器、
                            验收探针、上游兼容补丁、我们自己的 4 个引擎 commit
dev/tp4-tensorfold-zig/     TP4 Zig 启动/停止脚本、环上路由诊断脚本
dev/tp4-sglang/             TP4 SGLang 启动脚本 + env 示例（上游树不重分发）
dev/tp4-tensorfold-mia/     TP4 MiaAI-Lab TensorFold（原生 FP8）：我们的引擎补丁 + --learn 接线
docs/                       选型、逐栈 bring-up、排障索引
```

引擎本体**不重分发**：它是 `bertholomus/TensorFold` 分支 `deepseek-v41-zig-tp4` 编出的 Zig 原生二进制。
我们把自己的 4 个 commit 以 `git format-patch` 形式放在 `dev/tp2-tensorfold-zig/engine-patches/`
（叠在上游 `5940208` 之上）——它们存在的唯一原因就是**引擎在 CUDA 侧不做运行期编译**：去审查重打包的权重，
必须先让它编译进去的 dtype 表和 linear 路径认识这个布局，再重编 fatbin 与三个二进制。SGLang 栈来自
`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`，我们给的是 env 文件 + 两处启动脚本修复。
`dev/tp4-tensorfold-mia/` 放的是我们给 MiaAI-Lab 四机栈打的引擎补丁——三个 commit，让 `--learn`
（前缀快照落盘 NVMe）在 `dsv41` 线上真正跑起来；它们叠在 MiaAI-Lab 补丁系列之上，不是裸 `5940208`。

## 四套栈各一段

### TP2 — TensorFold Zig + EXL3 2.9bpw

```bash
# 只在 rank-0 那台跑一次；幂等，内部 worker-first
bash dev/tp2-tensorfold-zig/start-dsv41-zig-tp2-0102.sh
python3 dev/tp2-tensorfold-zig/verify-dsv41-zig-tp2.py   # 直连 + 经 llm-proxy 双路探针
```

- rank-0 常驻约 99 GB 的 EXL3 权重；rank-cache **必须预生成**（`node_inputs.sh`，约 107 GB/rank）——
  引擎只读不自建，缺了直接 `NoRankCache` 死。
- **KV 池只有 1M positions**，这是这套栈最硬的限制。抬 `POOL` 每 1M positions 约吃 0.9 GiB 统一内存，
  还要给长 prompt 的 prefill 峰值留余量。
- API 就绪约 102–123 s。支持两组独立的机器对（0102 与 0304），0304 那对用的是每台**另一个** HCA 腿，
  所以 `NCCL_IB_HCA` / `TF_RDMA_DEVICES` 在那一版是按 rank 下发的。
- 权重目录里必须有 `chat_template.jinja`，且必须是 Mia 那一版模板：Zig 的 Jinja 不支持 `**kwargs`，
  vLLM 风格模板会 `TemplateFailed`，表现是**静默返回 1 个 token**。

细节与坑：[docs/tp2-tensorfold-zig.md](docs/tp2-tensorfold-zig.md)。

### TP4 — SGLang（四机首选）

```bash
bash dev/tp4-sglang/start-dsv41-tp4.sh serve    # 互斥清理 → worker-first 起栈 → 就绪等待 → smoke
```

- 上游树在 `64b02de`；镜像 `dsv41-4x-spark:canary-roce` 由 `Dockerfile.canary-roce` 构建。
- 我们的贡献是配置 + 两处启动脚本修复，最要紧的一条：**关掉 HiCache 时必须同时摘掉 `/hicache` 挂载**，
  否则每台 worker 容器都卡在 `Created` 状态，报 `mkdir /root/dsv41-hicache: file exists`（残留 NFS 挂载点），
  而 head 看起来完全正常。
- 实测按 **2026-10-25 生产配置**（RoCEnante mesh、`CHUNKED_PREFILL_SIZE=8192`、`mem-fraction-static 0.85`）：
  KV 池 **5,382,912 token**，池外 `available_gpu_mem` 9.77 GB，decode **73.0 tok/s c1**，
  prefill **5,710 / 5,212 / 5,266 / 4,685 tok/s**（4k / 8k / 16k / 32k）。
  （更早的 HiCache 关 + patched NCCL 配置是池 6,781,184、prefill 2,493–4,921 —— **引用必须带配置档**，
  两代 prefill 差到 2×。）
- Engram 要按节点打包（约 48 GB/节点，~100 s/层）——Zig 线直读 checkpoint、不产生这个目录，
  所以从 Zig 切回 SGLang 每次都要重打。

细节、标定表与 sparkDash 数据：[docs/tp4-sglang.md](docs/tp4-sglang.md)。

### TP4 — MiaAI-Lab TensorFold（原生 FP8）—— 我们现在的 TP4 生产栈

引擎 `bertholomus/TensorFold` 的 `5940208` + MiaAI-Lab 补丁系列，四机、原生 FP8 权重。我们跑的配置：
`--pool 12582912`（12M positions，起栈 device memory 99.21 GB）、`--prompt-cache-gib 6` 给前缀快照池
（CUDA 默认 0.5 GiB ≈ 80 个 state，几个长会话一小时就能轮一遍）、`--learn --learn-gib 200` 落在本机 NVMe。

- **`--learn` 在这条线上原本完全不工作。** 三个缺口全在引擎侧，且都与 family 无关：`dsv41` 的快照只带
  6.25 MiB 的 slot ring、丢了 extent 的压缩行；server 给的 shared cut 是纯 token 数，与 keep 时被
  chunk 对齐过的 state 永远哈希不上，于是「shared」这个开关恒 false，而 `resumed at` 看起来一切正常；
  imprint 索引只在 Metal/GLM 的 host 里建，`learn()` 第一行 `imprint orelse` 就返回。三个补丁在
  `dev/tp4-tensorfold-mia/engine-patches/`。
- 首个实测结果：`learned 2048 tokens to disk`，四台各一份 8,507,488 B 文件；随后同一条 prompt 仍
  `resumed at 2048`、`cached_tokens 2048`——命中率零回归。跨重启读回已实现，但还没在生产日志里观察到，
  文档里如实标了。
- 换引擎版本 = 三个只读单文件挂载（`tensorfold-native`、`tf-dsv41-lanes`、`tf-dsv41-rank-cache` 只链
  `libc`），不重建镜像，回滚只改一个路径。
- 单独一行提醒的坑：**别用 `planned`** 来对齐 cut。这条线的 `starts` 是空的（`info.prefill_step == 0`），
  每个 mark 都会被 floor 到 0、全部被过滤，整栈退化成 100% 全量 prefill，而日志里根本不会出现 `learned`。

机制、体积账、完整故障链与验收清单：[docs/prompt-cache-learn.md](docs/prompt-cache-learn.md)；接线与编译：
[dev/tp4-tensorfold-mia/README.md](dev/tp4-tensorfold-mia/README.md)。

### TP4 — TensorFold Zig（decode 最快，附条件多）

- 4 rank 走 2D 切分：`g^2` 对角对（01↔03、02↔04）在光纤环上**不相邻**，所以对角流量改走「四机
  all-gather + 取 peer 分片」，不再是点对点（我们的 commit `5727ac4`），全程仍在 200G RoCE 上。
- 数据面：补丁版 switchless NCCL（`LD_PRELOAD`）、`NCCL_IB_HCA` 双口、**每机动态**
  `NCCL_IB_GID_INDEX`（按本机 RoCE 地址算）、`ALGO=Ring`、`NCHANNELS=4`。
- `TF_DS_WINDOW=8388608`（8M positions）、`PARALLEL=16`；rank-cache 必须 `--parity` 构建。
- 实测 decode 133.3 tok/s c1（sparkDash structured）/ 98.1 tok/s（直连探针，prose），
  prefill 1423–1582 tok/s —— 比 TP2 **低 15–20 %**，这就是 all-gather 的代价。
- 踩过的故障：四台里三台 worker 报 `error: ShortRopeTable` 退出，而 head 的 `GET /v1/models`
  仍然 200。**`/v1/models` 200 不代表 Zig 栈还活着**——要看四台容器 + 一次真实 completion。

细节：[docs/tp4-tensorfold-zig.md](docs/tp4-tensorfold-zig.md)。

## 测法

除标注「直连探针」的以外，所有吞吐都来自 [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 内置压测
（`GET /api/sparks/<id>/llm/bench` 与 `/llm/prefill-bench`）。sparkDash 固定 400 token 生成、用自己的
prose/structured 提示集，所以**跨栈对比只在同一 prompt 类型内有效**——表里都注明了类型。直连探针用的是
450 token 计数提示（TP4 Zig）和 9,009 token prefill。

## 老实说的局限

- TP2 的 1M positions 池偏小，长上下文多会话会排队。
- TP4 Zig 的 prefill 低于 TP2（对角 all-gather），8M 池是真的，但不是免费的。
- SGLang TP4 的 decode c1（sparkDash prose 73 tok/s）和 TP2 Zig 同一档；它赢在 prefill 和池子。
- TP4 Zig 是实验栈：可跑、可复现，但我们还不敢说它「比 SGLang 稳」。

## 致谢

见 [CREDITS.md](CREDITS.md)：引擎是 TensorFold（bertholomus、ashhart），EXL3 重打包与 1017 起点来自
MiaAI-Lab，去审查权重是 pqhaz 的方向消融，SGLang 四机栈是 MiaAI-Lab 的配方，switchless NCCL 补丁是
alexellis 的。我们这部分是两机/四机的胶水、EXL3 dense 回退的引擎改造、环拓扑结论，以及这些实测数字。

我们的文件用 Apache-2.0；上游组件各自保留许可——见 [NOTICE](NOTICE) 与
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。