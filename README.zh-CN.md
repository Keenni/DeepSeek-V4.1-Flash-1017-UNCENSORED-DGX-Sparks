# DeepSeek-V4.1-Flash-1017（去审查）在 DGX Spark 上的三套可用栈

把 **DeepSeek-V4.1-Flash-1017** 的去审查版跑在 NVIDIA DGX Spark（GB10）上的部署件、引擎补丁与实测数据：
两机 **TP2** 走 TensorFold 的 Zig 引擎 + EXL3 2.9bpw 重打包；四机 **TP4** 走同一个 Zig 引擎，或者走
SGLang + 原生 FP8/FP4 权重。

这里的脚本是我们在跑的脚本，数字是我们实测的数字。测试环境是四台 GB10 组成的 **无交换机 200G 光纤环**
（不是全互联），所有请求都是真实推理。[English →](README.md)

## 选哪套，为什么

| | TP2（两机） | TP4（四机） |
|---|---|---|
| 引擎 | **TensorFold Zig**（`tensorfold-native`） | **SGLang**（生产）或 TensorFold Zig |
| 权重 | EXL3 2.9bpw 去审查重打包 | 原生 FP8 + FP4 routed experts（heretic） |
| 为什么 | **唯一装得下的组合**：477 GB 的 FP8 摊到 2 个 rank = 238 GB/rank，超过 GB10 的 128 GB 统一内存；EXL3 2.9bpw 约 99 GB/rank 才进得去，而能吃 EXL3 的引擎只有 Zig（Python 线已冻结）。 | FP8 四路切得下，SGLang 才成立。Zig TP4 能跑，但要付环上通信税（见下）。 |
| KV 池 | 1M positions（`POOL=1048576`、`PARALLEL=4`） | SGLang：6,781,184 token（`mem-fraction-static 0.85`）。Zig TP4：8M positions（`TF_DS_WINDOW=8388608`、`PARALLEL=16`） |
| decode c1（sparkDash，prose 400 tok） | 66.8 tok/s | SGLang 69.4 tok/s |
| prefill 32k | 1994 tok/s | SGLang 4921 tok/s |
| 定位 | TP2 的生产栈 | SGLang 是 TP4 的生产栈；Zig TP4 作为高 decode 的实验栈保留 |

完整推理与每个判断背后的数字：[docs/selection.md](docs/selection.md)。

## 仓库结构

```
dev/tp2-tensorfold-zig/    TP2 Zig 启动/停止脚本、rank-cache 与 token-map 生成器、
                            验收探针、上游兼容补丁、我们自己的 4 个引擎 commit
dev/tp4-tensorfold-zig/     TP4 Zig 启动/停止脚本、环上路由诊断脚本
dev/tp4-sglang/             TP4 SGLang 启动脚本 + env 示例（上游树不重分发）
docs/                       选型、逐栈 bring-up、排障索引
```

引擎本体**不重分发**：它是 `bertholomus/TensorFold` 分支 `deepseek-v41-zig-tp4` 编出的 Zig 原生二进制。
我们把自己的 4 个 commit 以 `git format-patch` 形式放在 `dev/tp2-tensorfold-zig/engine-patches/`
（叠在上游 `5940208` 之上）。SGLang 栈来自 `MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`，我们给的是
env 文件 + 两处启动脚本修复。

## 三套栈各一段

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
- 实测（HiCache 关、`mem-fraction-static 0.85`）：KV 池 **6,781,184 token**，池外余量 18.01 GB，
  编译缓存冷启到就绪 4.3 分钟。
- Engram 要按节点打包（约 48 GB/节点，~100 s/层）——Zig 线直读 checkpoint、不产生这个目录，
  所以从 Zig 切回 SGLang 每次都要重打。

细节、标定表与 sparkDash 数据：[docs/tp4-sglang.md](docs/tp4-sglang.md)。

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
- SGLang TP4 的 decode c1（sparkDash prose 约 69 tok/s）和 TP2 Zig 接近；它赢在 prefill 和池子。
- TP4 Zig 是实验栈：可跑、可复现，但我们还不敢说它「比 SGLang 稳」。

## 致谢

见 [CREDITS.md](CREDITS.md)：引擎是 TensorFold（bertholomus、ashhart），EXL3 重打包与 1017 起点来自
MiaAI-Lab，去审查权重是 pqhaz 的方向消融，SGLang 四机栈是 MiaAI-Lab 的配方，switchless NCCL 补丁是
alexellis 的。我们这部分是两机/四机的胶水、EXL3 dense 回退的引擎改造、环拓扑结论，以及这些实测数字。

我们的文件用 Apache-2.0；上游组件各自保留许可——见 [NOTICE](NOTICE) 与
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。