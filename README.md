# DeepSeek-V4.1-Flash-1017 (uncensored) on DGX Sparks — four working stacks

Deployment recipes, engine patches and measurements for serving **DeepSeek-V4.1-Flash-1017** in its
uncensored form on NVIDIA DGX Spark (GB10) nodes: two-node **TP2** with TensorFold's Zig engine over an
EXL3 2.9 bpw repack, and four-node **TP4** with either the same Zig engine or SGLang over the native
FP8/FP4 checkpoint.

Everything here ran on four GB10 nodes in a private lab (a **switchless** 200G fibre *ring*, no switch) and was
verified with real requests. The scripts are the ones we actually run; the numbers are the ones we actually
measured. [中文说明 →](README.zh-CN.md)

**Keywords:** switchless TP4 · switchless fibre ring · DGX Spark TP4 · TensorFold Zig (AOT, no CUDA JIT —
a new weight layout means engine source patches) · SGLang TP4 · EXL3 2.9 bpw · native FP8/FP4 ·
DeepSeek-V4.1-Flash-1017 uncensored.

Two things worth knowing before you read further:

* **Both TP4 stacks here are switchless-TP4**: four nodes, four direct fibre links forming a ring, no switch
  in the fabric. The Zig line is where that matters most — its 2D split wants diagonal pairs that a ring does
  not have, so the diagonal goes through a four-node all-gather (our patch).
* **The Zig engine has no runtime compilation on CUDA, so a weight layout it does not know is a source
  change.** SGLang and vLLM are Python engines: a new quantisation or layout is usually a new class, and the
  kernels are Triton-JIT / FlashInfer-autotune compiled on first use. TensorFold is AOT end to end — its CUDA
  kernels are prebuilt cubins loaded with `cuModuleLoadData` (runtime compilation exists only on the
  Metal/macOS path), and the checkpoint loader's dtype table and key dispatch are compiled into the binary.
  Our uncensored EXL3 repack is a layout the engine had no code for (plain `wo_b` instead of `.trellis`, F8
  dtypes in the Engram shards, bf16 copies shadowing the f32 mains), so there was nothing at runtime to fall
  back on: we wrote the Zig code, rebuilt the fatbins and the three binaries. Expect the same for the next
  variant.

## Which stack, and why

| | TP2 (2 nodes) | TP4 (4 nodes) |
|---|---|---|
| Engine | **TensorFold Zig** (`tensorfold-native`) | **SGLang** (production) or TensorFold Zig |
| Weights | EXL3 2.9 bpw uncensored repack | native FP8 + FP4 routed experts (heretic) |
| Why | the only combination that **fits**: 477 GB of FP8 over 2 ranks is 238 GB/rank, beyond a GB10's 128 GB unified memory. EXL3 2.9 bpw is ~99 GB/rank — and only the Zig engine reads EXL3. | FP8 fits 4 ways, so SGLang becomes usable; the Zig TP4 line works but pays a ring tax (see below). |
| KV pool | 1M positions (`POOL=1048576`, `PARALLEL=4`) | SGLang: 5.38M tokens (5,382,912; `mem-fraction-static 0.85` + RoCEnante mesh). Zig TP4: 8M positions (`TF_DS_WINDOW=8388608`, `PARALLEL=16`) |
| Decode c1 (sparkDash, prose 400 tok) | 66.8 tok/s | SGLang 73.0 tok/s |
| Prefill (sparkDash, 4k/8k/16k/32k) | 1,779 / 1,912 / 1,965 / 1,994 tok/s | SGLang **5,710 / 5,212 / 5,266 / 4,685 tok/s** |
| State | production for TP2 | production TP4 since 2026-10-26 is the MiaAI-Lab four-node TensorFold native-FP8 stack; SGLang is the equivalent fallback; Zig TP4 is kept as a fast-decode experiment |

Full reasoning, including the numbers behind each verdict: [docs/selection.md](docs/selection.md).

## What is in here

```
dev/tp2-tensorfold-zig/    TP2 Zig deploy scripts, rank-cache/token-map builders, verify probe,
                            upstream-compat patches and our four engine commits
dev/tp4-tensorfold-zig/     TP4 Zig deploy scripts, ring diagnostics routes
dev/tp4-sglang/             TP4 SGLang deploy script + env example (upstream tree not redistributed)
dev/tp4-tensorfold-mia/     TP4 MiaAI-Lab TensorFold (native FP8): our engine patches + --learn wiring
docs/                       selection, per-stack bring-up notes, troubleshooting index
```

The engine itself is **not** redistributed: it is `bertholomus/TensorFold` (branch `deepseek-v41-zig-tp4`)
built as a Zig native binary. We ship our four commits as `git format-patch` files under
`dev/tp2-tensorfold-zig/engine-patches/` (they apply on top of upstream `5940208`) — and they exist because
the engine compiles nothing at runtime on CUDA: the uncensored repack only loads after its compiled-in dtype
table and linear path are taught the new layout, then the binaries are rebuilt. The SGLang stack is
`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` with our environment file and two small start-script fixes.
`dev/tp4-tensorfold-mia/` holds our engine patches for the MiaAI-Lab four-node stack — three commits that
make `--learn` (prompt-cache snapshots persisted to NVMe) actually run on the `dsv41` line; they apply on
top of the MiaAI-Lab patch series, not on a bare `5940208`.

## The four stacks in one page

### TP2 — TensorFold Zig + EXL3 2.9 bpw

```bash
# on the rank-0 node only; idempotent, worker-first inside
bash dev/tp2-tensorfold-zig/start-dsv41-zig-tp2-0102.sh
python3 dev/tp2-tensorfold-zig/verify-dsv41-zig-tp2.py   # direct + proxy probes
```

* rank-0 GPU memory ~99 GB of EXL3 weights; rank-cache is **prebuilt** (`node_inputs.sh`, ~107 GB/rank) —
  the engine only ever *reads* it and dies with `NoRankCache` otherwise.
* KV pool is **1M positions** — the sharpest limitation of this stack. Raising `POOL` costs ≈0.9 GiB per
  1M positions of unified memory, and the long-prompt prefill peak has to fit next to it.
* Ready in ~102–123 s. Two independent node pairs are supported (0102 and 0304); the 0304 pair is wired
  on the *other* HCA leg of each node, so `NCCL_IB_HCA` / `TF_RDMA_DEVICES` are per-rank there.
* `chat_template.jinja` must be present in the weight directory, and it must be the Mia-variant template:
  the Zig engine's Jinja does not support `**kwargs`, so the vLLM-style template fails with `TemplateFailed`
  and you silently get 1-token answers.

Details and pitfalls: [docs/tp2-tensorfold-zig.md](docs/tp2-tensorfold-zig.md).

### TP4 — SGLang (recommended for 4 nodes)

```bash
bash dev/tp4-sglang/start-dsv41-tp4.sh serve    # mutual-exclusion cleanup, worker-first, smoke test
```

* Upstream tree at `64b02de`; image `dsv41-4x-spark:canary-roce` built from `Dockerfile.canary-roce`.
* Our contribution is configuration + two start-script fixes, notably: **with HiCache disabled you must also
  remove the `/hicache` bind mount**, otherwise every worker container dies in `Created` state with
  `mkdir /root/dsv41-hicache: file exists` (a stale NFS mount point) while the head still looks fine.
* Measured on the 2026-10-25 production config (RoCEnante mesh, `CHUNKED_PREFILL_SIZE=8192`,
  `mem-fraction-static 0.85`): KV pool **5,382,912 tokens**, 9.77 GB `available_gpu_mem` left outside the
  pool, decode **73.0 tok/s c1**, prefill **5,710 / 5,212 / 5,266 / 4,685 tok/s** at 4k / 8k / 16k / 32k.
  (The earlier HiCache-off / patched-NCCL config measured pool 6,781,184 and prefill 2,493–4,921 — quote the
  dated config, the two generations differ by up to 2× on prefill.)
* Engram has to be packed per node (~48 GB/node, ~100 s/layer) — the Zig line reads the checkpoint
  directly and never produces that directory, so switching back to SGLang always requires a repack.

Details, calibration table and the sparkDash numbers: [docs/tp4-sglang.md](docs/tp4-sglang.md).

### TP4 — MiaAI-Lab TensorFold (native FP8) — our current production TP4

Engine `bertholomus/TensorFold` at `5940208` + the MiaAI-Lab series, four nodes, native FP8 weights.
Serving config we run: `--pool 12582912` (12M positions, 99.21 GB of device memory at start),
`--prompt-cache-gib 6` for the prefix snapshot store (the CUDA default is 0.5 GiB ≈ 80 states, which a
few long sessions cycle through in an hour), and `--learn --learn-gib 200` on local NVMe.

* **`--learn` did not work on this line at all.** All three gaps are engine-side and none is
  family-specific: the `dsv41` snapshot only carried its 6.25 MiB slot ring and dropped the extent's
  compressed rows; the server's shared cut (plain token counts) never hashed equal to the chunk-floored
  state it kept, so the "shared" flag that gates learning stayed false while `resumed at` looked healthy;
  and the imprint index was built only in the Metal/GLM host, so `learn()` returned on its first line.
  Three patches, `dev/tp4-tensorfold-mia/engine-patches/`.
* First measured result: `learned 2048 tokens to disk`, one file per rank of 8,507,488 B, next identical
  request still `resumed at 2048` with `cached_tokens 2048` — no hit-rate regression. Cross-restart
  read-back is implemented but not yet observed in a production log; the doc says so.
* Engine upgrades are three read-only binary mounts (`tensorfold-native`, `tf-dsv41-lanes`,
  `tf-dsv41-rank-cache` link only `libc`) — no image rebuild, and rollback is one path.
* Trap worth its own line: do **not** set the prompt-cache rules to `planned` to align the cuts on this
  line. `starts` is empty here (`info.prefill_step == 0`), so every mark floors to 0, every state is
  filtered, and the whole stack degrades to 100 % re-prefill with no `learned` line to warn you.

Mechanism, sizes, the whole failure chain and the verification checklist:
[docs/prompt-cache-learn.md](docs/prompt-cache-learn.md) · wiring and build:
[dev/tp4-tensorfold-mia/README.md](dev/tp4-tensorfold-mia/README.md).

### TP4 — TensorFold Zig (fastest decode, extra caveats)

* 2D split over 4 ranks: the `g^2` diagonal pairs (01↔03, 02↔04) are **not** physically adjacent on a
  fibre ring, so all diagonal traffic goes through a four-node all-gather instead of point-to-point
  (our commit `5727ac4`). Everything stays on 200G RoCE.
* Data plane: patched switchless NCCL (`LD_PRELOAD`), `NCCL_IB_HCA` on both ports, per-node
  `NCCL_IB_GID_INDEX` computed from the node's own RoCE addresses, `ALGO=Ring`, `NCHANNELS=4`.
* `TF_DS_WINDOW=8388608` = 8M positions, `PARALLEL=16`; rank-cache must be built with `--parity`.
* Measured decode 133.3 tok/s c1 (sparkDash structured) / 98.1 tok/s (direct probe, prose), prefill
  1423–1582 tok/s — i.e. **15–20 % below TP2 on prefill**, the price of the all-gather.
* Known failure we hit: three of four workers exited with `error: ShortRopeTable` while the head kept
  answering `GET /v1/models` with 200. A 200 from `/v1/models` does **not** mean the Zig stack is alive —
  check all four containers plus one real completion.

Details: [docs/tp4-tensorfold-zig.md](docs/tp4-tensorfold-zig.md).

## Measurement method

All throughput numbers come from [sparkDash](https://github.com/MiaAI-Lab/sparkDash)'s built-in benchmark
(`GET /api/sparks/<id>/llm/bench` and `/llm/prefill-bench`) unless marked "direct probe". sparkDash uses a
400-token generation with its own prose/structured prompt sets, so **cross-stack comparisons are only valid
within the same prompt type** — the tables say which. Direct probes used a 450-token counting prompt
(TP4 Zig) and a 9,009-token prefill.

## Known limits, honestly

* TP2's 1M-position KV pool is small; long multi-session work will queue.
* TP4 Zig prefill is below TP2 because of the diagonal all-gather; the 8M pool is real but it is not free.
* SGLang TP4 decode c1 (73 tok/s on sparkDash prose) is in the same band as TP2 Zig; the win is prefill and the pool.
* The TP4 Zig line is a lab result: it works, it is reproducible, and we would not call it "more stable than
  SGLang" yet.

## Credits

See [CREDITS.md](CREDITS.md) — the engine is TensorFold (bertholomus, ashhart), the EXL3 repack and the
1017 starting points are MiaAI-Lab's, the uncensored weights are pqhaz's directional ablation, the SGLang
stack is MiaAI-Lab's four-node recipe, and the switchless NCCL patch is alexellis'. Our part is the
two- and four-node glue, the EXL3 dense-fallback engine work, the ring-topology findings, and the numbers.

Licensed under Apache-2.0 (our files); upstream components keep their own licences — see
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).