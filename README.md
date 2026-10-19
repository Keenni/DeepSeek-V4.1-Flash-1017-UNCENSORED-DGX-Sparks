# DeepSeek-V4.1-Flash-1017 (uncensored) on DGX Sparks — three working stacks

Deployment recipes, engine patches and measurements for serving **DeepSeek-V4.1-Flash-1017** in its
uncensored form on NVIDIA DGX Spark (GB10) nodes: two-node **TP2** with TensorFold's Zig engine over an
EXL3 2.9 bpw repack, and four-node **TP4** with either the same Zig engine or SGLang over the native
FP8/FP4 checkpoint.

Everything here ran on four GB10 nodes in a private lab (a **switchless** 200G fibre *ring*, no switch) and was
verified with real requests. The scripts are the ones we actually run; the numbers are the ones we actually
measured. [中文说明 →](README.zh-CN.md)

**Keywords:** switchless TP4 · switchless fibre ring · DGX Spark TP4 · TensorFold Zig (compiled binary —
patched by us per weight variant) · SGLang TP4 · EXL3 2.9 bpw · native FP8/FP4 · DeepSeek-V4.1-Flash-1017
uncensored.

Two things worth knowing before you read further:

* **Both TP4 stacks here are switchless-TP4**: four nodes, four direct fibre links forming a ring, no switch
  in the fabric. The Zig line is where that matters most — its 2D split wants diagonal pairs that a ring does
  not have, so the diagonal goes through a four-node all-gather (our patch).
* **The Zig engine is a binary, and it has no idea about your weights.** The engine ships as a compiled
  `tensorfold-native` / `tf-dsv41-lanes` / `tf-dsv41-rank-cache` set. Every weight variant that is not the
  upstream pack (our uncensored EXL3 repack has plain `wo_b` tensors, F8 dtypes in the Engram shards, and
  bf16 duplicates that shadow the f32 mains) breaks it at load time, and the only fix is to **patch the engine
  source and rebuild the binary** — which is exactly what our four commits do. Plan for that; it is not a
  config change.

## Which stack, and why

| | TP2 (2 nodes) | TP4 (4 nodes) |
|---|---|---|
| Engine | **TensorFold Zig** (`tensorfold-native`) | **SGLang** (production) or TensorFold Zig |
| Weights | EXL3 2.9 bpw uncensored repack | native FP8 + FP4 routed experts (heretic) |
| Why | the only combination that **fits**: 477 GB of FP8 over 2 ranks is 238 GB/rank, beyond a GB10's 128 GB unified memory. EXL3 2.9 bpw is ~99 GB/rank — and only the Zig engine reads EXL3. | FP8 fits 4 ways, so SGLang becomes usable; the Zig TP4 line works but pays a ring tax (see below). |
| KV pool | 1M positions (`POOL=1048576`, `PARALLEL=4`) | SGLang: 6.78M tokens (`mem-fraction-static 0.85`). Zig TP4: 8M positions (`TF_DS_WINDOW=8388608`, `PARALLEL=16`) |
| Decode c1 (sparkDash, prose 400 tok) | 66.8 tok/s | SGLang 69.4 tok/s |
| Prefill 32k | 1994 tok/s | SGLang 4921 tok/s |
| State | production for TP2 | SGLang is the production TP4 stack; Zig TP4 is kept as a fast-decode experiment |

Full reasoning, including the numbers behind each verdict: [docs/selection.md](docs/selection.md).

## What is in here

```
dev/tp2-tensorfold-zig/    TP2 Zig deploy scripts, rank-cache/token-map builders, verify probe,
                            upstream-compat patches and our four engine commits
dev/tp4-tensorfold-zig/     TP4 Zig deploy scripts, ring diagnostics routes
dev/tp4-sglang/             TP4 SGLang deploy script + env example (upstream tree not redistributed)
docs/                       selection, per-stack bring-up notes, troubleshooting index
```

The engine itself is **not** redistributed: it is `bertholomus/TensorFold` (branch `deepseek-v41-zig-tp4`)
built as a Zig native binary. We ship our four commits as `git format-patch` files under
`dev/tp2-tensorfold-zig/engine-patches/` (they apply on top of upstream `5940208`) — and they exist because
this is a **binary**: the uncensored repack only loads after patching the engine's safetensors dtypes, its
linear path and its rank-cache writer, then rebuilding. The SGLang stack is
`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` with our environment file and two small start-script fixes.

## The three stacks in one page

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
* Measured (HiCache off, `mem-fraction-static 0.85`): KV pool **6,781,184 tokens**, 18.01 GB left outside
  the pool, ready in 4.3 min from cold compile caches.
* Engram has to be packed per node (~48 GB/node, ~100 s/layer) — the Zig line reads the checkpoint
  directly and never produces that directory, so switching back to SGLang always requires a repack.

Details, calibration table and the sparkDash numbers: [docs/tp4-sglang.md](docs/tp4-sglang.md).

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
* SGLang TP4 decode c1 (~69 tok/s on sparkDash prose) is close to TP2 Zig; the win is prefill and pool.
* The TP4 Zig line is a lab result: it works, it is reproducible, and we would not call it "more stable than
  SGLang" yet.

## Credits

See [CREDITS.md](CREDITS.md) — the engine is TensorFold (bertholomus, ashhart), the EXL3 repack and the
1017 starting points are MiaAI-Lab's, the uncensored weights are pqhaz's directional ablation, the SGLang
stack is MiaAI-Lab's four-node recipe, and the switchless NCCL patch is alexellis'. Our part is the
two- and four-node glue, the EXL3 dense-fallback engine work, the ring-topology findings, and the numbers.

Licensed under Apache-2.0 (our files); upstream components keep their own licences — see
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).