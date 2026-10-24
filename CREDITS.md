# Credits

Everything here stands on other people's work. Chain, outermost first.

## Model

* **DeepSeek-V4.1-Flash-1017** — DeepSeek (deepseek-ai). FP8 weights, FP4 routed experts, Engram tables,
  DSpark/MTP draft layers, 1M context.
* **Uncensored variant (this repo's TP4 checkpoint)** — `pqhaz/DeepSeek-V4.1-Flash-1017-heretic`:
  a directional ablation of the refusal direction applied to 38 `layers.N.attn.wo_b.weight` tensors
  (strength 2.5, blocks 2–39), everything else byte-identical to the base release, plus a
  `chat_template.jinja`. Method: [Heretic](https://github.com/p-e-w/heretic).
* **EXL3 2.9 bpw repack (this repo's TP2 checkpoint)** — derived from LibertAI's
  `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` (196.3 GiB, 43 shards); the uncensored repack is
  `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`, whose `wo_b` tensors carry **dealignai's own
  CRACK ablation re-done at the target precision** (rank-1 per layer on layers 10–36, attention `wo_b` and
  shared-expert `w2`), *not* pqhaz's heretic edit — the two stacks in this repo run two different
  ablations. Quantisation format: **ExLlamaV3**
  (turboderp). `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw` was the reference for what a mixed-bits
  EXL3 pack of this model looks like.

## Inference engines

* **TensorFold** (Zig native CUDA engine) — [ashhart/TensorFold](https://github.com/ashhart/TensorFold).
  The DeepSeek-V4.1 support branch we build is **`bertholomus/TensorFold`, branch `deepseek-v41-zig-tp4`,
  commit `5940208`**. Kernel kit (`kit/aot`, RoPE tables, vision `gate_bias`, cubins) comes from
  `bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10`.
* **SGLang** — [sglang](https://github.com/sgl-project/sglang) (Apache-2.0). Base image
  `lmsysorg/sglang:dev-dsv41`; the DeepSeek-V4.1 kernels (`b12x`, `b12x_next`, RoCEnante) come from
  local-inference-lab's work as vendored by the stack below.

## Reference stacks we started from

* **`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`** — the four-node SGLang recipe we run in TP4 production:
  `start.sh` / `start-tp4*.sh`, `boot.py`, `adapter/`, RoCEnante overlay, the `canary-roce` Dockerfile. Our
  `dev/tp4-sglang/start-dsv41-tp4.sh` is a **modified copy** of their launcher (see NOTICE for licensing).
  Their `sparkDash` is the benchmark behind every number we quote. Copyright (C) 2026 zurih, AGPL-3.0.
* **`MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`** — the two-node Zig recipe, and the source of the
  **`chat_template.jinja`** we ship (md5 `dcf25a1f…`), which is the variant that works with TensorFold's
  Jinja subset.
* **`bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10`** — TP2 recipe, `node_inputs.sh`, kernel kit.
* **`tonyd2wild`**, `runs/2026-10-17-1017-tensorfold/patches/` — the four compatibility patches shipped in
  `dev/tp2-tensorfold-zig/patches/` (Triton-variant CUDA launch, prompt-indexer row blocks, `TF_RDMA_GID`,
  server seed/top-k).
* **`alexellis`** — the switchless-ring NCCL patch (skip-tree / advertise-all-listener-GIDs) that lets a ring
  behave like a fabric.

## Our part (so the boundary is clear)

* Two-node and four-node deploy glue: worker-first start/stop, memory gates, rank-cache/token-map
  preparation, verify probes, per-rank HCA selection for a pair wired on mixed legs.
* **Engine work on the EXL3 repack** (commits `69158a0`, `7def4b4`): dense (non-trellis) linear fallback in
  the Zig path, `dn` cache entries, dense cuBLAS GEMM, F8 safetensors dtypes so the Engram shards can even be
  opened, the `shared-w2` extra shard, and the bf16→f32 widen that bf16 copies in the Engram shards require.
* **Ring-topology findings on TP4** (commits `b2252e0`, `5727ac4`): why diagonal RoCE p2p cannot work on a
  ring, why TCP-over-fibre collapses decode, and the four-node all-gather that replaces the diagonal exchange.
* **SGLang stack**: the environment file, the HiCache-off-must-remove-the-mount fix, the NFS-helper gate, the
  pool calibration with HiCache off, and the claim that a Zig head answering `/v1/models` proves nothing.
* All measurements quoted here.

Not ours: the model, the quantisation formats, the engines, the kernels, the NCCL patch, the upstream
launchers.

## Corrections

If an attribution here is wrong or incomplete, open an issue — we would rather fix the credit than keep it.