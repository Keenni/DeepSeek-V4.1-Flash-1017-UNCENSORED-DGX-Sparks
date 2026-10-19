# TP2 — TensorFold Zig engine + EXL3 2.9 bpw uncensored

Two GB10 nodes, one rank each. `start-dsv41-zig-tp2-0102.sh` for the pair whose fibre link is on each
node's **f0** leg (`10.0.0.0/24`), `start-dsv41-zig-tp2-0304.sh` for the pair wired on the mixed legs
(`10.0.2.0/24`, f1 on one node and f0 on the other). Both take ~102–123 s to a ready API.

## What you need before the first start

1. **Weights** — EXL3 2.9 bpw uncensored repack in one directory: 43 EXL3 shards (`of-00043`) *plus*
   DeepSeek's original 1017 shards 47 and 48 (`of-00048`, the Engram tables) *plus* `chat_template.jinja`.
   The index pins the Engram keys to the `of-00048` filenames, so those two files must keep exactly those
   names inside the same directory. ~416 GB. This is the UNCENSORED repack: `layers.2-39.attn.wo_b` is a
   plain (unquantized) `.weight`, not `.trellis` — see the engine patches below.
2. **Engram** — the engine reads the checkpoint's shard 47/48 through `TF_DS_ENGRAM`; point it at a
   directory that contains them (in our fleet both the EXL3 pack and the native FP8 directory carry them).
3. **Engine** — build from `bertholomus/TensorFold`, branch `deepseek-v41-zig-tp4`, commit `5940208`, plus
   the four patches in `engine-patches/` (see below), then `build.sh` inside
   `nvcr.io/nvidia/pytorch:26.07-py3`. Three binaries are needed on every node: `tensorfold-native`,
   `tf-dsv41-lanes`, `tf-dsv41-rank-cache`, plus `rdma_gather.fatbin`.
4. **Rank cache — prebuilt, not self-generating.** `node_inputs.sh MODEL_DIR CACHE_DIR <g>` (world=2)
   produces `rank-cache/rank<g>of2-*.bin` (~107 GB) and `dsv41_token_map.json`. The engine only *reads*
   this cache; if it is missing you get `NoRankCache` and exit, with no attempt to build it. Build it while
   the machine is idle — it needs the same unified memory the model will later occupy.
5. **Shared-w2 shard** — the repack stores 27 layers' `ffn.shared_experts.w2` as plain weights, but the
   fused expert kernel cannot be pulled apart per layer, so `fetch_shared_w2.py` copies the trellis trio for
   those layers out of the upstream Mia 2.9 bpw pack and writes `shared-w2-tf.safetensors` (172 MB) into the
   weight directory. Without it those layers fail to load.

## The engine has no CUDA JIT — a weight layout it does not know is a source change

This is the single biggest difference from vLLM/SGLang, and it is what the four commits are for.

**SGLang / vLLM:** Python engines. A new quantisation scheme or tensor layout is usually a new loader class
or a small Python patch; the kernels are Triton kernels JIT-compiled on first use and FlashInfer autotuned per
shape, and CUDA graphs are captured at runtime. You rarely rebuild the engine.

**TensorFold (Zig):** AOT end to end on CUDA. The kernels are **prebuilt cubins** (`kit/cubins/*.cubin`,
`kit/aot/`) loaded through `cuModuleLoadData`; there is no NVRTC and no `cuLink` anywhere in the CUDA path —
runtime compilation exists only for the Metal/macOS backend. The checkpoint loader's **dtype table** and its
per-key dispatch are compiled into the binary. So if the engine has no code path for your specific weight
format, nothing at runtime will rescue you: the load fails, and the fix is Zig source → rebuild fatbins and the
three binaries → redistribute to every node.

That is exactly what happened with our uncensored EXL3 repack:

| what the checkpoint does differently | how the binary fails | patch |
|---|---|---|
| `layers.2-39.attn.wo_b` stored as plain `.weight` (no `.trellis`/`.suh`/`.svh`) | `KeyError: …wo_b.trellis`, or `error: UnsupportedDType` | `69158a0` — dense linear fallback (`Linear.dense`, `dn` cache entries, dense cuBLAS GEMM) |
| the Engram shards (47/48) carry `F8_E4M3` / `F8_E8M0` tensors | the shard header parse dies before any key check, because the compiled-in dtype table has no F8 | `69158a0` — F8 dtypes added to the safetensors dtype table |
| the same shards also keep `bf16` copies of `q/k_weight` | they shadow the `f32` mains after name resolution → `UnsupportedConversion` while writing the rank cache | `7def4b4` — bf16→f32 widen that treats bf16 as the top 16 bits of an f32 (not f16) |
| 27 layers' `shared_experts.w2` are plain, but the fused expert kernel cannot be split per layer | those layers fail to load | `fetch_shared_w2.py` copies the trellis trio for them out of the Mia pack into `shared-w2-tf.safetensors` |

Two consequences:

1. **Budget for engine work, not configuration.** Reading Zig source, patching, rebuilding, and shipping three
   binaries to every node (md5-identical, containers stopped first or you get `Text file busy`) is the unit of
   work for a new weight variant.
2. **Check the weight *structure*, not the `config.json`.** `quantization_config` matching the upstream pack
   byte for byte does not mean the tensors do: what matters is whether a key ends in `.trellis` or `.weight`
   in `model.safetensors.index.json`. That single check tells you whether you need the dense patches.

## Start / stop / verify

```bash
# rank-0 node only; the script starts the worker first, then the head, then smoke-tests
bash start-dsv41-zig-tp2-0102.sh
bash stop-dsv41-zig-tp2-0102.sh

# direct probe plus the same probe through the proxy in front of it
python3 verify-dsv41-zig-tp2.py
python3 verify-dsv41-zig-tp2.py http://<proxy-host>:8100
```

The verify probe checks three things twice (short prompt, 3k prompt): a clean marker answer, `finish_reason
== stop`, and no leaked ` thinking` content.

## Answers to "why is my model returning one token"

The engine's Jinja implementation does not support `**kwargs` (`dict(x, **{...})`). The vLLM-style
`chat_template.jinja` shipped with most repacks therefore fails to compile, and a raw completion against an
instruct model produces a single EOS token. Use the Mia-variant template (no `**kwargs`; it is included
here as `model/chat_template.jinja`, md5 `dcf25a1f7a4d03a41c9592bb0aaad28e`) and restart — the template is
read at startup.

## Operational pitfalls we hit, in the order we hit them

| symptom | cause | fix |
|---|---|---|
| script exits silently, no output at all | `set -e` + `grep -c` returning 1 on zero matches inside an assignment | append `|| true` to the remote count |
| `docker: invalid reference format` | `-e` was prepended twice (the envs string already carried it) | keep the prefix in one place only |
| rank-0 `FileNotFoundError`, rank-1 fine | env values given as **host** paths; only `/model /engram /cache /kit /engine` exist in the container | always write container paths in env |
| `unrecognized arguments: --engram` | `serve` reads these from **env**, not CLI | `TF_DS_KIT`, `TENSORFOLD_CUDA_KERNELS=/kit/aot`, `TF_RDMA_DEVICES`, `TF_DS_RDMA_KERNELS`, `TF_DS_GRAPHS`, `TF_DS_HC_SIDE`, `TF_DS_ENGRAM_AIO` |
| `ibv_modify_qp` 110 timeout across nodes | wrong RoCE leg (each node's f1 faces a different neighbour) | pick the leg from `ip neigh`, then set `NCCL_IB_HCA` **and** `TF_RDMA_DEVICES` to that pair |
| model loads, then complains about memory | `TENSORFOLD_MEMORY_RESERVE_GIB` default 12.2 leaves 99.6 > 96.8 GiB required | we run 6 on TP2 (a compromise; raise it back to 12 before chasing a long-context OOM by other means) |

Sampling: the engine caps `top_k` at 56; requests without `top_k` get a 500 unless the server default is
set, which is why `SERVER_FLAGS` carries `--top-k 20`.

## Engine patches (what each one is for)

Applied on top of upstream `5940208`, in order:

1. `69158a0` — support the uncensored repack: dense (non-trellis) linear fallback in the Zig linear path,
   `dn` rank-cache entries, a dense cuBLAS GEMM, plain `wo_b` layers, **F8 dtypes in the safetensors table**
   (without them, opening the Engram shards kills the load before any key is even checked), and the shared-w2
   shard path. Also folds in the working-tree state from the RDMA GID probe and the seed/top-k server flags.
2. `7def4b4` — widen `bf16 → f32` on checkpoint load. The Engram shards carry `bf16` copies of `q/k_weight`
   that shadow the `f32` mains once names are resolved; the widen must treat bf16 as the top 16 bits of an
   f32 (it is *not* f16).
3. `b2252e0` — world=4 dense `wo_b` decode (see the TP4 doc; harmless at world=2).
4. `5727ac4` — world=4 diagonal exchanges over the all-gather (see the TP4 doc).

`patches/` (0001–0004) are the upstream-compatibility patches from the reference recipes (Triton-variant CUDA
launch fix, prompt-indexer row-block fix, `TF_RDMA_GID`, server seed/top-k).

## Measured

| metric | value |
|---|---|
| API ready | 102–123 s from container start |
| decode c1 (sparkDash, prose 400 tok) | 66.8 tok/s |
| decode aggregate c2 / c3 / c4 | 91.6 / 116.9 / 128.0 tok/s |
| prefill 4k / 8k / 16k / 32k | 1779 / 1912 / 1965 / 1994 tok/s |
| KV pool | 1M positions (`POOL=1048576`, `PARALLEL=4`, `CONTEXT=1048576`) |
| smoke / verify | clean marker answer, `finish_reason=stop`, direct and via proxy |

The 1M pool is the stack's real ceiling. `POOL` is in positions, and each additional 1M costs roughly
0.9 GiB of unified memory (measured on the TP4 line: 2M → 60.51 GiB of model, 8M → 65.76 GiB). Raise it
only with the long-prompt prefill peak in mind.