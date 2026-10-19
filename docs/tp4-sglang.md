# TP4 — SGLang on four DGX Sparks (native FP8/FP4 checkpoint)

This is the four-node production stack. It is
[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
at `64b02de` serving `DeepSeek-V4.1-Flash-1017-heretic` (native FP8 weights, FP4 routed experts, 48 shards,
477 GB). We contribute the environment file and two start-script fixes; the tree itself is upstream and is
not redistributed here.

Measured: ready in 4.3 min from cold compile caches, KV pool 6,781,184 tokens, decode 69.4 tok/s c1
(sparkDash, prose 400 tok), prefill 2,493–4,921 tok/s over 4k–32k.

## 1. What we changed, and why

### 1.1 HiCache off must also remove the `/hicache` bind mount

Turning HiCache off by dropping `--enable-hierarchical-cache / --hicache-*` from `EXTRA_SGLANG_ARGS` is not
enough. The container spec still carries `-v $HICACHE_HOST_DIR:/hicache`, and on a worker that path is the
**leftover NFS mount point** from a previous HiCache-era run:

```
docker inspect dsv41-worker
State = created, Pid = 0
State.Error: error while creating mount source path '/root/dsv41-hicache':
             mkdir /root/dsv41-hicache: file exists
```

The container never reaches `running` (the orchestrator gives up with `worker rank1 never entered running`
after 120 s). The trap: on the head node that path is an ordinary empty directory, so the head starts fine
and only workers are stuck.

Fixes, all three needed:

1. comment out `-v "$HICACHE_HOST_DIR:/hicache"` and the two `SGLANG_HICACHE_FILE_BACKEND_*` env lines;
2. gate the NFS sharing helper: `[[ "${HICACHE_ENABLE:-0}" == "1" ]] || return 0` at the top of
   `ensure_hicache_nfs()`, and set `HICACHE_ENABLE=0` in the env file;
3. on every worker: `systemctl stop 'root-dsv41\x2dhicache.mount'`, `systemctl reset-failed …`,
   `umount -f -l /root/dsv41-hicache`, delete the `/etc/fstab` line; on the head delete the `/etc/exports`
   line and `exportfs -ra`.

### 1.2 Everything that has to agree across four nodes

Deploy script and env file must be byte-identical on all four nodes (md5). Our `sync` subcommand pushes the
head's copies; a drifted script once produced "workers never started" with no other symptom.

## 2. Pool calibration with HiCache off

| date | HiCache | `mem-fraction-static` | full pool (tokens) | swa | memory left outside the pool | `available_gpu_mem` after boot |
|---|---|---|---|---|---|---|
| 2026-09-26 | ratio 0.18 | 0.85 | 3,930,368 | 88,576 | 14.14 GB | — |
| 2026-10-20 | off | 0.85 | **6,781,184** | 88,576 | **18.01 GB** | 7.29 GB |

Also: `c4=1,695,296`, `c128=52,978`, `c4_state=11,072`, `max_running_requests=16`.

The HiCache host pool is carved out of the *same* unified memory as the device pool, so turning it off
improves both the device pool and the free headroom. **Do not raise `mem-fraction-static` because HiCache is
off** — you would eat the prefill-transient headroom for nothing.

`MAX_TOTAL_TOKENS=8000000` is above the profiled capacity, so boot logs a warning and uses 6,781,184. It is
a decorative pin in this configuration.

## 3. Engram must be repacked after coming back from the Zig line

SGLang wants a packed Engram directory (`engram-l1-r<rank>of4.bin`, `engram-l14-r<rank>of4.bin`, ~48 GB per
node, ~100 s per layer at ~500 MiB/s). The Zig line reads the checkpoint's own shard 47/48 and never creates
that directory, so every switch back needs a repack. The orchestrator does it automatically when the marker
file is missing, but it packs the four ranks **sequentially** (4 × ~2 min of I/O plus model open). Packing the
four nodes in parallel (`pack-only <rank>` on each node) takes about the same wall time as one node.

## 4. Compile caches are persistent — and that is where the 4.3 minutes go

| cache | location (host) | note |
|---|---|---|
| Triton kernels, FlashInfer workspace, HF home | `/root/.cache/huggingface` (mounted at `/cache/huggingface`) | 34 MB after the first cold boot here; wipe it and the next boot is slow again |
| b12x_next CuTe-DSL JIT | `$DEPLOY/state-r<rank>/b12x-next-compile` | survives container `rm -f` |
| FlashInfer autotune decisions | alongside, gated by `DSV41_AUTOTUNE_KEEP=1` | writes a per-rank fingerprint so one rank cannot end up with a partial hit |

`docker rm -f` does not touch any of them; a first boot after clearing them was 4.3 min, warm starts are
shorter.

## 5. Measured (sparkDash, prose 400 tokens, HiCache off)

| concurrency | decode mean | aggregate | TTFT median |
|---|---|---|---|
| 1 | 69.41 tok/s | 69.41 | 239 ms |
| 2 | 50.26 | 99.53 | 516 ms |
| 3 | 42.58 | 124.80 | 285 ms |
| 4 | 37.94 | 143.12 | 249 ms |

Prefill: 4k 2,493 / 8k 4,664 / 16k 4,857 / 32k 4,921 tok/s (TTFT 1.65 / 1.76 / 3.38 / 6.66 s).

## 6. Reading the numbers out of sparkDash

```bash
curl -s http://<sparkdash-host>:5555/api/sparks/<spark-id>/llm/bench          # last decode/TTFT run
curl -s http://<sparkdash-host>:5555/api/sparks/<spark-id>/llm/prefill-bench # last prefill run
```

Each endpoint returns only `active` and `last` — there is no history list (`/llm/bench/list` and friends
fall through to the SPA). Copy the JSON if you want a record. In a fleet, the head node's spark id is not
necessarily its hostname.

## 7. Things that look like errors and are not

* `V4.1 Rust image processor unavailable … cargo --version --verbose` — we deliberately remove the rustup
  shim from `PATH`; the PIL fallback is fine.
* `NCCL INFO ncclOsDlopen(libnccl-*.so) failed` — optional NCCL plugins, absent by design.
* `No HuggingFace chat template found` / `Tokenizer is still TokenizersBackend` — the checkpoint's
  `chat_template.jinja` is used; chat completions work.
* `sleep_on_idle` plus an idle GPU: normal.

## 8. Known gaps we have not closed

* `DSV41_L2_PREFETCH` only hooks the RoCEnante path (`b12x.comm.roce(_ring)`); on a pure NCCL data plane it
  logs `saw no RoCE collective (NCCL path); not learned` and never arms. Same for the RoCE-only compact
  gather. If you run RoCEnante instead, those become live.
* The DSpark SPS table is not consumed on an Engram model unless `DSPARK_SPS_ALLOW_ENGRAM=1`, so speculative
  decoding is running without the tuned accept table.
* Upstream's open PR for forced/named `tool_choice` on V4.1 (`sglang#39665` backport) targets the default
  `Dockerfile`, not `Dockerfile.canary*` — if you build `canary-roce` as we do, forced tool calls come back
  as `message.content` with no `tool_calls`.