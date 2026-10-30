# TP4 — MiaAI-Lab TensorFold stack (native FP8) + our engine patches

This is the four-node stack we actually run in production for TP4: the **MiaAI-Lab four-node TensorFold
recipe** (DeepSeek-V4.1-Flash-1017 heretic, native FP8), engine `bertholomus/TensorFold` at `5940208`
plus the MiaAI-Lab patch series on top of it. The deploy scripts are upstream's (a `local.sh` with the
serving parameters, a `start.sh` that starts the head plus three workers over SSH); what we add here are
**engine patches** and the two settings that make the prompt cache survive a restart.

The engine tree itself is not redistributed. Patches apply to the upstream tree **after** the MiaAI-Lab
patch series — not to a bare `5940208` checkout.

## engine-patches/

| Patch | What it does |
|---|---|
| `0001-learn-1-n-dsv41-learn-stage-1-...` | dsv41 `--learn` stage 1: a kept state is persisted **with its extent's compressed rows**; `learn_write/read/forget` wire frames; per-rank files; `disk_bytes` accounting |
| `0002-learn-2-n-floor-prompt-cache-marks-onto-the-family-s.patch` | `Rules.chunk`: floor `marks()` and `noteShared()` onto the family's chunk, so the server's shared cut hashes equal to the kept (chunk-floored) state |
| `0003-learn-3-n-the-CUDA-host-builds-the-Imprint-...` | the CUDA host builds the imprint (it existed only in `glm_host`), + `osBuild()` non-macOS |

Mechanism, numbers, pitfalls and the verification checklist: [../../docs/prompt-cache-learn.md](../../docs/prompt-cache-learn.md).

```bash
cd <tensorfold-tree>            # 5940208 + MiaAI-Lab series
git am /path/to/engine-patches/*.patch
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast \
  --prefix <out> -j8 install native
ldd <out>/native/bin/tensorfold-native | grep -c 'not found'   # must be 0
```

`sm=121` is GB10. First build with this tree ≈10 min, incremental ≈6 min; sizes shift from
29.82 → 30.07 MB (`tensorfold-native`) and 18.08 → 18.44 MB (`tf-dsv41-lanes`), which is a useful
sanity check that the code is in.

## Deploy side (mount the binaries, do not rebuild the image)

`tensorfold-native`, `tf-dsv41-lanes` and `tf-dsv41-rank-cache` link only `libc`; CUDA, cuBLAS, NCCL
and libibverbs are resolved at runtime. A version bump is therefore three read-only single-file mounts:

```bash
# scripts/local.sh
ENGINE_BIN_DIR=/root/tf-bin/<build-tag>     # holds tensorfold-native, tf-dsv41-lanes, tf-dsv41-rank-cache
PROMPT_CACHE_GIB=6                          # prefix snapshot store; CUDA default is 0.5 GiB
LEARN_DIR=/root/tf-learn                    # empty = learning off
LEARN_GIB=200
export TF_DS_LEARN_DIR="${LEARN_DIR}"
```

`start.sh` needs `LOCAL_MOUNTS+=" -v $LEARN_DIR:$LEARN_DIR"` and
`SERVE+=(--learn --learn-dir "$LEARN_DIR" --learn-gib "$LEARN_GIB")`. Roll back by pointing
`ENGINE_BIN_DIR` at the previous directory (and `LEARN_DIR=` empty to switch learning off) plus one
restart; the old learned files are ignored automatically because the directory name is the build
identity.

## Serving parameters we run (2026-10-30)

| Parameter | Value | Why |
|---|---|---|
| `--pool` | 12,582,912 positions | 12M positions of KV; measured 99.21 GB device memory at start |
| `--prompt-cache-gib` | 6 | ≈900 prefix states instead of the default 80; capacity, not TTL, is what evicts long sessions |
| `--learn` / `--learn-gib` | on / 200 | shared (system-prompt) states on local NVMe; the pool was trimmed from 16M to 12M to pay for it |
| `TF_DS_NO_COPIES=1` | on | drafted streams stop proposing suffix-lookup copies (matches the reference implementation) |
| `TF_DS_SOLO_OVERHEAD` | 0 | with `DRAFT_MS=25` against a measured 13.3 ms/token it costs ≈9 % single-stream decode |

## Known behaviour of this line

* Two independent budgets: the KV pool (positions, preallocated) and the snapshot store (`gib`, lazily
  allocated). A full snapshot store causes full re-prefill even when the KV pool is mostly free.
* `spare 0 MiB` in the `prompt cache:` log line is normal here — the family does not implement the
  `spare` callback; compute headroom as `of <Z> MiB` − `<X> MiB`.
* Long-running degradation we see: after ~48 h of uptime, `/server_info` can go past 1 s and single-
  stream decode drifts from 39.4 to ~51 ms per step. A restart restores it; watch `/server_info`
  latency as the cheap trigger, not accept-rate.
