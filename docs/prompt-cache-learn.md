# Persisting the prompt cache to disk (`--learn`) on the dsv41 TensorFold stack

**One line:** TensorFold's `--learn` (prompt-cache snapshots written to NVMe and read back after a
restart) worked only on the GLM/Metal line; on the DeepSeek-V4.1 (`dsv41`) CUDA line it silently did
nothing. Three engine patches make it work for the shared prefix — the system prompt — and nothing else.
Measured on four GB10 nodes: `learned 2048 tokens to disk`, one file per rank of 8,507,488 B, and the
next identical request still resumes at 2048 tokens (`cached_tokens 2048`), i.e. learning costs nothing
in hit rate.

Everything below is about the **MiaAI-Lab four-node native-FP8** stack (engine `bertholomus/TensorFold`
at `5940208` + the MiaAI-Lab patch series), which is what we run in production now. The patches live in
[`dev/tp4-tensorfold-mia/engine-patches/`](../dev/tp4-tensorfold-mia/engine-patches).

## 1. Two pools, not one — get this right before reading anything else

| Pool | Set by | Size (our config) | Holds |
|---|---|---|---|
| main KV pool | `--pool <positions>` (12,582,912) | ≈99 GB of device memory, preallocated at start | the live token/KV state of running requests |
| prefix snapshot store | `--prompt-cache-gib` (**CUDA default 0.5 GiB**, `native/cuda.zig`) | 6 GiB | one **state snapshot** per reusable prefix |

A state snapshot is a fixed **≈6.25 MiB** regardless of prefix length (window ring + 64 raw rows per
layer). 6 GiB ⇒ ≈900 states; the upstream default 512 MiB ⇒ 80, which a handful of long sessions
cycles through in an hour.

Consequences that look like bugs but are not:

* `spare 0 MiB` in the `prompt cache:` log line is **always 0 on dsv41** — the `spare` callback is not
  implemented for this family (`core/prompt_cache.zig` returns 0). Read the real headroom as
  `of <Z> MiB` minus `<X> MiB`, not from `spare`.
* Eviction is capacity-only: `victimBut()` first drops states already extended by a longer prefix, then
  LRU. There is **no TTL**. A miss therefore always means "prefix changed" or "capacity rotation".
* A roomy main pool does **not** protect the snapshot store: full snapshots ⇒ full re-prefill while the
  KV pool sits mostly empty.

## 2. How `--learn` works (engine side, already upstream)

| Layer | Where | Role |
|---|---|---|
| CLI | `server/cli.zig` | `--learn`, `--learn-dir`, `--learn-gib` (default 32) |
| index | `core/prompt_imprint.zig` | IMPR v2 records `{key, at, tokens, starts, bytes}`; `key = Wyhash(0x1e47, token ids)` ⇒ stable across processes and restarts |
| write | `core/prompt_cache.zig` `learn()` | writes the state of a *shared cut*; evicts by LRU when the cap is hit |
| read | same file, `recall()` | picks the longest on-disk state for this prompt, `vtable.read`, uses it as a shared entry; any read error ⇒ `unlearn` (self-healing, no permanent bad entry) |
| file | `core/snapshot_file.zig` | header + payload + full SHA-256; length or byte mismatch ⇒ rejected |
| directory | `core/lanes/learned_dirs.zig` | `<learn-dir>/<identity:016x>/<key:016x>.r{rank}.bin`; identity is the server build identity, so a new binary invalidates the whole directory |
| family hooks | `prompt_cache.Snapshots.VTable` | optional `write` / `read` / `forget` |

Upstream only the GLM family wires those three hooks, **and** only `native/glm_host.zig` ever builds the
imprint. Both halves are missing for dsv41 — that is the entire reason `--learn` was a no-op here.

## 3. Three gaps, three patches

### 3.1 `0001` — the snapshot must carry the compressed extent, not just the slot

`families/dsv41/lanes.zig` keeps `Snap{ id, at, base, end, buf, stale }`; `buf` is the per-stream ring
(6.25 MiB). The **compressed extent rows** (`comp_codes` / `comp_scales` / `idx_codes` /
`idx_scales`, indexed by row, alive in `[base, end)`) live in the shared plane of the caches and are
moved with `copyExtent`. `snapCopy` deliberately moves only the slot's own blocks.

So a state saved as-is loses every row before `at`: after a restart the device plane is empty and
6.25 MiB of ring is context-free. The patch adds:

* `model.zig`: `extentBytes(at)`, `extentCopy(base, at, host, save)` (per layer, split by `ratio`,
  four planes in order) and a synchronous `copyHost` (`cuMemcpyDtoH` / `HtoD`).
* `lanes.zig`: `TF_DS_LEARN_DIR`, `learnedPath/Header/Write/Read`, `snapWriteFn/snapReadFn/
  snapForgetFn/snapDiskBytesFn`, three wire frames `learn_write / learn_read / learn_forget`
  (12/13/14) so **each rank writes and reads its own shard**; rank0 does not serialise the payload.
* The read path **places the extent itself** — `placeKept(taken, at, null)` at read time, then H2D into
  the new base, `Snap.base = base`, `Snap.end = base + at`. Row numbers never cross processes, so the
  fill path's later `copyExtent` sees `from == to` and short-circuits.
* File layout: header of 16 × u32 (`DSL1`, version, rank, `chunk_rows`, window, streams, `at`,
  slot bytes, extent bytes, pool) + payload `slot ‖ extent` + SHA-256. **Every header field is
  compared** on read, so changing pool, window, streams or rank silently invalidates old files instead
  of misplacing rows — a wrong-base read would be silent corruption.
* `core/prompt_cache.zig`: optional `disk_bytes(ptr, at)` vtable entry, and `learn()` charges it as
  `cost`. Without it the cap accounts 6.25 MiB per state while the file is ≈87 MB for a 98k prefix
  (≈890 B/token of extent) — a 32 GiB cap would overrun by ~14×.

### 3.2 `0002` — the shared flag never matched the kept `at`

`server/chat.zig` asks the store to keep the shared cuts `{system_len-2048, system_len-512,
system_len}`; those are **plain token counts**. dsv41 only keeps a state at a chunk end: its
`markEnds()` floors the mark to `model.chunk_rows` (2048) before calling the keep hook. The keyed entry
therefore hashed the *floored* token prefix, while `shared_keys` held the hash of the *unfloored* cut,
so `e.shared` stayed `false` and `if (shared and key == 0) s.learn()` was never reached — while
`resumed at 2048` looked perfectly healthy.

The patch adds `Rules.chunk` (0 = none) and floors `marks()` **and** `noteShared()` by the same rule
(`w / chunk * chunk`). For dsv41 `chunk = model.chunk_rows`. The set of keep points is unchanged
(`markEnds` already floored identically), so hit/resume behaviour is bit-for-bit the same; only the
shared flag starts matching. Families that do not report a chunk keep `chunk = 0` and are untouched.

**Do not "fix" this with `planned = true`.** With `planned`, `marks()` goes through
`floorStart(starts, w)` and `starts` comes from the chunk plan, which dsv41 leaves empty
(`info.prefill_step == 0` — the family cuts its own prompt grid from position 0). Empty `starts` ⇒
every mark becomes `at = 0` ⇒ `marks()` filters all of them ⇒ **zero states kept, every request 100 %
re-prefill**, while `learned` never appears. We shipped exactly that for eight minutes.

### 3.3 `0003` — the CUDA host never built the Imprint

`learn()` starts with `const im = s.imprint orelse return;`. The imprint is opened in
`native/glm_host.zig` (Metal/GLM) only; `native/cuda.zig` — the host the dsv41 stack runs — accepted
`--learn` / `--learn-gib` into `EngineOptions` and never consumed them. Patch:

* `Host.learned: ?Imprint`, opened after `Store.init` with
  `identity = Wyhash(model dir, prompt_rows, window, streams, snaps.bytes(ptr, prompt_rows))`.
  The last term also separates world size and quantisation layout (per-state bytes differ between TP2
  and TP4); close order is `store.deinit()` then `imprint.deinit()`.
* `prompt_imprint.osBuild()` used `sysctlbyname` — macOS only. On Linux the link fails with
  `undefined symbol`; the patch returns `""` for non-macOS.

## 4. Wiring on the serving side (no image rebuild)

The engine is a plain ELF binary (`ldd`: only `libc`; CUDA / NCCL / verbs are loaded at runtime), so a
new build is published by **mounting the three binaries**, not by building an image:

```bash
# scripts/local.sh
ENGINE_BIN_DIR=/root/tf-bin/native-6bcb475cb17+learn3
LEARN_DIR=/root/tf-learn          # each rank writes its own part here
LEARN_GIB=200
export TF_DS_LEARN_DIR="${LEARN_DIR}"
```

```bash
# start.sh — two additions
[[ -z "${LEARN_DIR:-}" ]] || LOCAL_MOUNTS+=" -v $LEARN_DIR:$LEARN_DIR"
[[ -z "${LEARN_DIR:-}" ]] || SERVE+=(--learn --learn-dir "$LEARN_DIR" --learn-gib "${LEARN_GIB:-32}")
```

Two details that cost us a round each:

* **`TF_DS_LEARN_DIR` is an env var, not an argv flag.** Only the head parses `--learn-dir`; worker
  processes get their settings through the engine env passthrough, so the directory must travel in
  `TF_DS_LEARN_DIR` to be present on all four ranks.
* **The directory has to be bind-mounted.** Without the mount the containers write into their own
  overlay, the log still says `learned N tokens to disk`, the host sees nothing, and every restart
  starts from zero.
* Put it on local NVMe. One 98k prefix is ≈87 MB and the write is a synchronous D2H; on NFS it drags
  the prefill.

Sanity check before starting:

```bash
DRY_RUN=1 ./start-switchless.sh | grep -oE 'TF_DS_WINDOW=[0-9]+|--pool [0-9]+|--prompt-cache-gib [0-9.]+|--learn[^ ]*| -v /root/tf-learn:[^ ]*'
```

## 5. Measured (four GB10, switchless ring, 2026-10-30)

| Item | Value |
|---|---|
| request | 4,175-token prompt, system ≈4,160 tokens, so cuts at 2,112 / 3,663 / 4,175 → kept at 2048 |
| learning | `prompt cache: learned 2048 tokens to disk` |
| files | four, `<key>.r0..r3.bin`, each **8,507,488 B** (64 B header + ≈6.55 MB slot + ≈1.95 MB extent[0,2048) + 32 B SHA) |
| next identical request | `resumed at 2048`, `usage.prompt_tokens_details.cached_tokens = 2048` — no regression |
| cap | `--learn-gib 200` ⇒ log line `0 of 204800 MiB`; `du -sh` converges under the cap |

## 6. What is **not** verified yet

* **Read-back across a restart.** `recall()` only runs when memory has nothing longer for the prefix, so
  the only trigger is a real restart: the expected lines are `read <N> learned tokens from disk` and a
  `resumed at` equal to the learned `at` (not 0). The `placeKept` + extent H2D path is written and
  reviewed, but we have not observed it in a production log yet — treat cross-restart reuse as
  implemented, not proven.
* Short prompts can only persist their front part: `snapSaveFn` requires
  `at <= prompt_len - window` (window ≈1024–2048 tokens), so a 4.1k prompt keeps `at = 2048` only.
  That is by design, not a bug.
* Only the **shared cut** is persisted (`shared and key == 0`, and `key == 0` also excludes
  image-bearing states). Conversation history is still recomputed on a cold start; the full
  "every kept state on disk, block-deduplicated" tier is the B-line `core/rows_store.zig`, which this
  tree does not have.

## 7. If it does not fire

1. `resumed at 0` + `kept 0` for every request ⇒ you are on `planned = true` (see §3.2) or the
   `Rules.chunk` patch is not in the binary.
2. `resumed at` healthy, no `learned` line ⇒ the imprint is not built (patch `0003` missing) or
   `--learn` did not reach the engine (`DRY_RUN` check in §4).
3. `learned` present, host directory empty ⇒ the bind mount is missing.
4. Files present on one or two ranks only ⇒ the three `learn_*` wire frames are not in all four
   binaries; all four nodes must run the same build.
