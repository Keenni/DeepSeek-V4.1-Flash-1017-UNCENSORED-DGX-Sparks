# Troubleshooting index

Symptom → where to look. Every row is a failure we actually hit.

## All stacks

| symptom | likely cause | section |
|---|---|---|
| Stack answers `GET /v1/models` with 200 but generates nothing | multi-node workers are dead; the head's API process answers on its own | [TP4 Zig §5](tp4-tensorfold-zig.md#5-the-failure-mode-that-fooled-us) |
| Switch to another engine succeeds, then the model returns 1 token | wrong `chat_template.jinja` (engine Jinja lacks `**kwargs`) or none at all | [TP2 §"one token"](tp2-tensorfold-zig.md#answers-to-why-is-my-model-returning-one-token) |
| Start script exits with no output | `set -e` plus `grep -c` returning 1 on no match | [TP2 pitfalls](tp2-tensorfold-zig.md#operational-pitfalls-we-hit-in-the-order-we-hit-them) |
| Nothing loads after a checkpoint swap | wrong checkpoint family for the engine (EXL3 vs native FP8) | [selection §1](selection.md#1-the-memory-arithmetic-decides-tp2-for-you) |
| `Text file busy` when copying a new engine binary | a container is still running the old one | [TP4 Zig §4](tp4-tensorfold-zig.md#4-start--stop) |

## TensorFold Zig (TP2 and TP4)

| symptom | cause | fix |
|---|---|---|
| `NoRankCache` | rank cache never prebuilt — the engine only reads | run `node_inputs.sh` first |
| `error: MissingEntry` at request time | non-parity rank cache on world=4 | rebuild with `<g> 4` (adds `--parity`) |
| `error: UnsupportedDType` at load | safetensors dtype table had no F8 entries (old binary) | use a build with `69158a0` |
| `KeyError: 'layers.N.attn.wo_b.trellis'` / `error: ShortRopeTable` | plain (unquantized) `wo_b` in the uncensored repack | dense-fallback build (`69158a0`), or check which failure you actually have — see log line |
| `unrecognized arguments: --engram …` | flags belong in env, not CLI | see the env table in [TP2](tp2-tensorfold-zig.md#operational-pitfalls-we-hit-in-the-order-we-hit-them) |
| `ibv_modify_qp` 110 / hangs at collective init | RoCE leg chosen from the wrong port | pick the leg from `ip neigh`; set HCA + RDMA devices together |
| model loads, then `99.6 > 96.8 GiB` | `TENSORFOLD_MEMORY_RESERVE_GIB` too small for the weights | raise the reserve; do not shrink the pool to compensate first |
| 500 on every request | sampling: engine caps `top_k` at 56, requests without `top_k` fail unless a default is set | `--top-k 20` in the server flags |
| TP4 prefill much lower than TP2 | expected on a ring: diagonal traffic goes through a four-node all-gather | [TP4 Zig §1](tp4-tensorfold-zig.md#1-why-a-ring-is-not-a-flat-fabric) |

## SGLang (TP4)

| symptom | cause | fix |
|---|---|---|
| workers stay `Created` (`Pid=0`), head healthy | `/hicache` bind mount points at a stale NFS mount point | remove the mount + gate the NFS helper ([§1.1](tp4-sglang.md#11-hicache-off-must-also-remove-the-hicache-bind-mount)) |
| `worker rankN never entered running` after 120 s | same as above | same |
| Engram misses every lookup after switching back from Zig | packed Engram directory was never rebuilt | let the orchestrator repack, or pack four nodes in parallel |
| KV pool smaller than expected | `mem-fraction-static` too low, or HiCache quietly consuming memory | read the boot log's `DSV41 pool sizes` line |
| No prefill speedup from L2 prefetch | pure NCCL data plane; the hook only sees RoCEnante collectives | expected — see [§8](tp4-sglang.md#8-known-gaps-we-have-not-closed) |
| Boot takes much longer than usual | `/root/.cache/huggingface` was cleared (Triton/FlashInfer cold compile) | expected once |

## Prompt cache / `--learn` (TP4, MiaAI-Lab TensorFold stack)

| symptom | cause | fix |
|---|---|---|
| every request `resumed at 0`, `kept 0`, no states | prompt-cache rules set to `planned` — `starts` is empty on this line (`info.prefill_step == 0`), so all marks floor to 0 and are filtered | keep `planned = false`; align cuts with `Rules.chunk` instead ([§3.2](prompt-cache-learn.md#32-0002--the-shared-flag-never-matched-the-kept-at)) |
| `resumed at` healthy but no `learned … to disk` line | imprint not built (CUDA host patch missing) or `--learn` never reached the engine | patches `0003`; check with `DRY_RUN=1 ./start-switchless.sh` |
| `learned` in the log, host learn dir empty | the learn directory is not bind-mounted into the containers | add `-v $LEARN_DIR:$LEARN_DIR` ([§4](prompt-cache-learn.md#4-wiring-on-the-serving-side-no-image-rebuild)) |
| files only on some ranks | the four nodes are not running the same build (the `learn_*` wire frames) | ship the same three binaries to all four |
| `spare 0 MiB` with a nearly empty pool | family does not implement the `spare` callback | read headroom as `of Z MiB` − `X MiB`; not a fault |
| long sessions suddenly re-prefill while the KV pool is free | prefix snapshot store full (capacity, no TTL) | raise `--prompt-cache-gib`; the main KV pool cannot help |

## Measurement

| symptom | cause |
|---|---|
| Two stacks look 2× apart in decode | different `promptType` in sparkDash (`prose` vs `structured`); re-run with the same one |
| Numbers do not match the upstream README | sparkDash uses 400-token generations and its own prompts; upstream uses 256 tokens and its own corpus |