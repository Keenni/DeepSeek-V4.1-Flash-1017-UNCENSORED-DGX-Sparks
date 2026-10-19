# Which stack to run, and the arithmetic behind it

**One line: TP2 has exactly one viable stack (TensorFold Zig + EXL3 2.9 bpw), TP4 has two (SGLang is the
production one, Zig is the fast-decode experiment).**

## 1. The memory arithmetic decides TP2 for you

| checkpoint | size | per rank at TP2 | fits a GB10? |
|---|---|---|---|
| native FP8 + FP4 experts (heretic / 1017) | 477 GB | 238 GB | **no** — GB10 has 128 GB unified memory |
| EXL3 2.9 bpw uncensored repack | ~388 GB | ~99 GB + rank-cache | yes |

So a two-node deployment cannot use the native checkpoint at all, and SGLang only reads the native format —
which removes SGLang from TP2 entirely. Of the engines that read EXL3, the Zig native engine is the one still
being developed (the Python TensorFold line is frozen, and the vLLM EXL3 lane was measurably slower in our
own tests).

At TP4 the FP8 checkpoint splits four ways (119 GB/rank) and SGLang becomes available — and it wins on
prefill and KV pool. That is why the four-node production stack is SGLang.

## 2. The ring topology decides how good Zig TP4 can be

Our four nodes are wired as a **ring**: 01↔02, 02↔03, 03↔04, 04↔01, two HCA legs per node, no switch.
The Zig engine only supports world ∈ {2, 4}; world=4 is a **2D split**, so besides the pair exchanges
(adjacent, fine) it needs a `g^2` **diagonal** exchange between 01↔03 and 02↔04 — which are *not* connected.

Three things we measured:

1. **RoCE point-to-point across the diagonal does not work**: the QP is bound to a specific HCA port, and a
   one-hop-forwarded packet arrives on the wrong port → `ibv_modify_qp` retries 35 times, then `NcclFailed`.
2. **TCP over the fibre NICs works but collapses decode** (~31 tok/s) — usable to prove the topology, not to serve.
3. **Four-node all-gather** for the diagonal does work, keeps everything on 200G RoCE, and is what we ship
   (`5727ac4`). Cost: prefill 1423–1582 tok/s versus 1779–1994 tok/s on TP2 — a 15–20 % tax.

## 3. What each stack actually measures

sparkDash benchmark, 400-token generations. **Prompt type differs between rows — do not compare across
them** (this is the mistake we made first).

| stack | prompt | decode c1 | c2 agg | c3 agg | c4 agg | prefill 4k / 8k / 16k / 32k |
|---|---|---|---|---|---|---|
| Zig TP2 | prose | 66.8 | 91.6 | 116.9 | 128.0 | 1779 / 1912 / 1965 / 1994 |
| Zig TP4 | structured | 133.3 | 153.7 | 165.0 | 197.8 | 1423 / 1514 / 1559 / 1582 |
| SGLang TP4 | prose | 69.4 | 99.5 | 124.8 | 143.1 | 2493 / 4664 / 4857 / 4921 |

Direct probes (our own client, prose/counting prompts): Zig TP4 decode 98.1 tok/s and prefill 1387 tok/s
(9,009-token prompt); Zig TP2 prefill ~1696 tok/s.

Read it this way:

* **Prefill**: SGLang TP4 ≫ everything. Zig TP2 is solid. Zig TP4 is the weakest — the all-gather tax.
* **Decode c1**: SGLang TP4 and Zig TP2 are in the same band on the same prompt type (69.4 vs 66.8 prose);
  Zig TP4's much higher number is on a different (structured) prompt and from a different run.
* **KV pool**: SGLang 6.78M tokens vs Zig TP2 1.0M positions vs Zig TP4 8.0M positions. If your workload is
  many long sessions, TP2's 1M pool is the first thing you will hit.

## 4. Decision table

| workload | pick |
|---|---|
| only two Sparks | Zig TP2 + EXL3 2.9 bpw. Accept the 1M pool, or raise `POOL` and pay ~0.9 GiB per 1M positions. |
| four Sparks, agent / long-context / multi-session | SGLang TP4. Bigger pool, far better prefill, one process tree we can reason about. |
| four Sparks, single-user decode throughput is everything | Zig TP4 is worth benchmarking — but read `docs/tp4-tensorfold-zig.md` first, and do not trust `/v1/models` for liveness. |

## 5. Things that are true for all three stacks

* `:8000` is globally exclusive in our fleet; all three serve the same model name
  (`deepseek-v4.1-flash-uncensored`) so the proxy in front needs no change when we switch.
* Switching engines means switching checkpoints (EXL3 ↔ native FP8) **and** re-preparing Engram:
  Zig reads the checkpoint's shard 47/48 directly, SGLang needs its own packed directory.
* Stop-then-start must be one script: stop the head first, start workers first, probe last. Any manual
  two-step leaves the fleet serving nothing while you debug.