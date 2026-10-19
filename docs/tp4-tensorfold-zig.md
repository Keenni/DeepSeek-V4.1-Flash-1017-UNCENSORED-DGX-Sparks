# TP4 — TensorFold Zig engine, switchless (no switch) four-node fibre ring

Four GB10 nodes, one rank each, wired as a **switchless ring**: four direct fibre links (01↔02, 02↔03,
03↔04, 04↔01), two HCA legs per node, **no switch anywhere in the fabric**. This document is about the
things that are different from TP2; the weights, Engram and template prerequisites are identical
(see [tp2-tensorfold-zig.md](tp2-tensorfold-zig.md)) — including the fact that the engine is a **compiled
binary** and any weight variant beyond the upstream pack needs engine patches before it will load.

Measured: API ready in 60–61 s, decode 98.1 tok/s (direct probe, prose) / 133.3 tok/s c1 (sparkDash,
structured), prefill 1423–1582 tok/s, 8M-position pool.

## 1. Why a ring is not a flat fabric

world=4 in this engine is a **2D split**: rank pairs exchange along `g^1` (adjacent, fine) and along
`g^2` — the **diagonal**: 01↔03 and 02↔04. On a ring those two pairs are two hops apart.

Three approaches, three outcomes:

| approach | result |
|---|---|
| RoCE point-to-point across the diagonal, forwarded by the middle node | **fails**: the QP is bound to one HCA port; a forwarded packet arrives on the other port → `ibv_modify_qp` retries 35×, then `NcclFailed` |
| TCP over the fibre interfaces (`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=…`, `ALGO=Ring`) | works, proves the topology, but decode collapses to ~31 tok/s |
| **four-node all-gather for the diagonal** (what we ship) | works, stays on 200G RoCE, prefill pays 15–20 % |

The all-gather is our commit `5727ac4`: instead of `packExchange`/`engramSum` second hops to `g^2`, every
rank gathers across all four and picks its peer's slice. Adjacent `g^1` p2p is unchanged.

> We do **not** recommend the management network as a data plane. Ours is 1GbE; every result above is on
> the fibre.

## 2. NCCL / RoCE settings that matter

The patched switchless NCCL library is `LD_PRELOAD`-ed so that both the engine's own collectives and
torch's see the same runtime; without it the QP setup picks the wrong physical leg.

| variable | value | why |
|---|---|---|
| `NCCL_IB_HCA` | both ports of the node | single-port caps at ~100 Gb/s; both ports + `NCCL_CROSS_NIC=1` reaches ~200 Gb/s |
| `NCCL_IB_GID_INDEX` | **computed per node** | the RoCEv2 GID index of each node's own fibre address; a fixed number is wrong on at least one node |
| `NCCL_IB_ADDR_RANGE` | `10.0.0.0/22` | covers all four ring subnets |
| `NCCL_SUBNET_PREFIX_LEN` / `NCCL_SUBNET_AWARE_ROUTING` | `24` / `1` | lets the ring choose the right listener GID |
| `NCCL_SKIP_TREE_CONNECT` / `NCCL_SWITCHLESS_RING_ONLY` | `1` / `1` | no tree/PAT transport on a ring |
| `NCCL_ALGO` / `NCCL_MAX_NCHANNELS` | `Ring` / `4` | ring only |
| `NCCL_CROSS_NIC` | `1` | spread QPs over both rails |
| `NCCL_MERGE_NICS` | `0` | do not fuse the two ports into one device |
| `NCCL_ADDR_FAMILY` / `NCCL_ROCE_VERSION` | `AF_INET` / `2` | IPv4 RoCEv2 |

Control plane (rendezvous) is TCP on the management network; the data plane is entirely fibre.

## 3. Rank cache must be `--parity`

`node_inputs.sh MODEL_DIR CACHE_DIR <g> 4` — world=4 gets `--parity` automatically. A non-parity cache
from an earlier TP2 experiment loads and then fails at request time with `error: MissingEntry`. Keep the two
cache flavours in separate directories and let the script pick.

## 4. Start / stop

```bash
ssh <rank-0-node> 'bash /root/start-dsv41-zig-tp4.sh'    # worker-first, 85 GiB MemFree gate per node
ssh <rank-0-node> 'bash /root/stop-dsv41-zig-tp4.sh'
```

Before a restart: `docker rm -f dsv41-zig-tp4` on all four nodes and `echo 3 > /proc/sys/vm/drop_caches`
(the memory gate is 85 GiB and the rank cache is ~56 GB/rank). Copying a new engine binary while a container
still runs it gives `Text file busy`.

## 5. The failure mode that fooled us

Three of four workers exited with:

```
error: ShortRopeTable
```

while the rank-0 process kept serving `GET /v1/models` with **200 OK**. The head answers that endpoint from
its own API process, so the stack looks healthy in every naive health check while no token can be generated
(a multi-node collective will simply never complete).

**Liveness rule: four containers `Running=true` + one real completion.** Never `/v1/models` alone.

We have not root-caused `ShortRopeTable` yet. It appeared on a warm restart after a long benchmark session;
the same binaries had started cleanly minutes earlier. If you see it, `docker logs` on the workers, not the
head.

## 6. Routes scripts

`tp4-fiber-routes.sh` (per-address host routes + `ip_forward` on every node, so the diagonal is reachable
by hop) and `tp4-policy-routes.sh` (source-based policy routing that keeps each flow's egress on the leg its
GID belongs to) are **diagnostics** in the shipped form: since `5727ac4` the data plane does not need
diagonal reachability. They stay useful to prove or disprove path hypotheses, and they are what the TCP-based
experiments used.

## 7. Measured

| metric | value |
|---|---|
| API ready | 60–61 s |
| decode c1 (direct probe, prose) | 98.1 tok/s |
| decode c1 / c2 / c3 / c4 agg (sparkDash, structured) | 133.3 / 153.7 / 165.0 / 197.8 tok/s |
| prefill (direct probe, 9k prompt) | 1387 tok/s |
| prefill 4k / 8k / 16k / 32k (sparkDash) | 1423 / 1514 / 1559 / 1582 tok/s |
| KV pool | 8M positions (`TF_DS_WINDOW=8388608`, `PARALLEL=16`, `ROUND_ROWS=48`, `STREAMS=16`) |
| memory (startup banner) | 2M window → 60.51 GiB model, 8M → 65.76 GiB, ~0.9 GiB per 1M positions |

The 8M window is real and it is why this line exists for single-user decode work — but prefill is the
weakest of the three stacks, and the head can look alive with three dead workers.