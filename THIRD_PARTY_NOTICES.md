# Third-party notices

| component | in this repo | upstream | licence | notes |
|---|---|---|---|---|
| SGLang TP4 launcher (`start-tp4.sh`) | `dev/tp4-sglang/start-dsv41-tp4.sh` (**modified copy**) | `MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` @ `64b02de` | **AGPL-3.0**, © 2026 zurih | our edits: HiCache mount removal, `HICACHE_ENABLE` gate, Engram packing, per-rank HCA plumbing |
| chat template | `dev/tp2-tensorfold-zig/model/chat_template.jinja` (unmodified) | `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` | Apache-2.0 | the variant that survives TensorFold's Jinja subset |
| 4 compatibility patches | `dev/tp2-tensorfold-zig/patches/0001..0004` (unmodified `git format-patch`) | `tonyd2wild`, `runs/2026-10-17-1017-tensorfold/patches/` → applied to `bertholomus/TensorFold` | Apache-2.0 | Triton CUDA launch, prompt-indexer blocks, `TF_RDMA_GID`, server seed/top-k |
| our engine commits | `dev/tp2-tensorfold-zig/engine-patches/0001..0004` | on top of `bertholomus/TensorFold` `5940208` | Apache-2.0 | © 2026 Keenni |
| TensorFold engine | not redistributed | `bertholomus/TensorFold`, branch `deepseek-v41-zig-tp4`; originally `ashhart/TensorFold` | Apache-2.0 | Zig native CUDA engine; we ship patches only |
| kernel kit | not redistributed | `bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10` (`kit/`) | see upstream | RoPE tables, vision `gate_bias`, cubins |
| SGLang tree, `boot.py`, `adapter/`, Dockerfiles | not redistributed | `MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` | AGPL-3.0 | we ship our env + launcher only |
| SGLang | not redistributed | `sgl-project/sglang` | Apache-2.0 | base image `lmsysorg/sglang:dev-dsv41` |
| b12x / b12x_next / RoCEnante kernels | not redistributed | local-inference-lab, vendored by the stack above | see upstream | |
| switchless NCCL patch | not redistributed | `alexellis` | see upstream | skip-tree + advertise-all-listener-GIDs |
| sparkDash (benchmark) | not redistributed | `MiaAI-Lab/sparkDash` | see upstream | source of every throughput number quoted |
| model weights | not redistributed | DeepSeek (base), pqhaz (heretic), MiaAI-Lab / dealignai (EXL3) | model licences | 477 GB FP8 or ~416 GB EXL3 |
| EXL3 format | not redistributed | `turboderp/ExLlamaV3` | see upstream | quantisation used by the TP2 checkpoint |
| Heretic (abliteration method) | not redistributed | `p-e-w/heretic` | see upstream | method behind the uncensored weights |

## Licence texts

* **Apache-2.0** — full text in `LICENSE` (also https://www.apache.org/licenses/LICENSE-2.0.txt).
* **AGPL-3.0** — applies to `dev/tp4-sglang/start-dsv41-tp4.sh`. Full text:
  https://www.gnu.org/licenses/agpl-3.0.txt . Upstream notice, verbatim:

  > DeepSeek-V4.1-Flash on 3x NVIDIA DGX Spark
  > Copyright (C) 2026 zurih
  >
  > This repository (start.sh, stop.sh, boot.py changes, adapter/, scripts/, files/, runtime glue,
  > documentation) is free software: you can redistribute it and/or modify it under the terms of the GNU
  > Affero General Public License as published by the Free Software Foundation, either version 3 of the
  > License, or (at your option) any later version.

## What this repository does *not* contain

No model weights, no engine binaries, no container images, no kernel cubins. Everything under
`engine-patches/` and `patches/` is patch text; everything under `dev/` is a launcher, a builder or a probe.