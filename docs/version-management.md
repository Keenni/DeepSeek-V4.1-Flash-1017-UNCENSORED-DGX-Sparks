# Version discipline for "one image, N engine builds"

**One line:** the container image ships a complete TensorFold toolchain at *its* patch level. Everything
this repo changes about the engine is mounted over it at run time, so the thing that can silently rot is not
the image — it is the *set* of files you mount, and the provenance of the ones you don't. Two rules fix it:
**one authoritative source tree**, and **mount the whole tool directory, not three files**.

Everything here is written from a production stack that has been through the failure modes below.

## 1. What went wrong (so the rules make sense)

Three things, all of them "it works, so nobody looks":

1. **The source tree drifted between two machines.** The engine used in production was built on node A from a
   checkout whose HEAD was *detached* — no branch, no tag. The same repo on node B carried a "36 upstream
   patches" baseline commit that was missing two of them (no `idleRdma()` call, no `errdefer lanes.deinit()`,
   and three files the structured-output patch creates were absent). Applying the next upstream patch series
   to that baseline produced **fake conflicts** — three of six patches failed on context that only the
   incomplete tree lacked.
2. **The `BUILD` file lied.** It listed eight of our patches; the tree actually contained two. A `strings`
   check for the environment switches the remaining six introduce came back empty. Nobody had compared.
3. **The container ran a mixed version.** `start.sh` mounted three files (native, lanes, rank-cache) as
   single files, because that is the cheapest thing to do. The image's other **16 tools** — the weight
   loader, the Engram packer, the checker, the vision tool, the replays — stayed at the image's patch level.
   A benchmark of the load path using the image's own `tf-dsv41-load` was therefore measuring the *old*
   code. Verified on the running container: 17 files `IMAGE`, 1 file `SAME`.

## 2. One authoritative tree

```
<authority>:/src/TensorFold        # single clone, single build host
  serving   <sha>   image's source snapshot (upstream patches 0001-0036) + ours 9001/9002 + learn 1-3
  up42      <sha>   serving + the upstream 2026-11-01 series (0039-0043, 0046)
  arena     <sha>   up42 + ours 9003-9008                       ← the default build
  tags: build/<name>, base/serving-snapshot, engine/upstream-<sha>
```

* **`serving` is the reference point**, not a branch you build from any more: it reproduces the image's own
  engine source byte for byte, so "does the tree match what is running" is a one-command question.
* **`arena` is where new work stacks.** Every later upstream pull is applied on `arena`, tagged
  `build/<date>+arena`, and recorded in the registry. No side branches that only exist on one machine.
* **Old trees are frozen, not deleted.** The incomplete A-line tree and the previous build host are kept as
  `<name>-legacy-<date>` and marked read-only in a `README-AUTHORITY.md` next to them, with the reason (which
  patches are missing) written down. A stale tree without a "do not use" sign is the whole bug.
* **Bundle backup**: `git bundle --all` (≈94 MB) next to the tree on two machines. Note the trap: if the
  source clone is shallow, bundle it *after* `git fetch --unshallow`, or the clone fails with
  *"remote did not send all necessary objects"* on the oldest commit.

## 3. Registry + tool (the part that answers "which version am I on, and where do I roll back to")

`BUILDS.md`, identical on every node, one row per build:

| build id | tag / commit | patch set | native md5 | state |
|---|---|---|---|---|
| `+up1101+arena` | `build/up1101+arena` | 42 upstream + 9001-9008 + learn 1-3 | `5e24eab1` | **default / running** |
| `+up1101` | `build/up1101` | 42 upstream + 9001/9002 + learn 1-3 | `6386cd8e` | measured, no memory fixes; kept |
| `+learn3` | `build/learn3` | 36 upstream + 9001/9002 + learn 1-3 | `029f7295` | rollback target |

`tfver` is the only interface:

```
tfver list                build dirs on this node: tag, three key md5s, full set or partial
tfver verify [id]         compare every file across all nodes, report DIFF (absent is not an error)
tfver status              live ENGINE_BIN_DIR, what the container actually mounts, the git tag, PREV
tfver image               per file: SAME (from the build dir) or IMAGE (still the image's copy)
tfver use <id>            back up local.sh, record PREV, rewrite one line
tfver rollback            back to PREV
tfver tag <id>            resolve the build id to its commit + tag in the authority tree
```

`tfver image` is the check that would have caught failure mode 3: after a start with a complete set, all 19
entries read `SAME`. On a container started before the fix it reads 17 × `IMAGE`.

## 4. Mount the whole tool directory

```
<build>/bin/*            → /opt/tensorfold/bin          (18 tools)
<build>/native/bin/*     → /opt/tensorfold/native/bin   (tensorfold-native)
```

`start.sh` prefers the directory mount when `<build>/bin` exists and falls back to single files otherwise
(older build dirs keep working). Only `bin` is overridden — CUDA, cuBLAS, NCCL, PyAV and the grammar
library still come from the image, and that is safe because every engine binary links only `libc`:

```bash
ldd <build>/native/bin/tensorfold-native | grep -c 'not found'   # 0
```

Size: a complete build dir is 144–195 MB per node. Worth it — the alternative is benchmarking the wrong
binary.

## 5. Build discipline on a node that is also serving

The build host in this lab is a worker node running a live TP4 rank (~95 GiB of unified memory in use), so
the compiler's memory is the risk, not its CPU:

```bash
docker run --rm --memory 5g --cpus 6 -v <authority-tree>:/src -w /src -e HOME=/tmp \
  --entrypoint /opt/zig/zig <engine-image> \
  build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast \
        --prefix /src/build-<name> -j6 install native
```

* **`install native` is not optional**: `tensorfold-native` lands in `<prefix>/native/bin/`, the rest in
  `<prefix>/bin/`.
* **One compile container at a time**, `--rm` always. The way to exhaust a GB10's unified memory is to stack
  containers that load weights; a compiler is not that, but the discipline is the same.
* Measured on the live worker: `MemAvailable` 8.82 → 8.79 GiB across a full build (Δ ≈ 20 MB; the compiler
  lives in page cache), no OOM in `dmesg`, ~5 min warm / ~11 min cold.

## 6. Rollback

```bash
tfver rollback && bash <deploy>/tp4ctl restart       # stop+start in one script, ~2 min
```

Previous build dirs are never deleted, so a rollback is a one-line change plus one restart. Keep at least
the currently running build and the one before it on every node.
