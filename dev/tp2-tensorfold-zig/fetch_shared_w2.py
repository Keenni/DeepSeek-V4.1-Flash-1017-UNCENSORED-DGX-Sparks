#!/usr/bin/env python3
"""Fetch the missing shared-experts w2 trellis/suh/svh for the 2.9bpw UNCENSORED repack
from the Mia-AiLab母包 via HTTP range reads; write them as one extra safetensors shard
for the zig loader (the repack keeps only plain .weight for shared w2).

The heretic patch touches attn.wo_b only, so the母包's shared w2 tensors are exactly
consistent with this pack's plain w2. Run on a machine with the repack mounted:
    python3 fetch_shared_w2.py REPACK_DIR OUT_FILE   (HF_TOKEN env optional)
"""
import json
import math
import os
import re
import struct
import sys
import urllib.request

MU_INDEX = "/tmp/mu.json"  # the母包's model.safetensors.index.json weight_map, fetched beforehand
BASE = "https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw/resolve/main/"
PARTS = ("trellis", "suh", "svh")
DSIZE = {"I16": 2, "F16": 2, "BF16": 2, "F32": 4}


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    token = os.environ.get("HF_TOKEN", "")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    last = None
    for att in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # retry with backoff
            last = e
            import time
            time.sleep(2 ** att)
    raise RuntimeError(f"fetch failed {url} {start}-{end}: {last}")


def shard_header(url):
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    return json.loads(fetch_range(url, 0, 8 + n - 1)[8:]), 8 + n


def main():
    repack_dir, out_file = sys.argv[1], sys.argv[2]
    mu = json.load(open(MU_INDEX))["weight_map"]
    rp = json.load(open(os.path.join(repack_dir, "model.safetensors.index.json")))["weight_map"]

    has_tr = {k for k in rp if k.endswith(".trellis")}
    layers = []
    for k in rp:
        m = re.match(r"layers\.(\d+)\.ffn\.shared_experts\.w2\.weight", k)
        if m and f"layers.{m.group(1)}.ffn.shared_experts.w2.trellis" not in has_tr:
            layers.append(int(m.group(1)))
    layers = sorted(set(layers))
    print(f"layers needing shared w2 trellis: {len(layers)} -> {layers[:3]} ... {layers[-3:]}")

    want = {}
    for li in layers:
        for part in PARTS:
            key = f"layers.{li}.ffn.shared_experts.w2.{part}"
            if key not in mu:
                raise SystemExit(f"{key} missing in母包 index")
            want[key] = mu[key]
    per_shard = {}
    for key, shard in want.items():
        per_shard.setdefault(shard, []).append(key)

    tensors = {}
    for shard, keys in sorted(per_shard.items()):
        url = BASE + shard
        hdr, data0 = shard_header(url)
        for key in keys:
            e = hdr[key]
            begin = data0 + e["data_offsets"][0]
            end = data0 + e["data_offsets"][1]
            expect = math.prod(e["shape"]) * DSIZE[e["dtype"]]
            assert end - begin == expect, (key, end - begin, expect)
            tensors[key] = (e["dtype"], e["shape"], fetch_range(url, begin, end - 1))
            print(f"  {key} {e['dtype']} {e['shape']} {len(tensors[key][2])}B", flush=True)

    header = {}
    off = 0
    blobs = []
    for key in sorted(tensors):
        dt, shape, blob = tensors[key]
        header[key] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    hb = json.dumps(header, separators=(",", ":")).encode()
    if len(hb) % 8:
        hb += b" " * (8 - len(hb) % 8)
    with open(out_file, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    print(f"wrote {out_file} ({os.path.getsize(out_file)} bytes, {len(tensors)} tensors)")


main()
