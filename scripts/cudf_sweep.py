#!/usr/bin/env python3
"""cuDF reference for the synthetic sweep.  For each point the C++ generator dumps the *decoded*
columns (key int64, val int32, pred uint16) - cuDF cannot read our dict/RLE encoding - and cuDF runs
  df[df.pred < sel].groupby('key').agg(count, sum, min, max, sum(val*0.01))
on device-resident data.  Caveat (stated in RESULTS.md): cuDF does more work than our engine
(materialises the filtered frame, general-purpose hash groupby, sorts nothing but builds a full
output table) and reads ~2.8x more bytes (uncompressed 14 B/row vs 5-9 B/row).

usage: cudf_sweep.py --rows 10M,100M [--cards ...] [--skews ...] [--sels ...]
"""
import argparse, json, os, statistics, subprocess, sys, time
import numpy as np

def num(s):
    m = {"M": 1e6, "K": 1e3, "G": 1e9}
    return int(float(s[:-1]) * m[s[-1]]) if s[-1] in m else int(float(s))

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="10M,100M")
ap.add_argument("--cards", default="16,256,4096,65536,1048576,16777216")
ap.add_argument("--skews", default="0,0.5,1,1.5")
ap.add_argument("--sels", default="1,10,50,100")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--bench", default="build/bench")
ap.add_argument("--tmp", default="/kaggle/tmp")
a = ap.parse_args()

import cudf, cupy as cp
out_path = os.path.join(os.environ.get("RESULTS_DIR", "."), "results.jsonl")
os.makedirs(a.tmp, exist_ok=True)
for rows in [num(x) for x in a.rows.split(",")]:
    for card in [int(x) for x in a.cards.split(",")]:
        for skew in [float(x) for x in a.skews.split(",")]:
            base = os.path.join(a.tmp, "dump")
            subprocess.run([a.bench, "dump", f"rows={rows}", f"card={card}", f"skew={skew}", f"out={base}",
                            f"sels={a.sels}"], check=True, stdout=subprocess.DEVNULL)
            expected = {}
            for line in open(base + "_expected.txt"):
                s, g, c, sm = line.split()
                expected[int(s)] = (int(g), int(c), int(sm))
            try:
                gdf = cudf.DataFrame({"key": np.fromfile(base + "_key.u64", dtype=np.uint64),
                                      "val": np.fromfile(base + "_val.i32", dtype=np.int32),
                                      "pred": np.fromfile(base + "_pred.u16", dtype=np.uint16)})
                gdf["d"] = gdf["val"].astype("float64") * 0.01
            except Exception as e:  # e.g. out of device memory at 500M rows
                print("cudf load failed", rows, card, skew, repr(e), flush=True)
                continue
            for sel in [int(x) for x in a.sels.split(",")]:
                def q():
                    f = gdf[gdf.pred < sel]
                    return f.groupby("key", sort=False).agg({"val": ["count", "sum", "min", "max"], "d": "sum"})
                row = {"exp": "cudf", "rows": rows, "card": card, "skew": skew, "sel": sel}
                try:
                    r = q()
                    g, c, sm = expected[sel]
                    ok = len(r) == g and int(r[("val", "count")].sum()) == c and int(r[("val", "sum")].sum()) == sm
                    ts = []
                    for i in range(a.iters + 5):
                        cp.cuda.Device().synchronize(); t = time.perf_counter(); q(); cp.cuda.Device().synchronize()
                        if i >= 5: ts.append((time.perf_counter() - t) * 1e3)
                    ts.sort()
                    row.update({"correct": ok, "median_ms": statistics.median(ts), "p25_ms": ts[len(ts) // 4],
                                "p75_ms": ts[3 * len(ts) // 4], "n_iter": len(ts), "groups": len(r)})
                except Exception as e:
                    row.update({"correct": False, "skipped": repr(e)[:200]})
                print(json.dumps(row), flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            del gdf
            for ext in ("_key.u64", "_val.i32", "_pred.u16"):
                os.remove(base + ext)
