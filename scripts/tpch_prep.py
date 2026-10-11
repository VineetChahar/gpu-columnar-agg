#!/usr/bin/env python3
"""Generate TPC-H SF-10 lineitem with DuckDB's dbgen, sort by l_shipdate, export the columns the
engine reads as raw binaries, and record DuckDB's own answers (and timings) for the two queries.

usage: tpch_prep.py OUT_DIR [--sf 10]
Writes OUT_DIR/{returnflag.u8, linestatus.u8, shipdate.u16, extprice.i32, params.i32,
q1_expected.csv, q6_expected.csv} and appends DuckDB timing rows to $RESULTS_DIR/results.jsonl.
"""
import argparse, datetime, json, os, statistics, time
import duckdb
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sf", type=float, default=10)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
con = duckdb.connect()
con.execute(f"SET threads TO {os.cpu_count()}")
con.execute("INSTALL tpch; LOAD tpch")
t0 = time.time()
con.execute(f"CALL dbgen(sf={a.sf})")
print(f"dbgen sf={a.sf}: {time.time() - t0:.1f}s", flush=True)

# Columns in engine form; prices as integer cents (DECIMAL(15,2) -> exact int)
con.execute("""
CREATE TABLE li AS
SELECT CASE l_returnflag WHEN 'A' THEN 0 WHEN 'N' THEN 1 ELSE 2 END::UTINYINT AS rf,
       CASE l_linestatus WHEN 'F' THEN 0 ELSE 1 END::UTINYINT AS ls,
       (l_shipdate - DATE '1992-01-01')::USMALLINT AS sd,
       (l_extendedprice * 100)::INTEGER AS price,
       l_returnflag, l_linestatus, l_shipdate
FROM lineitem ORDER BY l_shipdate""")
con.execute("DROP TABLE lineitem")
for col, dt, name in [("rf", np.uint8, "returnflag.u8"), ("ls", np.uint8, "linestatus.u8"),
                      ("sd", np.uint16, "shipdate.u16"), ("price", np.int32, "extprice.i32")]:
    arr = con.execute(f"SELECT {col} FROM li").fetchnumpy()[col].astype(dt)
    arr.tofile(os.path.join(a.out, name))
    print(name, arr.shape, flush=True)
d0 = datetime.date(1992, 1, 1)
q1_hi = (datetime.date(1998, 9, 2) - d0).days
q6_lo = (datetime.date(1994, 1, 1) - d0).days
q6_hi = (datetime.date(1995, 1, 1) - d0).days
np.array([q1_hi, q6_lo, q6_hi], dtype=np.int32).tofile(os.path.join(a.out, "params.i32"))

Q1 = f"""SELECT l_returnflag, l_linestatus, count(*), sum(price), min(price), max(price), sum(price*0.01::DOUBLE)
FROM li WHERE sd <= {q1_hi} GROUP BY 1, 2 ORDER BY 1, 2"""
Q6 = f"""SELECT '-', '-', count(*), sum(price), min(price), max(price), sum(price*0.01::DOUBLE)
FROM li WHERE sd >= {q6_lo} AND sd < {q6_hi}"""
# and the official-SQL versions over the real date column, to show the encodings changed nothing
Q1_SQL = """SELECT l_returnflag, l_linestatus, count(*), sum(price) FROM li
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL 90 DAY GROUP BY 1, 2 ORDER BY 1, 2"""
res = {}
for name, q in [("q1", Q1), ("q6", Q6)]:
    rows = con.execute(q).fetchall()
    with open(os.path.join(a.out, f"{name}_expected.csv"), "w") as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
    res[name] = rows
    ts = []
    for i in range(a.iters + 1):
        t = time.perf_counter(); con.execute(q).fetchall(); dt = (time.perf_counter() - t) * 1e3
        if i: ts.append(dt)
    ts.sort()
    row = {"exp": "tpch_external", "engine": "duckdb", "query": name.upper(), "median_ms": statistics.median(ts),
           "p25_ms": ts[len(ts) // 4], "p75_ms": ts[3 * len(ts) // 4], "n_iter": len(ts), "threads": os.cpu_count(),
           "note": "DuckDB on its own (uncompressed, in-memory) table; includes its planning"}
    with open(os.path.join(os.environ.get("RESULTS_DIR", "."), "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(name, "duckdb", row["median_ms"], "ms", rows, flush=True)
check = con.execute(Q1_SQL).fetchall()
assert [(r[0], r[1], r[2], r[3]) for r in res["q1"]] == [(r[0], r[1], r[2], r[3]) for r in check], "date encoding mismatch"
print("Q1 over encoded shipdate == Q1 over the DATE column: ok")

# cuDF on the same columns (does more work: no compression, materialised filter)
try:
    import cudf, cupy as cp
    gdf = cudf.DataFrame({
        "rf": np.fromfile(os.path.join(a.out, "returnflag.u8"), dtype=np.uint8),
        "ls": np.fromfile(os.path.join(a.out, "linestatus.u8"), dtype=np.uint8),
        "sd": np.fromfile(os.path.join(a.out, "shipdate.u16"), dtype=np.uint16),
        "price": np.fromfile(os.path.join(a.out, "extprice.i32"), dtype=np.int32)})
    gdf["d"] = gdf["price"].astype("float64") * 0.01
    def q1():
        f = gdf[gdf.sd <= q1_hi]
        return f.groupby(["rf", "ls"]).agg({"price": ["count", "sum", "min", "max"], "d": "sum"})
    def q6():
        f = gdf[(gdf.sd >= q6_lo) & (gdf.sd < q6_hi)]
        return (len(f), f.price.sum(), f.price.min(), f.price.max(), f.d.sum())
    for name, fn in [("Q1", q1), ("Q6", q6)]:
        ts = []
        for i in range(a.iters + 3):
            cp.cuda.Device().synchronize(); t = time.perf_counter(); r = fn(); cp.cuda.Device().synchronize()
            if i >= 3: ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        row = {"exp": "tpch_external", "engine": "cudf", "query": name, "median_ms": statistics.median(ts),
               "p25_ms": ts[len(ts) // 4], "p75_ms": ts[3 * len(ts) // 4], "n_iter": len(ts),
               "note": "cuDF on uncompressed device-resident columns; does more work (materialises the filtered frame)"}
        with open(os.path.join(os.environ.get("RESULTS_DIR", "."), "results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(name, "cudf", row["median_ms"], "ms", flush=True)
except Exception as e:  # recorded, not fatal
    print("cudf failed:", repr(e))
