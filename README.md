# GPU columnar aggregation & join engine

A CUDA `GROUP BY` / hash-join engine that runs **directly on compressed columns**. Keys are
dictionary-encoded, the predicate column is run-length-encoded, and decoding is fused into the
scan, so decoded data never reaches global memory. It is benchmarked against a multithreaded CPU
engine and against cuDF to answer one question: *at what group cardinality, skew and selectivity
does the GPU stop winning?*

The deliverable is the **ladder** of implementations (V0–V6) and the crossovers between them, not
one fast kernel. Every rung stays in the repo, is checked against a CPU reference at every sweep
point, and is benchmarked.

| doc | contents |
|---|---|
| [RESULTS.md](RESULTS.md) | crossover plots, roofline, one paragraph per rung, real-data (TPC-H) results |
| [FAILURES.md](FAILURES.md) | things that did not help, and the mechanism |
| [NOTEBOOK.md](NOTEBOOK.md) | running log: every change, measurement and decision |

## Scope

```sql
SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v), SUM(v * 0.01)   -- int64-exact SUM, double SUM
FROM t WHERE lo <= p < hi GROUP BY k                          -- k: single or composite (2 x u16) key

SELECT d.cat, COUNT(*), SUM(f.v), MIN(f.v), MAX(f.v), SUM(f.v * 0.01)
FROM fact f JOIN dim d ON f.fk = d.key WHERE d.p < sel GROUP BY d.cat
```

| variant | idea |
|---|---|
| `v0_naive` | global open-addressing table, 64-bit `atomicCAS` slot claim, one atomic per aggregate |
| `v1_vec` | 16-byte vector loads, predicate → `__ballot_sync`, `__popc`-prefix stream compaction |
| `v2_smem_{aos512,soa512,soa}` | block-private shared-memory tables + global overflow; AoS-64B vs SoA bank-conflict study |
| `v3_warpagg` | `__match_any_sync` warp aggregation: one atomic per distinct key per warp |
| `v4_radix_{smem,l2}` | radix partition (1–2 passes), then per-partition shared-memory or L2-resident tables |
| `v4_sort` | `cub::DeviceRadixSort` on dictionary codes + `ReduceByKey` |
| V5 (`bench stream`) | chunked H2D / kernel / D2H pipeline over ≥ 3 streams, PCIe-wall experiment |
| V6 (`bench join`) | non-partitioned / partitioned hash join, ± Bloom prefilter, dictionary-translated join |

## Layout

```
src/common.hpp, data.hpp, datagen.cpp   types, hashing, synthetic generator (Zipf alias sampler)
src/encoding/rle.cuh                    fused warp-cooperative RLE decode
src/gpu/                                device data, hash table, processors, scan front-ends
src/kernels/                            v0 .. v6, one file per rung, shared interface (variants.hpp)
src/cpu/                                single-threaded reference + multithreaded engine (+ join)
bench/                                  harness: verify / sweep / micro / stream / join / tpch / dump
scripts/kaggle_run.py                   headless Kaggle GPU runner
scripts/plot_all.py                     every plot + table from results.json
scripts/tpch_prep.py, cudf_sweep.py     DuckDB TPC-H generation, cuDF reference
runs/<UTC>/                             raw outputs of every GPU run (run.log, results.json)
```

## Build and run (any CUDA machine with an sm_75+ GPU)

```bash
make build    # CMake, -O3, -arch=sm_75, warnings as errors (CPU code: -O3 -march=native)
make verify   # every variant vs the CPU reference + compute-sanitizer memcheck & racecheck
make bench    # full sweep + micro + stream + join + TPC-H + cuDF -> results.jsonl
python3 scripts/plot_all.py   # plots/*.png + plots/tables.md from results.json
```

## Reproduce every number (Kaggle, headless)

```bash
python3 -m venv .venv && .venv/bin/pip install kaggle numpy matplotlib
# Kaggle API token in ~/.kaggle/access_token (or kaggle.json)
.venv/bin/python scripts/kaggle_run.py --cmd "make verify"        # correctness gate
.venv/bin/python scripts/kaggle_run.py --cmd "make bench-part1"   # 10M + 100M sweeps, micro, ablations
.venv/bin/python scripts/kaggle_run.py --cmd "make bench-part2"   # 500M sweep
.venv/bin/python scripts/kaggle_run.py --cmd "make bench-part3"   # stream, join, TPC-H, cuDF
.venv/bin/python scripts/plot_all.py                              # merges runs/*/results.json
```

`kaggle_run.py` packs the source into a private GPU script kernel (T4 x2), polls it every 30 s,
downloads the outputs into `runs/<UTC timestamp>/`, and exits non-zero on failure. The `make bench`
targets are split in three because Kaggle sessions are capped at 9–12 h.
