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
| `v4_sort_chunked` | the same over memory-sized row chunks, partials folded into a dense per-code accumulator (fits 500M rows) |
| V5 (`bench stream`) | chunked H2D / kernel / D2H pipeline over ≥ 3 streams, PCIe-wall experiment |
| multi-GPU (`bench mgpu`) | rows split over 2 GPUs, partial results merged on GPU 0 |
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
make bench    # everything: sweeps, micro, stream, join, TPC-H, cuDF, follow-ups (chunked, multi-GPU) -> results.jsonl
python3 scripts/plot_all.py   # plots/*.png + plots/tables.md from results.json
```

## Reproduce every number (Kaggle, headless)

```bash
python3 -m venv .venv && .venv/bin/pip install kaggle numpy matplotlib
# Kaggle API token in ~/.kaggle/access_token (or ~/.kaggle/kaggle.json)
.venv/bin/python scripts/kaggle_run.py --cpu --cmd "bash scripts/compile_check.sh"   # compile check, no GPU quota
.venv/bin/python scripts/kaggle_run.py --slug gca-part1a --cmd "make verify && make bench-part1a"
.venv/bin/python scripts/kaggle_run.py --slug gca-part1b --cmd "make bench-part1b"
.venv/bin/python scripts/kaggle_run.py --slug gca-part2a --cmd "make bench-part2a"
.venv/bin/python scripts/kaggle_run.py --slug gca-part2b --cmd "make bench-part2b"
.venv/bin/python scripts/kaggle_run.py --slug gca-part3  --cmd "make bench-part3"
.venv/bin/python scripts/kaggle_run.py --slug gca-part4  --cmd "make bench-part4"
.venv/bin/python scripts/kaggle_run.py --slug gca-part5  --cmd "make bench-part5"
.venv/bin/python scripts/kaggle_run.py --slug gca-part5b --cmd "make bench-part5b"
.venv/bin/python scripts/plot_all.py      # merges runs/*/results.json -> results.json, plots/, plots/tables.md
```

| target | contents | ~GPU time |
|---|---|---|
| `make verify` | every variant vs CPU reference on edge cases, checker self-test, joins, memcheck, racecheck | 10 min |
| `bench-part1a` | microbenchmarks + main sweep at 10M and 100M rows (6 cardinalities × 4 skews × 4 selectivities × 9 variants) | 1.5 h |
| `bench-part1b` | nsys trace, ablations, composite keys, V5 streaming, V6 joins, TPC-H SF-10 (+DuckDB), cuDF | 1 h |
| `bench-part2a/2b` | main sweep at 500M rows | 2–5 h |
| `bench-part3` | capacity-policy experiment, late materialisation | 0.5 h |
| `bench-part4` | follow-up: verify + sanitizers for chunked sort and multi-GPU, chunked sort at 500M, full TPC-H Q6 | 1 h |
| `bench-part5` | follow-up: multi-GPU (2 x T4) row split, cap=est at 500M | 2 h |
| `bench-part5b` | multi-GPU re-run with the merge table sized by the partial groups (supersedes part 5's mgpu rows) | 1 h |

`kaggle_run.py` packs the source into a private GPU script kernel (T4 x2), queues if the account's
two GPU sessions are busy, polls every 30–60 s, downloads the outputs into `runs/<UTC timestamp>/`,
and exits non-zero (printing the last 100 log lines) on failure. Splitting the targets lets two
sessions run in parallel and keeps each one under Kaggle's session limit. `make bench` runs all of
them in sequence on a local CUDA machine.

## Deviations from the brief (deliberate, documented)

- **Value column** is plain int32 (fixed-point cents). Keys are dictionary-encoded and the predicate is
  RLE-encoded. SUM is exact in int64, and the floating aggregate is `SUM(v*0.01)` in double.
- **Q6** in the generic engine is date-predicate-only, because the engine has one RLE predicate column.
  The **full Q6** (date + discount + quantity) runs in a dedicated kernel (`bench/tpch.cu`) and matches
  DuckDB's official Q6 SQL exactly. Q1's composite key and date predicate are exact.
- **Multi-GPU stretch**: done as a row split over the Kaggle T4 × 2 with a merge of partial results
  (`bench mgpu`; RESULTS.md, Multi-GPU). A key-partitioned split (no merge) was not built.
- **ncu** is blocked on Kaggle (`ERR_NVGPUCTRPERM`). Occupancy comes from the occupancy API, bytes are
  analytic, bandwidth is from our own microbenchmark. **nsys** is installed at run time from NVIDIA's apt repo.
