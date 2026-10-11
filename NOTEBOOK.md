# NOTEBOOK — running log

One entry per significant step: what changed, what was measured, what it means.
Hardware for every number: Kaggle "T4 x2" (2x Tesla T4, sm_75, 40 SMs, 4 MB L2, 64 KB smem/SM,
15 GB GDDR6), CUDA 12.8, driver 580; host Xeon @2.0 GHz, 2 cores / 4 threads (AVX-512), 31 GB RAM.
Single-GPU numbers use device 0 only.

## 2026-10-11

**Harness first.** `scripts/kaggle_run.py` packs `src bench tests scripts CMakeLists.txt Makefile` into a
base64 tarball embedded in a generated `main.py` driver, pushes it as a private GPU script kernel
(`--accelerator NvidiaTeslaT4Highmem` = T4 x2), polls `kaggle kernels status` every 30 s, pulls the
outputs into `runs/<UTC>/`, and exits non-zero (printing the last 100 lines of `run.log`) on failure.
Verified end-to-end with `src/devprop.cu` (deviceProp dump + a trivial kernel): run
`runs/20261011T095418Z` — both T4s visible, kernel result correct.

**Environment facts that shaped the design.**
- `cmake 3.31`, but `CUDA::nvml` is not found by FindCUDAToolkit on this image; libnvidia-ml lives in
  `/usr/local/nvidia/lib64` (fixed with an explicit `find_library`).
- `nsys` is *not* installed (only `ncu`, `compute-sanitizer`, `nvprof` in `/usr/local/cuda/bin`).
  Consequence: the V5 overlap evidence is an event-based timeline (every async op bracketed by CUDA
  events on its own stream, all relative to one base event). `make nsys` is attempted opportunistically.
- cuDF 26.2 and DuckDB 1.3.2 are preinstalled → cuDF reference and a DuckDB-generated TPC-H SF-10
  without downloading anything.
- Only 4 CPU threads: the CPU baseline is weak in absolute terms. All speedups are stated "vs this
  4-thread CPU", never extrapolated.

**Design decisions (picked, not asked):**
1. *Keys are hashed on the decoded dictionary value, not the code.* With a single dictionary-encoded
   GROUP BY column the dense code could index an array directly (no hash table at all). That shortcut
   disappears with composite keys, multiple chunks with different dictionaries, or joins, so the engine
   treats it as the general case. The CPU *reference* deliberately uses the dense-code array method — a
   different algorithm from every GPU variant, so a shared hashing bug cannot hide.
2. *Encodings.* Key: dictionary codes stored at the narrowest byte width (u8 ≤256, u16 ≤64K, u32),
   dictionary of random 64-bit values. Composite keys: two u16 dictionary columns. Predicate column:
   RLE (u32 run end, u16 value), mean run 32. Values: plain int32 (fixed-point "cents"), SUM exact in int64;
   the one floating aggregate is SUM(v*0.01) in double.
3. *Fused RLE decode.* Warp-cooperative: each warp loads 32 run-ends (one 128 B transaction) and every lane
   finds its run with a 5-step `__shfl_sync` binary search; for 4-rows-per-lane the next 3 rows advance
   locally. Nothing decoded ever reaches global memory.
4. *Hash table.* Open addressing, linear probing, SoA arrays, slot = top log2(cap) bits of fmix64(key),
   capacity = nextpow2(2 x dictionary size) for **every** variant (load factor ≤ 0.5, asserted and
   reported per point — never resized between variants).
5. *Timed region* = table clear + every kernel/host round trip + extraction of the dense result on the
   device. D2H of results is excluded except in V5, where end-to-end is the point.
6. *Partition buffers (V4) store decoded 64-bit keys*: decode is fused into pass 1; re-decoding codes in
   each later pass would cost one random dictionary gather per row per pass.

**Correctness gate first run** (`make verify`, run 20261011T101807Z): 4864 checks — every variant ×
{1, 4099, 1000003 rows} × 6 cardinalities × 3 skews × {single, composite} × sel {0,1,50,100} diffed
against the reference — **0 failures**; memcheck 0 errors / 0 leaks; racecheck 0 hazards.
A gate that never fails proves little, so a checker self-test was added (inject wrong COUNT, a 1e-3
error in the FP sum, a missing group → all must be flagged).

**GPU-slot hygiene.** The Kaggle account allows 2 concurrent batch GPU sessions, shared with another
project's kernels; a push while both are busy is rejected. `kaggle_run.py` now queues (retry every
120 s, up to 6 h). One GPU session was wasted on a compile error (`-Werror` flagged an unused device
function), so `scripts/compile_check.sh` + `kaggle_run.py --cpu` now compile every translation unit with
the real flags in a *CPU-only* Kaggle session (CUDA 12.8 compiler installed from NVIDIA's apt repo
there) before anything is pushed to a GPU.

**Profiler probe** (run 20261011T110053Z): `ncu` → `ERR_NVGPUCTRPERM` (no hardware counters), as
expected. Nothing in this repo relies on counters. `nsys` is not installed, but `apt-cache` lists
`nsight-systems-2024.5.1` → `make nsys` installs it at run time and traces one V5 pipeline.

**Microbenchmarks — the denominators** (same run, T4 #0, clocks recorded per row):
- DRAM read 277.8 GB/s (best over grid sizes; 87% of the 320 GB/s spec — the spec is never used),
  copy 226 GB/s. Under `ncu` the same kernel read 150 GB/s → profilers perturb; never time under one.
- PCIe pinned H2D 12.3 GB/s, D2H 13.1; pageable 4.5 / 4.3 GB/s.
- Global u64 atomics, random addresses: 29 G/s while the working set is L2-resident (≤1 MB), falling
  to **1.08 G/s at 8 MB and 0.72 G/s at ≥128 MB**. A single hot address: 36 G/s (the compiler/L2
  coalesce same-address atomics). → A hash table that does not fit L2 is bound by ~0.7 G random
  atomics/s, i.e. ~140 M rows/s at 5 atomics per row. This one number explains the high-cardinality
  results below.
- Shared memory, random slot of a 512-slot table, cycles per warp instruction:
  key probe (u64 load) AoS-64B 604 vs SoA 192; u32 atomicAdd 584 vs 113; **u64 atomicAdd 4560 vs 869;
  f64 atomicAdd 4573 vs 1248**. Two lessons: (a) the AoS-64B bank-conflict penalty is 3–5×; (b) 64-bit
  shared atomics cost ~8× a 32-bit one (emulated with a CAS loop on sm_75).
  Monte-Carlo bank model: 18.9 wavefronts per u64 warp load for AoS-64B vs 6.0 for SoA (ideal 2).

**Dev sweep, 10M rows, every variant** (run 20261011T110053Z, 10 iters — tuning only, not results):
- Hash variants fall off a cliff at 1M–16M groups (V0 76 → 101 ms at sel 100%) while `v4_sort`
  stays at 7–11 ms: random DRAM atomics (above) vs sequential radix-sort passes.
- Shared-memory tables lose at *low* cardinality because of occupancy: the 1024-slot SoA table +
  warp ring buffers = 52 KB → 1 block (8 warps) per SM. 256 slots → 25 KB → 2 blocks/SM made V3
  1.19 ms vs 2.11 ms (card 16, sel 100%). **Decision: DEFAULT_SLOTS = 256**; 512/1024 kept as ablations.
- `__match_any_sync` aggregation (V3) at Zipf 1.5, sel 100%: 9.5 ms (V0) → 1.3–2.1 ms. At uniform
  64K+ it changes nothing (≤1% of warps have duplicate keys).
- V4 radix-to-smem is *slower* than V3 at 1M groups (27.8 vs 76 ms is a win over hashing, but far from
  sort's 7.4 ms) → added CUDA-event phase timers to see where the time goes.
- Join: dictionary-translated join is 2–7 ms vs 5–28 ms for row-wise probing (it probes the 64K–1M
  dictionary entries once instead of 20M rows); partitioned join is slower than non-partitioned at
  D ≤ 1M — phase timers added there too.
- V5 at 100M rows: global aggregate — CPU (4 thr) 38.6 ms vs GPU 44 ms pinned+overlapped (+27 ms
  one-time `cudaHostRegister`) vs 91 ms pageable. **The PCIe-wall regime exists on this box.**
- CPU engine at 16M groups (1.4 s for 10M rows) used 256 partitions → 1.3 MB tables per partition,
  larger than the core's L2: not a fair baseline. Now picks P so a partition holds ≤8K groups.

**Change: global table → AoS.** One 64-byte aligned slot {key,cnt,sum,dsum,mn,mx}. In SoA the five
atomics of an update touch six different lines; with a DRAM-resident table that is six random
misses per row. The SoA layout stays as the `v0_naive_soa` ablation. Note the irony, which is the
point of V2: the *same* AoS-64B layout that is right for the global table is wrong in shared memory.

**AoS global table — measured** (run 20261011T112912Z, 10M rows; `v0_naive` AoS vs `v0_naive_soa`):
| point | AoS | SoA |
|---|---|---|
| uniform, 64K groups, sel 100% | 8.8 ms | 28.5 ms |
| uniform, 1M, sel 100% | 24.4 | 74.9 |
| uniform, 16M, sel 100% | 58.4 | 100.0 |
| uniform, 16M, sel 10% | 31.1 | 18.7 |
| uniform, 16 groups, sel 100% | 8.6 | 3.5 |
| Zipf 1.5, 16 groups, sel 100% | 22.1 | 9.4 |
A trade-off, not a free win. (1) DRAM-resident table: AoS wins 2–3× (one line per update instead of six).
(2) A 16M-group AoS table is 2 GB vs 1.3 GB, so clear+extract dominate when few rows survive the
filter (16M, sel 10%). (3) Hot keys: AoS puts all five atomics of a hot slot on one L2 line, so the
L2 atomic unit serialises them; SoA spreads them over five lines → SoA is 2.4× faster under contention.
V3's warp aggregation removes most of that contention, so AoS stays the default and the SoA numbers
are kept as an ablation.

**nsys works** once installed at run time: the 3-stream pinned pipeline shows the H2D copies back to
back on the copy engine with the aggregation kernels underneath (93–98% of kernel time hidden under
copies, card 16 — the pipeline is copy-bound, which is exactly the regime V5 is about).
`bench stream nsys_only=1` traces exactly one configuration so the timeline is unambiguous.

**V4 partition pass: naive scatter → write-combined staging** (run 20261011T114910Z):
phase timers showed pass 1 at 20.8 ms for 10M rows / 256 partitions — ~10× its byte count.
Each lane wrote its row straight to its partition cursor, so every row was its own partial 32-byte
sector; with nblk×P ≈ 40K open write streams the partially written lines overflow the 4 MB L2 (and
with ECC on, partial-sector writes cost a DRAM read-modify-write). Fix: blocks stage a round of rows in
shared memory, counting-sort them by partition, then write each partition's run contiguously
(`src/gpu/partition.cuh`).
| 100M rows, uniform, sel 100% | pass 1 naive | pass 1 staged | pass 2 naive | pass 2 staged |
|---|---|---|---|---|
| 64K groups (8 bits, 1 pass) | 217.8 ms | 96.7 ms | – | – |
| 1M groups (12 bits, 2 passes) | 151.0 | 90.3 | 67.8 | 33.7 |
| 16M groups (16 bits, 2 passes) | 333.4 | 196.5 | 197.5 | 80.4 |
1.5–2.4× better, still several× above the write volume: with 256 partitions a 1024-row round gives runs
of ~4 rows (32 B). Next step taken: 512-thread blocks → 2048-row rounds (8-row runs) at the same
occupancy. `v4_radix_nostage` keeps the naive scatter as an ablation.
Even so, **`v4_sort` (cub onesweep radix sort on dictionary codes, 2–3 digit passes) is 2–5× faster than
any hash/partition variant above ~64K groups.** It sorts 4-byte codes (not 8-byte keys), touches no
hash table, and cub's partitioning is exactly the staged, write-combined kind.

**V6 partitioned join redesigned.** The first version partitioned *both* sides into shared-memory
build tables (≤ 2K keys each, up to 2^13 partitions); it lost to the plain join everywhere. Now only the
probe side is partitioned (staged, Bloom applied while partitioning), on the same top hash bits the
build table uses for its home slot, so partition p only touches a contiguous ≤ 1 MB slice of the table
and partitions are processed in order (L2-resident probes). Result at D = 1M, sel 100%: 25.2 ms
(part) vs 26.8 ms (np); D = 16M: 109 vs 90 ms — partitioning pays the random dictionary gather
(fact code → key) twice (histogram + scatter pass), which is as expensive as the probe it localises.
The dictionary-translated join (probe the D dictionary entries once, then a 1-byte gather per row) wins
at every D < F.

**Full benchmark launched** as two parallel Kaggle sessions (`gca-part1a`: verify + micro + 10M/100M
sweeps; `gca-part1b`: nsys, ablations, composite keys, V5, V6, TPC-H, cuDF); 500M rows follow in two
more sessions. Scratch buffers are now freed after every variant so 500M-row V4/sort runs do not
inherit each other's 4–12 GB of partition/sort buffers.

## 2026-10-11 — gca-part2b (500M rows, 64K/1M/16M groups) landed
- runs/20261011T131144Z: 432 rows, 352 timed, all correct; 80 `skip: oom:*` (v4_sort at 1M/16M,
  v4_radix_smem at 1M/16M, v4_radix_l2 at 16M). Out-of-place plans need a second copy of the selected
  rows (+ cub temp) next to ~4 GB of columns, which exceeds 15 GB.
- Consequence: at 16M uniform / 100%, best drops from v4_sort 1.3 G rows/s (100M) to v3_warpagg 0.4 G/s.
  GPU/CPU 119x -> 35x. Where the plan is unchanged (64K, skew/V3), scaling is 4.3-4.8x for 5x rows.
- Not built: chunked sort plan + partial-aggregate merge (would remove the OOM cliff). Written up in RESULTS Scale.
