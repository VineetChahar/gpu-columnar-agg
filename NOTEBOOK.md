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
