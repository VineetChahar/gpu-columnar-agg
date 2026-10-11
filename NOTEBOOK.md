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
