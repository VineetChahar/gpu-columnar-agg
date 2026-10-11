// Block-cooperative radix partitioning with shared-memory write combining.
//
// Naive GPU scatter (each lane writes its row straight to its partition's cursor) turns every row
// into its own partially-written 32-byte sector; with nblk x P open write streams the partial lines
// overflow L2 and the pass runs ~10x slower than its byte count (measured, see NOTEBOOK).  Here a
// block stages one round of rows (blockDim x 4) in shared memory, groups them by partition with a
// counting sort, and then writes each partition's run contiguously.
#pragma once
#include "../encoding/rle.cuh"
#include "dev.cuh"

constexpr int STAGE_PMAX = 256;   // fan-out per staged pass
constexpr int STAGE_ITEMS = 4;    // items per thread per round
constexpr int STAGE_THREADS = 512;  // 2048-row rounds; 2 blocks x 31.7 KB fill the SM's 64 KB
constexpr int STAGE_CAP = STAGE_THREADS * STAGE_ITEMS;

struct StageSmem {
  u64 key[STAGE_CAP];
  i32 val[STAGE_CAP];
  u16 part[STAGE_CAP];
  u32 cnt[STAGE_PMAX];
  u32 off[STAGE_PMAX];
  u32 cur[STAGE_PMAX];
};

// One round: all threads of the block must call it (invalid items have ok=false).
// S.cur[p] must hold this block's next output position for partition p.
__device__ __forceinline__ void stage_round(StageSmem& S, u32 P, const bool ok[STAGE_ITEMS], const u32 p[STAGE_ITEMS],
                                            const u64 key[STAGE_ITEMS], const i32 v[STAGE_ITEMS], u64* okey, i32* oval) {
  for (u32 i = threadIdx.x; i < P; i += blockDim.x) S.cnt[i] = 0;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGE_ITEMS; ++j)
    if (ok[j]) atomicAdd(&S.cnt[p[j]], 1u);
  __syncthreads();
  if (threadIdx.x < 32) {  // exclusive scan of cnt[0..P) by one warp
    u32 carry = 0;
    for (u32 base = 0; base < P; base += 32) {
      u32 i = base + lane_id();
      u32 x = i < P ? S.cnt[i] : 0, inc = x;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        u32 y = __shfl_up_sync(FULL, inc, d);
        if ((int)lane_id() >= d) inc += y;
      }
      if (i < P) S.off[i] = carry + inc - x;
      carry += __shfl_sync(FULL, inc, 31);
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGE_ITEMS; ++j)
    if (ok[j]) {
      u32 pos = atomicAdd(&S.off[p[j]], 1u);
      S.key[pos] = key[j];
      S.val[pos] = v[j];
      S.part[pos] = (u16)p[j];
    }
  __syncthreads();
  const u32 total = S.off[P - 1];  // after the increments off[q] = end of q's run
  for (u32 i = threadIdx.x; i < total; i += blockDim.x) {
    u32 q = S.part[i];
    u32 g = S.cur[q] + (i - (S.off[q] - S.cnt[q]));
    okey[g] = S.key[i];
    oval[g] = S.val[i];
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < P; i += blockDim.x) S.cur[i] += S.cnt[i];
  __syncthreads();
}

// Block-synchronous scan front end: the block owns rows [begin, end) and walks them in rounds of
// blockDim/32 * 128 rows (each warp 128 rows, 4 per lane, vector loads, fused RLE + dictionary
// decode).  f(ok[4], key[4], v[4]) is called once per round by every thread (block-uniform trip count).
template <class KD, class F>
__device__ void fe_block_rounds(const Cols<KD>& C, u64 begin, u64 end, F&& f) {
  const u32 w = threadIdx.x >> 5, lane = lane_id();
  const u64 BT = (u64)(blockDim.x >> 5) * 128;
  if (begin >= end) return;
  u32 cur = rle_upper_bound(C.run_end, C.nruns, C.row0 + min(begin + w * 128, end - 1));
  for (u64 rb = begin; rb < end; rb += BT) {
    const u64 r = rb + w * 128 + 4 * lane;
    u32 code[4] = {0, 0, 0, 0};
    i32 v[4] = {0, 0, 0, 0};
    if (r < end) {
      C.kd.code4(r, code);
      int4 x = *reinterpret_cast<const int4*>(C.vals + r);
      v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
    }
    const u64 last = end - 1;
    u32 run = warp_rle_find(C.run_end, C.nruns, cur, C.row0 + (r < end ? r : last));
    bool ok[4];
    u64 key[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u64 row = r + j;
      ok[j] = false;
      if (row < end) {
        while (C.run_end[run] <= C.row0 + row) ++run;
        u16 p = C.run_val[run];
        ok[j] = p >= C.lo && p < C.hi;
      }
      key[j] = ok[j] ? C.kd.decode(code[j]) : 0;
    }
    cur = __shfl_sync(FULL, run, 31);
    f(ok, key, v);
  }
}

// rows per block for block-contiguous assignment (multiple of the round size)
inline u64 block_chunk(u64 n, int nblk, int threads) {
  u64 bt = (u64)(threads / 32) * 128;
  u64 c = (n + nblk - 1) / nblk;
  return (c + bt - 1) / bt * bt;
}
