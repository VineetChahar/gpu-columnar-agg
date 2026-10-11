// Fused, warp-cooperative RLE decode of the predicate column.  Never materialises the column:
// each lane resolves the run covering its row with a 5-step shuffle binary search over a
// 32-run window that the warp loads cooperatively (one coalesced 128-byte load per window).
#pragma once
#include "../gpu/dev.cuh"

// first run whose (exclusive) end is > row
__device__ __forceinline__ u32 rle_upper_bound(const u32* run_end, u32 nruns, u64 row) {
  u32 lo = 0, hi = nruns;
  while (lo < hi) {
    u32 mid = (lo + hi) >> 1;
    if (run_end[mid] <= row) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Rows must be non-decreasing across lanes and `base` <= the answer for lane 0.
// All 32 lanes must call it.  Typically one window load resolves every lane.
__device__ __forceinline__ u32 warp_rle_find(const u32* run_end, u32 nruns, u32 base, u64 row) {
  u32 res = 0;
  bool done = false;
  while (true) {
    u32 idx = base + lane_id();
    u32 e = idx < nruns ? run_end[idx] : 0xffffffffu;
    u32 pos = 0;
#pragma unroll
    for (u32 step = 16; step >= 1; step >>= 1) {
      u32 v = __shfl_sync(FULL, e, pos + step - 1);
      if (v <= row) pos += step;
    }
    u32 last = __shfl_sync(FULL, e, 31);
    if (!done && !(pos == 31 && last <= row)) { res = base + pos; done = true; }
    if (__all_sync(FULL, done)) break;
    base += 32;
  }
  return res;
}
