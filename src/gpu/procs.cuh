// Aggregation "processors": what happens to one (valid, key, value) per lane, all lanes converged.
//   PGlobal<MATCH>      : global-table atomics (V0/V1; MATCH=true is the V3-global ablation)
//   PSmem<Layout,MATCH> : block-private shared-memory table + overflow to global (V2/V3)
#pragma once
#include "dev.cuh"

// register copy of one aggregate state
struct AggR {
  u32 c;
  i64 s;
  i32 mn, mx;
  double ds;
  __device__ __forceinline__ static AggR of(bool valid, i32 v) {
    return valid ? AggR{1u, (i64)v, v, v, vscale(v)} : AggR{0u, 0, INT_MAX, INT_MIN, 0.0};
  }
};

// Reduce AggR over each group of `peers` lanes (from __match_any_sync).  After the call, the
// lowest lane of each peer group holds the group total.  Tree reduction over peer rank:
// log2(group size) shuffle rounds (after E. Westphal, "Voting and Shuffling to Optimize Atomic Operations").
__device__ __forceinline__ void reduce_peers(u32 peers, AggR& a) {
  const u32 lane = lane_id();
  u32 rel = __popc(peers & lanemask_lt());
  u32 rest = peers & ~((2u << lane) - 1u);
  while (__any_sync(FULL, rest)) {
    int next = __ffs(rest);
    int src = next ? next - 1 : (int)lane;
    u32 c = __shfl_sync(FULL, a.c, src);
    i64 s = __shfl_sync(FULL, a.s, src);
    i32 mn = __shfl_sync(FULL, a.mn, src);
    i32 mx = __shfl_sync(FULL, a.mx, src);
    double ds = __shfl_sync(FULL, a.ds, src);
    if (next) { a.c += c; a.s += s; a.mn = min(a.mn, mn); a.mx = max(a.mx, mx); a.ds += ds; }
    bool done = rel & 1u;
    rest &= __ballot_sync(FULL, !done);
    rel >>= 1;
  }
}

// ------------------------------------------------------------------ global
template <bool MATCH>
struct PGlobal {
  struct Params { GT T; };
  static constexpr size_t SMEM = 0;
  static constexpr bool WANTS_CODE = false;
  GT T;
  __device__ PGlobal(const Params& p, unsigned char*) : T(p.T) {}
  __device__ void init() {}
  __device__ void finish() {}
  __device__ __forceinline__ void process(bool valid, u64 key, i32 v) {
    if constexpr (!MATCH) {
      if (valid) gt_add(T, gt_slot(T, key, fmix64(key)), 1, v, v, v, vscale(v));
    } else {
      u32 peers = __match_any_sync(FULL, valid ? key : EMPTY_KEY);
      AggR a = AggR::of(valid, v);
      reduce_peers(peers, a);
      if (valid && lane_id() == (u32)__ffs(peers) - 1)
        gt_add(T, gt_slot(T, key, fmix64(key)), a.c, a.s, a.mn, a.mx, a.ds);
    }
  }
};

// ------------------------------------------------------------------ shared-memory layouts
// AoS with 64-byte slots: the "cache-line bucket" layout one would pick for a *global* table.
// In shared memory every key sits at word offset 16*s -> only 2 distinct banks -> heavy conflicts.
template <int S>
struct LAoS64 {
  struct alignas(64) Slot { u64 key; u64 sum; double dsum; u32 cnt; i32 mn; i32 mx; };
  Slot sl[S];
  __device__ __forceinline__ u64& K(u32 s) { return sl[s].key; }
  __device__ __forceinline__ u64& SUM(u32 s) { return sl[s].sum; }
  __device__ __forceinline__ double& DS(u32 s) { return sl[s].dsum; }
  __device__ __forceinline__ u32& CNT(u32 s) { return sl[s].cnt; }
  __device__ __forceinline__ i32& MN(u32 s) { return sl[s].mn; }
  __device__ __forceinline__ i32& MX(u32 s) { return sl[s].mx; }
};
// SoA: key[s] at word 2s -> 16 distinct bank pairs per half-warp, conflicts only on true collisions.
template <int S>
struct LSoA {
  u64 key[S];
  u64 sum[S];
  double dsum[S];
  u32 cnt[S];
  i32 mn[S];
  i32 mx[S];
  __device__ __forceinline__ u64& K(u32 s) { return key[s]; }
  __device__ __forceinline__ u64& SUM(u32 s) { return sum[s]; }
  __device__ __forceinline__ double& DS(u32 s) { return dsum[s]; }
  __device__ __forceinline__ u32& CNT(u32 s) { return cnt[s]; }
  __device__ __forceinline__ i32& MN(u32 s) { return mn[s]; }
  __device__ __forceinline__ i32& MX(u32 s) { return mx[s]; }
};

constexpr int SMEM_MAX_PROBES = 16;

template <class L, int S, bool MATCH>
struct PSmem {
  struct Params { GT T; };
  static constexpr size_t SMEM = (sizeof(L) + 15) / 16 * 16 + 16;
  static constexpr bool WANTS_CODE = false;
  GT T;
  L* tab;
  u32* nkeys;
  __device__ PSmem(const Params& p, unsigned char* smem)
      : T(p.T), tab(reinterpret_cast<L*>(smem)), nkeys(reinterpret_cast<u32*>(smem + SMEM - 16)) {}
  __device__ void init() {
    for (u32 s = threadIdx.x; s < (u32)S; s += blockDim.x) {
      tab->K(s) = EMPTY_KEY; tab->SUM(s) = 0; tab->DS(s) = 0.0; tab->CNT(s) = 0;
      tab->MN(s) = INT_MAX; tab->MX(s) = INT_MIN;
    }
    if (threadIdx.x == 0) *nkeys = 0;
  }
  // insert-or-update in the private table; false => caller falls back to the global table
  __device__ __forceinline__ bool try_update(u64 key, u64 h, const AggR& a) {
    u32 s = (u32)h & (S - 1);
#pragma unroll 1
    for (int p = 0; p < SMEM_MAX_PROBES; ++p) {
      u64 k = *(volatile u64*)&tab->K(s);
      if (k == EMPTY_KEY) {
        if (*(volatile u32*)nkeys >= (u32)S / 2) return false;  // load-factor cap 0.5
        atomicAdd(nkeys, 1u);
        k = atomicCAS((unsigned long long*)&tab->K(s), (unsigned long long)EMPTY_KEY, (unsigned long long)key);
        if (k == EMPTY_KEY) k = key;
      }
      if (k == key) {
        atomicAdd(&tab->CNT(s), a.c);
        atomicAdd((unsigned long long*)&tab->SUM(s), (unsigned long long)a.s);
        atomicAdd(&tab->DS(s), a.ds);
        atomicMin(&tab->MN(s), a.mn);
        atomicMax(&tab->MX(s), a.mx);
        return true;
      }
      s = (s + 1) & (S - 1);
    }
    return false;
  }
  __device__ __forceinline__ void apply(u64 key, const AggR& a) {
    u64 h = fmix64(key);
    if (!try_update(key, h, a)) gt_add(T, gt_slot(T, key, h), a.c, a.s, a.mn, a.mx, a.ds);
  }
  __device__ __forceinline__ void process(bool valid, u64 key, i32 v) {
    AggR a = AggR::of(valid, v);
    if constexpr (MATCH) {
      u32 peers = __match_any_sync(FULL, valid ? key : EMPTY_KEY);
      reduce_peers(peers, a);
      if (valid && lane_id() == (u32)__ffs(peers) - 1) apply(key, a);
    } else {
      if (valid) apply(key, a);
    }
  }
  // block-level merge of the private table into the global table
  __device__ void finish() {
    for (u32 s = threadIdx.x; s < (u32)S; s += blockDim.x) {
      u64 k = tab->K(s);
      if (k != EMPTY_KEY && tab->CNT(s))
        gt_add(T, gt_slot(T, k, fmix64(k)), tab->CNT(s), (i64)tab->SUM(s), tab->MN(s), tab->MX(s), tab->DS(s));
    }
  }
};
