// Scan front-ends: fused RLE-decode + predicate + dictionary decode, feeding a processor.
//   FE_SCALAR       one row per lane per step, scalar loads              (V0)
//   FE_VEC          four rows per lane, int4/uint4 vector loads          (V1-nocompact, partition passes)
//   FE_VEC_COMPACT  FE_VEC + ballot/popc stream compaction into a warp ring buffer so the
//                   aggregation step always runs with 32 active lanes  (V1..V4)
//   FE_VEC_COMPACT_LATE  same, but the RLE predicate is decoded first and a lane loads its codes and
//                   values only if one of its 4 rows survives (late materialisation)  (v3_latemat)
#pragma once
#include "../encoding/rle.cuh"
#include "procs.cuh"

enum FEMode { FE_SCALAR = 0, FE_VEC = 1, FE_VEC_COMPACT = 2, FE_VEC_COMPACT_LATE = 3 };

template <class KD, class P>
struct DirectSink {
  const KD& kd;
  P& p;
  __device__ __forceinline__ void push1(bool ok, u32 code, i32 v) {
    u64 key;
    if constexpr (P::WANTS_CODE) key = code;
    else key = ok ? kd.decode(code) : EMPTY_KEY;
    p.process(ok, key, v);
  }
  __device__ __forceinline__ void push4(const bool ok[4], const u32 code[4], const i32 v[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) push1(ok[j], code[j], v[j]);
  }
  __device__ void flush() {}
};

constexpr int WBUF = 256;  // ring entries per warp (max fill before drain is 31 + 128)
struct WarpBuf {
  u32 code[WBUF];
  i32 val[WBUF];
};

template <class KD, class P>
struct CompactSink {
  const KD& kd;
  P& p;
  WarpBuf* buf;
  u32 head = 0, cnt = 0;  // warp-uniform
  __device__ CompactSink(const KD& k, P& pp, WarpBuf* b) : kd(k), p(pp), buf(b) {}
  __device__ __forceinline__ void drain_one(bool ok, u32 i) {
    u32 c = buf->code[i & (WBUF - 1)];
    i32 v = buf->val[i & (WBUF - 1)];
    __syncwarp();
    u64 key;
    if constexpr (P::WANTS_CODE) key = c;
    else key = ok ? kd.decode(c) : EMPTY_KEY;
    p.process(ok, key, v);
  }
  __device__ __forceinline__ void push4(const bool ok[4], const u32 code[4], const i32 v[4]) {
    u32 off = cnt;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      u32 b = __ballot_sync(FULL, ok[j]);
      if (ok[j]) {
        u32 pos = (head + off + __popc(b & lanemask_lt())) & (WBUF - 1);
        buf->code[pos] = code[j];
        buf->val[pos] = v[j];
      }
      off += __popc(b);
    }
    cnt = off;
    __syncwarp();
    while (cnt >= 32) {
      drain_one(true, head + lane_id());
      head = (head + 32) & (WBUF - 1);
      cnt -= 32;
    }
  }
  __device__ void push1(bool ok, u32 code, i32 v) {
    bool o[4] = {ok, false, false, false};
    u32 c[4] = {code, 0, 0, 0};
    i32 vv[4] = {v, 0, 0, 0};
    push4(o, c, vv);
  }
  __device__ void flush() {
    if (cnt) drain_one(lane_id() < cnt, head + lane_id());
    cnt = 0;
  }
};

template <class KD, class Sink>
__device__ void fe_scalar(const Cols<KD>& C, Sink& sink) {
  const u32 lane = lane_id();
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  const u64 ntiles = (C.n + TILE - 1) / TILE;
  for (u64 t = gw; t < ntiles; t += nw) {
    const u64 tb = t * TILE, te = min(tb + TILE, C.n);
    u32 cur = rle_upper_bound(C.run_end, C.nruns, C.row0 + tb);
    for (u64 b = tb; b < te; b += 32) {
      const u64 r = b + lane;
      const bool in = r < te;
      u32 code = in ? C.kd.code(r) : 0;
      i32 v = in ? C.vals[r] : 0;
      u32 run = warp_rle_find(C.run_end, C.nruns, cur, C.row0 + (in ? r : te - 1));
      u16 p = C.run_val[run];
      bool ok = in && p >= C.lo && p < C.hi;
      cur = __shfl_sync(FULL, run, 31);
      sink.push1(ok, code, v);
    }
  }
}

template <class KD, class Sink, bool LATE = false>
__device__ void fe_vec(const Cols<KD>& C, Sink& sink) {
  const u32 lane = lane_id();
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  const u64 ntiles = (C.n + TILE - 1) / TILE;
  for (u64 t = gw; t < ntiles; t += nw) {
    const u64 tb = t * TILE, te = min(tb + TILE, C.n);
    u32 cur = rle_upper_bound(C.run_end, C.nruns, C.row0 + tb);
    for (u64 b = tb; b < te; b += 128) {
      const u64 r = b + 4 * lane;
      u32 code[4] = {0, 0, 0, 0};
      i32 v[4] = {0, 0, 0, 0};
      if (!LATE && r < te) {
        C.kd.code4(r, code);
        int4 w = *reinterpret_cast<const int4*>(C.vals + r);
        v[0] = w.x; v[1] = w.y; v[2] = w.z; v[3] = w.w;
      }
      u32 run = warp_rle_find(C.run_end, C.nruns, cur, C.row0 + (r < te ? r : te - 1));
      bool ok[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const u64 row = r + j;
        ok[j] = false;
        if (row < te) {
          while (C.run_end[run] <= C.row0 + row) ++run;
          u16 p = C.run_val[run];
          ok[j] = p >= C.lo && p < C.hi;
        }
      }
      cur = __shfl_sync(FULL, run, 31);
      if (LATE && (ok[0] | ok[1] | ok[2] | ok[3])) {  // untouched sectors are never fetched
        C.kd.code4(r, code);
        int4 w = *reinterpret_cast<const int4*>(C.vals + r);
        v[0] = w.x; v[1] = w.y; v[2] = w.z; v[3] = w.w;
      }
      sink.push4(ok, code, v);
    }
  }
}

template <class KD, class P, int MODE>
__global__ void __launch_bounds__(256) k_agg(Cols<KD> C, typename P::Params prm) {
  extern __shared__ __align__(16) unsigned char smem[];
  P p(prm, smem);
  p.init();
  __syncthreads();
  if constexpr (MODE == FE_SCALAR) {
    DirectSink<KD, P> s{C.kd, p};
    fe_scalar(C, s);
  } else if constexpr (MODE == FE_VEC) {
    DirectSink<KD, P> s{C.kd, p};
    fe_vec(C, s);
  } else {
    CompactSink<KD, P> s(C.kd, p, reinterpret_cast<WarpBuf*>(smem + P::SMEM) + (threadIdx.x >> 5));
    fe_vec<KD, CompactSink<KD, P>, MODE == FE_VEC_COMPACT_LATE>(C, s);
    s.flush();
  }
  __syncthreads();
  p.finish();
}

struct LaunchInfo {
  int grid = 0, block = 0, blocks_per_sm = 0;
  size_t smem = 0;
  double occupancy = 0;  // active warps / max warps per SM
};

// Launch k_agg with a grid sized from the occupancy calculator (one full wave, persistent-style).
template <class KD, class P, int MODE>
LaunchInfo launch_agg(const Cols<KD>& C, const typename P::Params& prm, cudaStream_t st, int block = 256,
                      int grid_override = 0) {
  auto kern = k_agg<KD, P, MODE>;
  LaunchInfo L;
  L.block = block;
  L.smem = P::SMEM + (MODE >= FE_VEC_COMPACT ? (block / 32) * sizeof(WarpBuf) : 0);
  static bool attr_set = false;
  if (!attr_set) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    attr_set = true;
  }
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&L.blocks_per_sm, kern, block, L.smem));
  if (L.blocks_per_sm < 1) throw Skip("does not fit on an SM");
  L.grid = grid_override ? grid_override : L.blocks_per_sm * num_sms();
  L.occupancy = L.blocks_per_sm * block / 1024.0;  // sm_75: 32 warps / 1024 threads per SM
  kern<<<L.grid, block, L.smem, st>>>(C, prm);
  CK(cudaGetLastError());
  return L;
}
