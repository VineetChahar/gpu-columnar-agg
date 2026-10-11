// V4: radix partition, then aggregate each partition in a table that fits shared memory (or L2).
//
//   pass 1 (fused with decode+filter): histogram -> exclusive scan -> scatter of (key, v) by the top
//          B1 bits of fmix64(key).  Block b processes the same tiles in both kernels, so the
//          scanned [partition][block] histogram gives every block private output cursors.
//   pass 2 (optional): each pass-1 partition is split again on the next B2 bits, tile by tile.
//   agg:   one block per (partition, 8K-row chunk) work item.
//          smem mode: block-private SoA table + warp aggregation, flushed to the global table.
//          l2 mode:   warp-aggregated global atomics; partition id = top bits of the slot hash,
//                     so each partition only touches a contiguous ~1 MB slice of the global table.
// Partition buffers hold the decoded 64-bit key: decoding happens once (fused into pass 1);
// re-decoding codes in every later pass would cost a random dictionary gather per row per pass.
#include <cub/cub.cuh>
#include <vector>
#include "../gpu/frontend.cuh"
#include "variants.hpp"

namespace {

constexpr int AGG_SLOTS = 1024;  // per-work-item table: one block per item, so occupancy matters less than capacity
constexpr u32 CHUNK = 8192;     // rows per aggregation work item
constexpr u32 TILE2 = 8192;     // rows per pass-2 tile
constexpr int MAX_1PASS_BITS = 12;  // smem histogram of 4096 counters
constexpr int FANOUT_BITS = 8;  // per-pass fan-out when two passes are used

struct Scratch {
  u64 cap = 0;
  u64 *k1 = nullptr, *k2 = nullptr;
  i32 *v1 = nullptr, *v2 = nullptr;
  bool has2 = false;
  u32* hist = nullptr; size_t hist_cap = 0;
  u32* hscan = nullptr; size_t hscan_cap = 0;
  void* tmp = nullptr; size_t tmp_cap = 0;
  u32* items = nullptr; size_t items_cap = 0;  // packed work items / tile descriptors
  u32* starts = nullptr; size_t starts_cap = 0;
} S;

template <class T>
void ensure(T*& p, size_t& cap, size_t n, const char* what) {
  if (n <= cap) return;
  if (p) CK(cudaFree(p));
  p = nullptr; cap = 0;
  p = (T*)dmalloc_or_skip(n * sizeof(T), what);
  cap = n;
}
void ensure_tmp(size_t bytes) {
  if (bytes <= S.tmp_cap) return;
  if (S.tmp) CK(cudaFree(S.tmp));
  S.tmp = nullptr; S.tmp_cap = 0;
  S.tmp = dmalloc_or_skip(bytes, "cub tmp");
  S.tmp_cap = bytes;
}

// ---------------- pass-1 processors (FE_VEC, decoded key) ----------------
struct PHist {
  struct Params { u32* hist; int bits; };
  static constexpr size_t SMEM = (1u << MAX_1PASS_BITS) * 4;
  static constexpr bool WANTS_CODE = false;
  u32* g; u32* h; int bits;
  __device__ PHist(const Params& p, unsigned char* smem) : g(p.hist), h((u32*)smem), bits(p.bits) {}
  __device__ void init() { for (u32 i = threadIdx.x; i < (1u << bits); i += blockDim.x) h[i] = 0; }
  __device__ __forceinline__ void process(bool valid, u64 key, i32) {
    u32 p = valid ? (u32)(fmix64(key) >> (64 - bits)) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    if (valid && lane_id() == (u32)__ffs(peers) - 1) atomicAdd(&h[p], (u32)__popc(peers));
  }
  __device__ void finish() {
    for (u32 i = threadIdx.x; i < (1u << bits); i += blockDim.x) g[(u64)i * gridDim.x + blockIdx.x] = h[i];
  }
};
struct PScatter {
  struct Params { const u32* hscan; int bits; u64* ok; i32* ov; };
  static constexpr size_t SMEM = (1u << MAX_1PASS_BITS) * 4;
  static constexpr bool WANTS_CODE = false;
  const u32* hs; u32* cur; int bits; u64* ok; i32* ov;
  __device__ PScatter(const Params& p, unsigned char* smem) : hs(p.hscan), cur((u32*)smem), bits(p.bits), ok(p.ok), ov(p.ov) {}
  __device__ void init() {
    for (u32 i = threadIdx.x; i < (1u << bits); i += blockDim.x) cur[i] = hs[(u64)i * gridDim.x + blockIdx.x];
  }
  __device__ __forceinline__ void process(bool valid, u64 key, i32 v) {
    u32 p = valid ? (u32)(fmix64(key) >> (64 - bits)) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    u32 leader = __ffs(peers) - 1;
    u32 base = 0;
    if (valid && lane_id() == leader) base = atomicAdd(&cur[p], (u32)__popc(peers));
    base = __shfl_sync(FULL, base, leader);
    if (valid) {
      u32 pos = base + __popc(peers & lanemask_lt());
      ok[pos] = key;
      ov[pos] = v;
    }
  }
  __device__ void finish() {}
};

// ---------------- pass 2: tile-wise re-partition of each pass-1 segment ----------------
// tile descriptor: [seg, a, b] ; per segment: tb (first tile), nt (tiles), start
struct P2 {
  const u32* tiles;  // 3 per tile
  const u32* seg_tb;
  const u32* seg_nt;
  int b1, b2;
};
__device__ __forceinline__ u32 p2_index(const P2& q, u32 seg, u32 bucket, u32 tile) {
  u32 tb = q.seg_tb[seg];
  return tb * (1u << q.b2) + bucket * q.seg_nt[seg] + (tile - tb);
}
__global__ void __launch_bounds__(256) k_p2_hist(P2 q, const u64* keys, u32* hist) {
  __shared__ u32 h[1 << FANOUT_BITS];
  const u32 t = blockIdx.x, seg = q.tiles[3 * t], a = q.tiles[3 * t + 1], b = q.tiles[3 * t + 2];
  for (u32 i = threadIdx.x; i < (1u << q.b2); i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (u32 base = a; base < b; base += blockDim.x) {
    u32 i = base + threadIdx.x;
    bool valid = i < b;
    u32 p = valid ? (u32)(fmix64(keys[i]) >> (64 - q.b1 - q.b2)) & ((1u << q.b2) - 1) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    if (valid && lane_id() == (u32)__ffs(peers) - 1) atomicAdd(&h[p], (u32)__popc(peers));
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < (1u << q.b2); i += blockDim.x) hist[p2_index(q, seg, i, t)] = h[i];
}
__global__ void __launch_bounds__(256) k_p2_scatter(P2 q, const u64* keys, const i32* vals, const u32* hs,
                                                     u64* ok, i32* ov) {
  __shared__ u32 cur[1 << FANOUT_BITS];
  const u32 t = blockIdx.x, seg = q.tiles[3 * t], a = q.tiles[3 * t + 1], b = q.tiles[3 * t + 2];
  for (u32 i = threadIdx.x; i < (1u << q.b2); i += blockDim.x) cur[i] = hs[p2_index(q, seg, i, t)];
  __syncthreads();
  for (u32 base = a; base < b; base += blockDim.x) {
    u32 i = base + threadIdx.x;
    bool valid = i < b;
    u64 key = valid ? keys[i] : 0;
    u32 p = valid ? (u32)(fmix64(key) >> (64 - q.b1 - q.b2)) & ((1u << q.b2) - 1) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    u32 leader = __ffs(peers) - 1, pos0 = 0;
    if (valid && lane_id() == leader) pos0 = atomicAdd(&cur[p], (u32)__popc(peers));
    pos0 = __shfl_sync(FULL, pos0, leader);
    if (valid) {
      u32 pos = pos0 + __popc(peers & lanemask_lt());
      ok[pos] = key;
      ov[pos] = vals[i];
    }
  }
}
// final partition starts after pass 2 (P = 2^(b1+b2) partitions, starts[P] = total)
__global__ void k_p2_starts(P2 q, const u32* hs, const u32* seg_start, u32 total, u32* starts) {
  u32 P = 1u << (q.b1 + q.b2);
  for (u32 p = blockIdx.x * blockDim.x + threadIdx.x; p <= P; p += gridDim.x * blockDim.x) {
    if (p == P) { starts[p] = total; continue; }
    u32 seg = p >> q.b2, bucket = p & ((1u << q.b2) - 1);
    starts[p] = q.seg_nt[seg] ? hs[p2_index(q, seg, bucket, q.seg_tb[seg])] : seg_start[seg];
  }
}
// pass-1 partition starts: starts[p] = hscan[p * nblk], starts[P] = total
__global__ void k_p1_starts(const u32* hs, u32 nblk, u32 P, u32 total, u32* starts) {
  for (u32 p = blockIdx.x * blockDim.x + threadIdx.x; p <= P; p += gridDim.x * blockDim.x)
    starts[p] = p == P ? total : hs[(u64)p * nblk];
}

// ---------------- aggregation over partitioned (key, v) ----------------
template <class P>
__global__ void __launch_bounds__(256) k_part_agg(const u32* items, const u64* keys, const i32* vals,
                                                   typename P::Params prm) {
  extern __shared__ __align__(16) unsigned char smem[];
  P p(prm, smem);
  p.init();
  __syncthreads();
  const u32 a = items[2 * blockIdx.x], b = items[2 * blockIdx.x + 1];
  for (u32 base = a; base < b; base += blockDim.x) {
    u32 i = base + threadIdx.x;
    bool valid = i < b;
    p.process(valid, valid ? keys[i] : EMPTY_KEY, valid ? vals[i] : 0);
  }
  __syncthreads();
  p.finish();
}

int ceil_log2(double x) { int b = 0; while ((double)(1ull << b) < x) ++b; return b; }

void scan_u32(const u32* in, u32* out, size_t n, cudaStream_t st) {
  size_t tb = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, in, out, (int)n, st));
  ensure_tmp(tb);
  CK(cub::DeviceScan::ExclusiveSum(S.tmp, tb, in, out, (int)n, st));
}

}  // namespace

void release_v4_scratch() {
  for (void* p : {(void*)S.k1, (void*)S.k2, (void*)S.v1, (void*)S.v2, (void*)S.hist, (void*)S.hscan, S.tmp,
                  (void*)S.items, (void*)S.starts})
    if (p) CK(cudaFree(p));
  S = Scratch{};
}

void run_v4_radix(RunCtx& c, bool l2_mode, bool force_1pass) {
  const DevData& D = *c.D;
  // ---- plan (host, from statistics) ----
  double groups_est = std::max<double>(1.0, std::min<double>(D.card, (double)c.sel_est));
  int B;
  if (l2_mode) B = ceil_log2((double)c.T->bytes() / (1 << 20));             // <= 1 MB table slice
  else B = ceil_log2(groups_est / (AGG_SLOTS / 4));                           // <= S/4 groups/partition
  int B1 = B, B2 = 0;
  if (B > FANOUT_BITS && !force_1pass) { B1 = (B + 1) / 2; B2 = B - B1; }
  if (B1 > MAX_1PASS_BITS) throw Skip("1-pass fan-out > 2^12 needs a >16KB histogram");
  if (B2 > FANOUT_BITS) throw Skip("needs a third partition pass");
  PhaseTimer pt(c.st);
  c.info["partition_bits"] = B; c.info["passes"] = B == 0 ? 0 : (B2 ? 2 : 1);

  table_init(*c.T, c.st);
  if (B == 0) {  // everything fits: plain V3
    with_kd(D, [&](auto kd) {
      auto C = make_cols(D, kd, c.lo, c.hi);
      launch_agg<decltype(kd), PSmem<LSoA<DEFAULT_SLOTS>, DEFAULT_SLOTS, true>, FE_VEC_COMPACT>(C, {c.T->t}, c.st, c.block);
    });
    table_extract(*c.T, *c.R, c.st);
    return;
  }
  // ---- buffers (allocated once per dataset size; allocation is not timed after the first run) ----
  if (S.cap < D.n) {
    release_v4_scratch();
    S.cap = D.n;
    S.k1 = (u64*)dmalloc_or_skip(D.n * 8, "v4 partition buffer");
    S.v1 = (i32*)dmalloc_or_skip(D.n * 4, "v4 partition buffer");
  }
  if (B2 && !S.has2) {
    S.k2 = (u64*)dmalloc_or_skip(S.cap * 8, "v4 pass-2 buffer");
    S.v2 = (i32*)dmalloc_or_skip(S.cap * 4, "v4 pass-2 buffer");
    S.has2 = true;
  }

  // ---- pass 1: fused decode + filter + partition ----
  const int P1 = 1 << B1;
  int nblk = 0;
  u32 total = 0;
  with_kd(D, [&](auto kd) {
    using KD = decltype(kd);
    auto C = make_cols(D, kd, c.lo, c.hi);
    int bps;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_agg<KD, PHist, FE_VEC>, c.block, PHist::SMEM));
    nblk = bps * num_sms();
    ensure(S.hist, S.hist_cap, (size_t)P1 * nblk + 1, "hist");
    ensure(S.hscan, S.hscan_cap, (size_t)P1 * nblk + 1, "hist");
    launch_agg<KD, PHist, FE_VEC>(C, {S.hist, B1}, c.st, c.block, nblk);
    scan_u32(S.hist, S.hscan, (size_t)P1 * nblk, c.st);
    u32 last[2];
    CK(cudaMemcpyAsync(&last[0], S.hist + (size_t)P1 * nblk - 1, 4, cudaMemcpyDeviceToHost, c.st));
    CK(cudaMemcpyAsync(&last[1], S.hscan + (size_t)P1 * nblk - 1, 4, cudaMemcpyDeviceToHost, c.st));
    launch_agg<KD, PScatter, FE_VEC>(C, {S.hscan, B1, S.k1, S.v1}, c.st, c.block, nblk);
    pt.mark("pass1");
    CK(cudaStreamSynchronize(c.st));
    total = last[0] + last[1];
  });
  c.info["selected"] = total;

  const u64* fk = S.k1;
  const i32* fv = S.v1;
  const int P = 1 << B;
  std::vector<u32> starts(P + 1);
  ensure(S.starts, S.starts_cap, (size_t)P + 1 + 2 * P1, "starts");
  k_p1_starts<<<64, 256, 0, c.st>>>(S.hscan, nblk, P1, total, S.starts);
  if (B2) {
    std::vector<u32> s1(P1 + 1);
    CK(cudaMemcpyAsync(s1.data(), S.starts, (P1 + 1) * 4, cudaMemcpyDeviceToHost, c.st));
    CK(cudaStreamSynchronize(c.st));
    std::vector<u32> tiles, seg_tb(P1), seg_nt(P1);
    for (int s = 0; s < P1; ++s) {
      seg_tb[s] = (u32)(tiles.size() / 3);
      u32 nt = 0;
      for (u32 a = s1[s]; a < s1[s + 1]; a += TILE2, ++nt) {
        tiles.push_back(s); tiles.push_back(a); tiles.push_back(std::min(a + TILE2, s1[s + 1]));
      }
      seg_nt[s] = nt;
    }
    u32 ntiles = (u32)(tiles.size() / 3);
    // device layout in S.items: tiles | seg_tb | seg_nt | seg_start
    ensure(S.items, S.items_cap, tiles.size() + 3 * P1 + 8, "tiles");
    CK(cudaMemcpyAsync(S.items, tiles.data(), tiles.size() * 4, cudaMemcpyHostToDevice, c.st));
    u32* d_tb = S.items + tiles.size();
    u32* d_nt = d_tb + P1;
    u32* d_ss = d_nt + P1;
    CK(cudaMemcpyAsync(d_tb, seg_tb.data(), P1 * 4, cudaMemcpyHostToDevice, c.st));
    CK(cudaMemcpyAsync(d_nt, seg_nt.data(), P1 * 4, cudaMemcpyHostToDevice, c.st));
    CK(cudaMemcpyAsync(d_ss, s1.data(), P1 * 4, cudaMemcpyHostToDevice, c.st));
    P2 q{S.items, d_tb, d_nt, B1, B2};
    size_t hn = (size_t)ntiles << B2;
    ensure(S.hist, S.hist_cap, hn + 1, "hist2");
    ensure(S.hscan, S.hscan_cap, hn + 1, "hist2");
    if (ntiles) {
      k_p2_hist<<<ntiles, 256, 0, c.st>>>(q, S.k1, S.hist);
      scan_u32(S.hist, S.hscan, hn, c.st);
      k_p2_scatter<<<ntiles, 256, 0, c.st>>>(q, S.k1, S.v1, S.hscan, S.k2, S.v2);
    }
    ensure(S.starts, S.starts_cap, (size_t)P + 1, "starts");
    k_p2_starts<<<64, 256, 0, c.st>>>(q, S.hscan, d_ss, total, S.starts);
    CK(cudaGetLastError());
    fk = S.k2;
    fv = S.v2;
    pt.mark("pass2");
    c.info["pass2_tiles"] = ntiles;
  }
  CK(cudaMemcpyAsync(starts.data(), S.starts, (P + 1) * 4, cudaMemcpyDeviceToHost, c.st));
  CK(cudaStreamSynchronize(c.st));

  pt.mark("plan");
  // ---- aggregation work items: (a, b) chunks in partition order ----
  std::vector<u32> items;
  u32 maxp = 0;
  for (int p = 0; p < P; ++p) {
    maxp = std::max(maxp, starts[p + 1] - starts[p]);
    for (u32 a = starts[p]; a < starts[p + 1]; a += CHUNK) { items.push_back(a); items.push_back(std::min(a + CHUNK, starts[p + 1])); }
  }
  c.info["max_partition_rows"] = maxp;
  u32 nitems = (u32)(items.size() / 2);
  ensure(S.items, S.items_cap, items.size() + 2, "items");
  if (nitems) {
    CK(cudaMemcpyAsync(S.items, items.data(), items.size() * 4, cudaMemcpyHostToDevice, c.st));
    if (l2_mode) {
      using PG = PGlobal<true>;
      k_part_agg<PG><<<nitems, 256, 0, c.st>>>(S.items, fk, fv, PG::Params{c.T->t});
    } else {
      using PS = PSmem<LSoA<AGG_SLOTS>, AGG_SLOTS, true>;
      static bool attr = false;
      if (!attr) { CK(cudaFuncSetAttribute(k_part_agg<PS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024)); attr = true; }
      k_part_agg<PS><<<nitems, 256, PS::SMEM, c.st>>>(S.items, fk, fv, PS::Params{c.T->t});
    }
    CK(cudaGetLastError());
  }
  c.info["work_items"] = nitems;
  pt.mark("aggregate");
  table_extract(*c.T, *c.R, c.st);
  pt.mark("extract");
  pt.dump(c.info);
}
