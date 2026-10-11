// V4-sort: sort-based aggregation.  Fused decode+filter compacts (code, v) of selected rows
// (warp-aggregated append), cub::DeviceRadixSort sorts on the *dictionary code* - only
// log2(card) bits, so low-cardinality sorts take one or two digit passes - then
// cub::DeviceReduce::ReduceByKey folds each run, and only the unique codes are decoded.
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include "../gpu/frontend.cuh"
#include "variants.hpp"

namespace {

struct PCompact {
  struct Params { u32* oc; i32* ov; unsigned long long* counter; };
  static constexpr size_t SMEM = 0;
  static constexpr bool WANTS_CODE = true;
  Params q;
  __device__ PCompact(const Params& p, unsigned char*) : q(p) {}
  __device__ void init() {}
  __device__ void finish() {}
  __device__ __forceinline__ void process(bool valid, u64 code, i32 v) {
    u32 b = __ballot_sync(FULL, valid);
    unsigned long long base = 0;
    if (lane_id() == 0 && b) base = atomicAdd(q.counter, (unsigned long long)__popc(b));
    base = __shfl_sync(FULL, base, 0);
    if (valid) {
      u64 pos = base + __popc(b & lanemask_lt());
      q.oc[pos] = (u32)code;
      q.ov[pos] = v;
    }
  }
};

struct ToAgg {
  __host__ __device__ Agg operator()(i32 v) const { return Agg{1ull, (i64)v, v, v, vscale(v)}; }
};
struct MergeAgg {
  __host__ __device__ Agg operator()(const Agg& a, const Agg& b) const {
    return Agg{a.cnt + b.cnt, a.sum + b.sum, a.mn < b.mn ? a.mn : b.mn, a.mx > b.mx ? a.mx : b.mx, a.dsum + b.dsum};
  }
};

template <class KD>
__global__ void k_decode_out(KD kd, const u32* codes, const Agg* aggs, const u32* nruns, ResV R) {
  u32 n = *nruns;
  if (blockIdx.x == 0 && threadIdx.x == 0) *R.n = n;
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    Agg a = aggs[i];
    R.key[i] = kd.decode(codes[i]);
    R.cnt[i] = a.cnt; R.sum[i] = a.sum; R.mn[i] = a.mn; R.mx[i] = a.mx; R.dsum[i] = a.dsum;
  }
}

struct SortScratch {
  u64 cap = 0;
  u32 *c0 = nullptr, *c1 = nullptr, *uniq = nullptr, *nruns = nullptr;
  i32 *v0 = nullptr, *v1 = nullptr;
  Agg* aggs = nullptr;
  unsigned long long* counter = nullptr;
  void* tmp = nullptr;
  size_t tmp_bytes = 0;
  u64 agg_cap = 0;
};
thread_local SortScratch S;  // per host thread = per GPU in the multi-GPU runner

// --- chunked plan: dense per-code accumulator, merged chunk by chunk ---
__global__ void k_acc_init(Agg* acc, u64 n) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x)
    acc[i] = Agg{0, 0, INT_MAX, INT_MIN, 0.0};
}
// codes within one chunk's ReduceByKey output are unique, so no two threads touch the same entry
__global__ void k_acc_merge(Agg* acc, const u32* uniq, const Agg* aggs, const u32* nruns) {
  u32 n = *nruns;
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    acc[uniq[i]] = MergeAgg()(acc[uniq[i]], aggs[i]);
}
template <class KD>
__global__ void k_acc_extract(KD kd, const Agg* acc, u64 card, ResV R) {
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  for (u64 base = gw * 32; base < card; base += nw * 32) {
    u64 k = base + lane_id();
    Agg a = k < card ? acc[k] : Agg{0, 0, 0, 0, 0.0};
    bool ok = a.cnt != 0;
    u32 b = __ballot_sync(FULL, ok);
    u32 off = 0;
    if (lane_id() == 0 && b) off = atomicAdd(R.n, __popc(b));
    off = __shfl_sync(FULL, off, 0);
    if (ok) {
      u32 i = off + __popc(b & lanemask_lt());
      R.key[i] = kd.decode((u32)k);
      R.cnt[i] = a.cnt; R.sum[i] = a.sum; R.mn[i] = a.mn; R.mx[i] = a.mx; R.dsum[i] = a.dsum;
    }
  }
}
struct ChunkScratch {
  u64 cap = 0, card = 0;
  u32 *c0 = nullptr, *c1 = nullptr, *uniq = nullptr, *nruns = nullptr;
  i32 *v0 = nullptr, *v1 = nullptr;
  Agg *aggs = nullptr, *acc = nullptr;
  unsigned long long* counter = nullptr;
  void* tmp = nullptr;
  size_t tmp_bytes = 0;
  u64 plan_n = ~0ull, plan_sel = ~0ull, plan_force = ~0ull, plan_chunk = 0, plan_buf = 0;  // cached plan
};
thread_local ChunkScratch K;

size_t sort_tmp_bytes(u64 n, cudaStream_t st) {
  size_t a = 0, b = 0;
  CK(cub::DeviceRadixSort::SortPairs(nullptr, a, (u32*)nullptr, (u32*)nullptr, (i32*)nullptr, (i32*)nullptr, (int)n, 0, 32, st));
  auto it = thrust::make_transform_iterator((i32*)nullptr, ToAgg());
  CK(cub::DeviceReduce::ReduceByKey(nullptr, b, (u32*)nullptr, (u32*)nullptr, it, (Agg*)nullptr, (u32*)nullptr, MergeAgg(), (int)n, st));
  return std::max(a, b);
}

}  // namespace

void release_sort_scratch() {
  for (void* p : {(void*)K.c0, (void*)K.c1, (void*)K.uniq, (void*)K.nruns, (void*)K.v0, (void*)K.v1, (void*)K.aggs,
                  (void*)K.acc, (void*)K.counter, K.tmp})
    if (p) CK(cudaFree(p));
  K = ChunkScratch{};
  for (void* p : {(void*)S.c0, (void*)S.c1, (void*)S.uniq, (void*)S.nruns, (void*)S.v0, (void*)S.v1, (void*)S.aggs,
                  (void*)S.counter, S.tmp})
    if (p) CK(cudaFree(p));
  S = SortScratch{};
}

void run_v4_sort(RunCtx& c) {
  const DevData& D = *c.D;
  const int bits = std::max(1, ilog2(next_pow2(D.card)));
  if (S.cap < D.n || S.agg_cap < D.card) {
    release_sort_scratch();
    S.cap = D.n;
    S.agg_cap = D.card;
    S.c0 = (u32*)dmalloc_or_skip(D.n * 4, "sort buffers");
    S.c1 = (u32*)dmalloc_or_skip(D.n * 4, "sort buffers");
    S.v0 = (i32*)dmalloc_or_skip(D.n * 4, "sort buffers");
    S.v1 = (i32*)dmalloc_or_skip(D.n * 4, "sort buffers");
    S.uniq = (u32*)dmalloc_or_skip((size_t)D.card * 4, "sort buffers");
    S.aggs = (Agg*)dmalloc_or_skip((size_t)D.card * sizeof(Agg), "sort buffers");
    S.nruns = (u32*)dmalloc_or_skip(4, "sort buffers");
    S.counter = (unsigned long long*)dmalloc_or_skip(8, "sort buffers");
    size_t a = 0, b = 0;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, a, S.c0, S.c1, S.v0, S.v1, (int)D.n, 0, 32, c.st));
    auto it = thrust::make_transform_iterator(S.v1, ToAgg());
    CK(cub::DeviceReduce::ReduceByKey(nullptr, b, S.c1, S.uniq, it, S.aggs, S.nruns, MergeAgg(), (int)D.n, c.st));
    S.tmp_bytes = std::max(a, b);
    S.tmp = dmalloc_or_skip(S.tmp_bytes, "sort tmp");
  }
  CK(cudaMemsetAsync(S.counter, 0, 8, c.st));
  u64 nsel = 0;
  with_kd(D, [&](auto kd) {
    auto C = make_cols(D, kd, c.lo, c.hi);
    auto L = launch_agg<decltype(kd), PCompact, FE_VEC_COMPACT>(C, {S.c0, S.v0, S.counter}, c.st, c.block);
    c.info["grid"] = L.grid;
    CK(cudaMemcpyAsync(&nsel, S.counter, 8, cudaMemcpyDeviceToHost, c.st));
    CK(cudaStreamSynchronize(c.st));
    size_t tb = S.tmp_bytes;
    if (nsel) {
      CK(cub::DeviceRadixSort::SortPairs(S.tmp, tb, S.c0, S.c1, S.v0, S.v1, (int)nsel, 0, bits, c.st));
      auto it = thrust::make_transform_iterator(S.v1, ToAgg());
      tb = S.tmp_bytes;
      CK(cub::DeviceReduce::ReduceByKey(S.tmp, tb, S.c1, S.uniq, it, S.aggs, S.nruns, MergeAgg(), (int)nsel, c.st));
    } else {
      CK(cudaMemsetAsync(S.nruns, 0, 4, c.st));
    }
    k_decode_out<<<num_sms() * 4, 256, 0, c.st>>>(kd, S.uniq, S.aggs, S.nruns, c.R->r);
    CK(cudaGetLastError());
  });
  c.info["selected"] = (double)nsel;
  c.info["sort_bits"] = bits;
}

// V4-sort, chunked: the same sort + ReduceByKey, but over row chunks whose buffers fit in free device
// memory.  Each chunk's (unique code, partial aggregate) runs are folded into a dense per-code
// accumulator (card x 32 B) - conflict-free, since codes are unique within a chunk - and the
// accumulator is compacted + decoded once at the end.  When the planner's exact selected-row count
// fits the budget, there is one chunk whose buffers are sized to the selected rows only.
void run_v4_sort_chunked(RunCtx& c) {
  const DevData& D = *c.D;
  const int bits = std::max(1, ilog2(next_pow2(D.card)));
  if (K.card < D.card) {
    release_sort_scratch();
    K.card = D.card;
    K.acc = (Agg*)dmalloc_or_skip((size_t)D.card * sizeof(Agg), "chunk accumulator");
    K.uniq = (u32*)dmalloc_or_skip((size_t)D.card * 4, "chunk accumulator");
    K.aggs = (Agg*)dmalloc_or_skip((size_t)D.card * sizeof(Agg), "chunk accumulator");
    K.nruns = (u32*)dmalloc_or_skip(4, "chunk accumulator");
    K.counter = (unsigned long long*)dmalloc_or_skip(8, "chunk accumulator");
  }
  // plan (once per dataset/selectivity; not repeated inside timed iterations):
  // buffer budget 16 B/row (two code + two value buffers) + cub temp, within 85% of free memory
  if (K.plan_n != D.n || K.plan_sel != c.sel_est || K.plan_force != c.chunk_rows) {
    u64 rows_cap;
    if (c.chunk_rows) rows_cap = c.chunk_rows;
    else {
      size_t fr, tot;
      CK(cudaMemGetInfo(&fr, &tot));
      fr += K.cap * 16 + K.tmp_bytes;  // what we already hold counts as available
      rows_cap = std::max<u64>(c.sel_est, 1);
      while (rows_cap > TILE && rows_cap * 16 + sort_tmp_bytes(rows_cap, c.st) > fr * 0.85) rows_cap /= 2;
    }
    const bool one_chunk = c.sel_est <= rows_cap;  // sel_est is exact (RLE statistics)
    K.plan_chunk = one_chunk ? D.n : std::max<u64>(TILE, rows_cap / TILE * TILE);
    K.plan_buf = one_chunk ? std::max<u64>(c.sel_est, 1) : K.plan_chunk;
    K.plan_n = D.n; K.plan_sel = c.sel_est; K.plan_force = c.chunk_rows;
  }
  const u64 chunk = std::max<u64>(K.plan_chunk, 1), buf = K.plan_buf;
  if (K.cap < buf) {
    for (void* p : {(void*)K.c0, (void*)K.c1, (void*)K.v0, (void*)K.v1, K.tmp}) if (p) CK(cudaFree(p));
    K.c0 = K.c1 = nullptr; K.v0 = K.v1 = nullptr; K.tmp = nullptr; K.cap = 0;
    K.c0 = (u32*)dmalloc_or_skip(buf * 4, "chunk buffers");
    K.c1 = (u32*)dmalloc_or_skip(buf * 4, "chunk buffers");
    K.v0 = (i32*)dmalloc_or_skip(buf * 4, "chunk buffers");
    K.v1 = (i32*)dmalloc_or_skip(buf * 4, "chunk buffers");
    K.tmp_bytes = sort_tmp_bytes(buf, c.st);
    K.tmp = dmalloc_or_skip(K.tmp_bytes, "chunk tmp");
    K.cap = buf;
  }
  k_acc_init<<<num_sms() * 8, 256, 0, c.st>>>(K.acc, D.card);
  CK(cudaGetLastError());
  u64 nchunks = 0, total_sel = 0;
  with_kd(D, [&](auto kd) {
    for (u64 r0 = 0; r0 < D.n; r0 += chunk, ++nchunks) {
      u64 len = std::min(chunk, D.n - r0);
      CK(cudaMemsetAsync(K.counter, 0, 8, c.st));
      auto C = make_cols_range(D, kd, c.lo, c.hi, r0, len);
      launch_agg<decltype(kd), PCompact, FE_VEC_COMPACT>(C, {K.c0, K.v0, K.counter}, c.st, c.block);
      u64 nsel = 0;
      CK(cudaMemcpyAsync(&nsel, K.counter, 8, cudaMemcpyDeviceToHost, c.st));
      CK(cudaStreamSynchronize(c.st));
      total_sel += nsel;
      if (!nsel) continue;
      size_t tb = K.tmp_bytes;
      CK(cub::DeviceRadixSort::SortPairs(K.tmp, tb, K.c0, K.c1, K.v0, K.v1, (int)nsel, 0, bits, c.st));
      auto it = thrust::make_transform_iterator(K.v1, ToAgg());
      tb = K.tmp_bytes;
      CK(cub::DeviceReduce::ReduceByKey(K.tmp, tb, K.c1, K.uniq, it, K.aggs, K.nruns, MergeAgg(), (int)nsel, c.st));
      k_acc_merge<<<num_sms() * 8, 256, 0, c.st>>>(K.acc, K.uniq, K.aggs, K.nruns);
      CK(cudaGetLastError());
    }
    CK(cudaMemsetAsync(c.R->r.n, 0, 4, c.st));
    k_acc_extract<<<num_sms() * 8, 256, 0, c.st>>>(kd, K.acc, D.card, c.R->r);
    CK(cudaGetLastError());
  });
  c.info["chunks"] = (double)nchunks;
  c.info["chunk_rows"] = (double)chunk;
  c.info["buffer_rows"] = (double)buf;
  c.info["selected"] = (double)total_sel;
  c.info["sort_bits"] = bits;
}
