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
} S;

}  // namespace

void release_sort_scratch() {
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
