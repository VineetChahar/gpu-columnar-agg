// V6: hash join + aggregate (see src/join.hpp for the query).
//   np        : global build table, every fact row decodes fk and probes it
//   np_bloom  : + register-blocked Bloom filter (one 32-bit word, 4 bits per key) checked first
//   part      : radix-partition build and probe sides so each build partition fits a shared-memory
//               table; one block per (partition, probe chunk)
//   part_bloom: + Bloom filter applied *while partitioning the probe side*, so rows that cannot
//               match are never written out
//   dict      : join the fact *dictionary* (D entries) instead of the fact rows (F >> D), producing a
//               code -> cat translation array; the scan is then one byte gather per row.
// All variants aggregate by cat with warp-aggregated shared-memory atomics (PSmem) into a GT.
#include <cub/cub.cuh>
#include <vector>
#include "../gpu/procs.cuh"
#include "v6_join.cuh"
#include "variants.hpp"

namespace {

using CatAgg = PSmem<LSoA<256>, 256, true>;
constexpr int JS = 4096;          // shared-memory build table slots per partition
constexpr int JMAXBITS = 13;      // single-pass fan-out limit (32 KB histogram)
constexpr u32 PROBE_CHUNK = 16384;

__device__ __forceinline__ u32 bloom_bits(u64 h) {
  return (1u << ((h >> 32) & 31)) | (1u << ((h >> 37) & 31)) | (1u << ((h >> 42) & 31)) | (1u << ((h >> 47) & 31));
}

__global__ void k_build(DevJoin J, u8 sel, JHT ht, u32* bloom, u32 bmask) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < J.D; i += (u64)gridDim.x * blockDim.x) {
    if (J.dp[i] >= sel) continue;
    u64 key = J.dkey[i], h = fmix64(key);
    u32 s = (u32)(h >> ht.shift);
    while (true) {
      u64 old = atomicCAS((unsigned long long*)&ht.keys[s], (unsigned long long)EMPTY_KEY, (unsigned long long)key);
      if (old == EMPTY_KEY) { ht.cats[s] = J.dcat[i]; break; }
      s = (s + 1) & ht.mask;
    }
    if (bloom) atomicOr(&bloom[h & bmask], bloom_bits(h));
  }
}

// returns cat or -1
__device__ __forceinline__ int ht_probe(const JHT& ht, u64 key, u64 h) {
  u32 s = (u32)(h >> ht.shift);
  while (true) {
    u64 k = ht.keys[s];
    if (k == key) return ht.cats[s];
    if (k == EMPTY_KEY) return -1;
    s = (s + 1) & ht.mask;
  }
}

template <bool BLOOM>
__global__ void __launch_bounds__(256) k_probe_np(DevJoin J, JHT ht, const u32* bloom, u32 bmask, GT out,
                                                   unsigned long long* stats) {
  __shared__ __align__(16) unsigned char sm[CatAgg::SMEM];
  CatAgg agg({out}, sm);
  agg.init();
  __syncthreads();
  u32 passed = 0, matched = 0;
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  for (u64 b = gw * 128; b < J.F; b += nw * 128) {
    u64 r = b + 4 * lane_id();
    uint4 c = make_uint4(0, 0, 0, 0);
    int4 v = make_int4(0, 0, 0, 0);
    if (r < J.F) { c = *reinterpret_cast<const uint4*>(J.fcode + r); v = *reinterpret_cast<const int4*>(J.fval + r); }
    u32 cc[4] = {c.x, c.y, c.z, c.w};
    i32 vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int cat = -1;
      if (r + j < J.F) {
        u64 key = __ldg(J.fdict + cc[j]), h = fmix64(key);
        bool maybe = true;
        if (BLOOM) { u32 bb = bloom_bits(h); maybe = (__ldg(bloom + (h & bmask)) & bb) == bb; }
        if (maybe) { ++passed; cat = ht_probe(ht, key, h); }
      }
      matched += cat >= 0;
      agg.process(cat >= 0, (u64)cat, vv[j]);
    }
  }
  atomicAdd(&stats[0], (unsigned long long)passed);
  atomicAdd(&stats[1], (unsigned long long)matched);
  __syncthreads();
  agg.finish();
}

// ---------------- partitioned ----------------
struct BuildSrc {
  const u64* key; const u8* cat; const u8* p; u64 n; u8 sel;
  __device__ __forceinline__ void get(u64 i, bool& valid, u64& k, u32& pay) const {
    valid = i < n && p[i] < sel;
    k = valid ? key[i] : 0;
    pay = valid ? cat[i] : 0;
  }
};
struct ProbeSrc {
  const u32* code; const u64* dict; const i32* val; u64 n; const u32* bloom; u32 bmask;
  __device__ __forceinline__ void get(u64 i, bool& valid, u64& k, u32& pay) const {
    valid = i < n;
    k = valid ? __ldg(dict + code[i]) : 0;
    pay = valid ? (u32)val[i] : 0;
    if (valid && bloom) {
      u64 h = fmix64(k);
      u32 bb = bloom_bits(h);
      valid = (__ldg(bloom + (h & bmask)) & bb) == bb;
    }
  }
};
template <class Src>
__global__ void __launch_bounds__(256) k_jhist(Src src, int B, u32* hist) {
  extern __shared__ u32 h[];
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (u64 base = blockIdx.x * (u64)blockDim.x; base < src.n; base += (u64)gridDim.x * blockDim.x) {
    bool valid; u64 k; u32 pay;
    src.get(base + threadIdx.x, valid, k, pay);
    u32 p = valid ? (u32)(fmix64(k) >> (64 - B)) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    if (valid && lane_id() == (u32)__ffs(peers) - 1) atomicAdd(&h[p], (u32)__popc(peers));
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) hist[(u64)i * gridDim.x + blockIdx.x] = h[i];
}
template <class Src>
__global__ void __launch_bounds__(256) k_jscatter(Src src, int B, const u32* hs, u64* okey, u32* opay) {
  extern __shared__ u32 cur[];
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) cur[i] = hs[(u64)i * gridDim.x + blockIdx.x];
  __syncthreads();
  for (u64 base = blockIdx.x * (u64)blockDim.x; base < src.n; base += (u64)gridDim.x * blockDim.x) {
    bool valid; u64 k; u32 pay;
    src.get(base + threadIdx.x, valid, k, pay);
    u32 p = valid ? (u32)(fmix64(k) >> (64 - B)) : 0xffffffffu;
    u32 peers = __match_any_sync(FULL, p);
    u32 leader = __ffs(peers) - 1, pos = 0;
    if (valid && lane_id() == leader) pos = atomicAdd(&cur[p], (u32)__popc(peers));
    pos = __shfl_sync(FULL, pos, leader) + __popc(peers & lanemask_lt());
    if (valid) { okey[pos] = k; opay[pos] = pay; }
  }
}
__global__ void k_starts(const u32* hs, u32 nblk, u32 P, u32 total, u32* starts) {
  for (u32 p = blockIdx.x * blockDim.x + threadIdx.x; p <= P; p += gridDim.x * blockDim.x)
    starts[p] = p == P ? total : hs[(u64)p * nblk];
}
// item = {partition, probe a, probe b}; build range from bstarts
__global__ void __launch_bounds__(256) k_jpart(const u32* items, const u32* bstarts, const u64* bkey, const u32* bcat,
                                               const u64* pkey, const u32* pval, GT out, u32* overflow) {
  __shared__ u64 tk[JS];
  __shared__ u8 tc[JS];
  __shared__ __align__(16) unsigned char sm[CatAgg::SMEM];
  CatAgg agg({out}, sm);
  agg.init();
  const u32 p = items[3 * blockIdx.x], a = items[3 * blockIdx.x + 1], b = items[3 * blockIdx.x + 2];
  const u32 ba = bstarts[p], bb = bstarts[p + 1];
  for (u32 i = threadIdx.x; i < JS; i += blockDim.x) tk[i] = EMPTY_KEY;
  __syncthreads();
  if (bb - ba > JS * 3 / 4) { if (threadIdx.x == 0) atomicAdd(overflow, 1u); return; }  // host guarantees this never happens
  for (u32 i = ba + threadIdx.x; i < bb; i += blockDim.x) {
    u64 key = bkey[i];
    u32 s = (u32)fmix64(key) & (JS - 1);
    while (atomicCAS((unsigned long long*)&tk[s], (unsigned long long)EMPTY_KEY, (unsigned long long)key) != EMPTY_KEY)
      s = (s + 1) & (JS - 1);
    tc[s] = (u8)bcat[i];
  }
  __syncthreads();
  for (u32 base = a; base < b; base += blockDim.x) {
    u32 i = base + threadIdx.x;
    int cat = -1;
    i32 v = 0;
    if (i < b) {
      u64 key = pkey[i];
      v = (i32)pval[i];
      u32 s = (u32)fmix64(key) & (JS - 1);
      while (true) {
        u64 k = tk[s];
        if (k == key) { cat = tc[s]; break; }
        if (k == EMPTY_KEY) break;
        s = (s + 1) & (JS - 1);
      }
    }
    agg.process(cat >= 0, (u64)cat, v);
  }
  __syncthreads();
  agg.finish();
}

// ---------------- dictionary-translated ----------------
__global__ void k_dict_translate(DevJoin J, JHT ht, u8* xlat) {
  for (u64 c = blockIdx.x * (u64)blockDim.x + threadIdx.x; c < J.D; c += (u64)gridDim.x * blockDim.x) {
    u64 key = J.fdict[c];
    int cat = ht_probe(ht, key, fmix64(key));
    xlat[c] = cat < 0 ? 0xff : (u8)cat;
  }
}
__global__ void __launch_bounds__(256) k_dict_scan(DevJoin J, const u8* xlat, GT out) {
  __shared__ __align__(16) unsigned char sm[CatAgg::SMEM];
  CatAgg agg({out}, sm);
  agg.init();
  __syncthreads();
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  for (u64 b = gw * 128; b < J.F; b += nw * 128) {
    u64 r = b + 4 * lane_id();
    uint4 c = make_uint4(0, 0, 0, 0);
    int4 v = make_int4(0, 0, 0, 0);
    if (r < J.F) { c = *reinterpret_cast<const uint4*>(J.fcode + r); v = *reinterpret_cast<const int4*>(J.fval + r); }
    u32 cc[4] = {c.x, c.y, c.z, c.w};
    i32 vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      u32 cat = r + j < J.F ? __ldg(xlat + cc[j]) : 0xffu;
      agg.process(cat != 0xffu, (u64)cat, vv[j]);
    }
  }
  __syncthreads();
  agg.finish();
}

int ceil_log2(double x) { int b = 0; while ((double)(1ull << b) < x) ++b; return b; }

}  // namespace

// ------------------------------------------------------------------ host side
void DevJoin::upload(const JoinData& H) {
  D = H.D; F = H.F;
  u64 fpad = (F + 4096) / 4 * 4;
  CK(cudaMalloc(&dkey, D * 8)); CK(cudaMemcpy(dkey, H.dkey.data(), D * 8, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dcat, D)); CK(cudaMemcpy(dcat, H.dcat.data(), D, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dp, D)); CK(cudaMemcpy(dp, H.dp.data(), D, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&fdict, D * 8)); CK(cudaMemcpy(fdict, H.fdict.data(), D * 8, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&fcode, fpad * 4)); CK(cudaMemset(fcode, 0, fpad * 4));
  CK(cudaMemcpy(fcode, H.fcode.data(), F * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&fval, fpad * 4)); CK(cudaMemset(fval, 0, fpad * 4));
  CK(cudaMemcpy(fval, H.fval.data(), F * 4, cudaMemcpyHostToDevice));
}
void DevJoin::release() {
  for (void* p : {(void*)dkey, (void*)dcat, (void*)dp, (void*)fdict, (void*)fcode, (void*)fval}) if (p) CK(cudaFree(p));
  *this = DevJoin{};
}

void JoinScratch::alloc(const DevJoin& J) {
  u64 cap = next_pow2(J.D * 2);
  ht.keys = (u64*)dmalloc_or_skip(cap * 8, "join table");
  ht.cats = (u8*)dmalloc_or_skip(cap, "join table");
  ht.shift = 64 - ilog2(cap); ht.mask = (u32)(cap - 1); ht_cap = cap;
  bloom_words = next_pow2(std::max<u64>(J.D / 2, 1024));  // 16 bits per build key
  bloom = (u32*)dmalloc_or_skip(bloom_words * 4, "bloom");
  stats = (unsigned long long*)dmalloc_or_skip(64, "stats");
  xlat = (u8*)dmalloc_or_skip(J.D, "xlat");
}
void JoinScratch::release() {
  for (void* p : {(void*)ht.keys, (void*)ht.cats, (void*)bloom, (void*)stats, (void*)xlat, (void*)bkey, (void*)bcat,
                  (void*)pkey, (void*)pval, (void*)hist, (void*)hscan, (void*)starts, (void*)items, tmp})
    if (p) CK(cudaFree(p));
  *this = JoinScratch{};
}

static void build(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, cudaStream_t st) {
  CK(cudaMemsetAsync(S.ht.keys, 0xff, S.ht_cap * 8, st));
  if (bloom) CK(cudaMemsetAsync(S.bloom, 0, S.bloom_words * 4, st));
  k_build<<<num_sms() * 8, 256, 0, st>>>(J, sel, S.ht, bloom ? S.bloom : nullptr, (u32)(S.bloom_words - 1));
  CK(cudaGetLastError());
}

void run_join_np(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, GTable& out, cudaStream_t st, JoinInfo& info) {
  table_init(out, st);
  build(J, S, sel, bloom, st);
  CK(cudaMemsetAsync(S.stats, 0, 16, st));
  if (bloom) k_probe_np<true><<<num_sms() * 4, 256, 0, st>>>(J, S.ht, S.bloom, (u32)(S.bloom_words - 1), out.t, S.stats);
  else k_probe_np<false><<<num_sms() * 4, 256, 0, st>>>(J, S.ht, nullptr, 0, out.t, S.stats);
  CK(cudaGetLastError());
  unsigned long long h[2];
  CK(cudaMemcpyAsync(h, S.stats, 16, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  info.probed = h[0]; info.matched = h[1];
}

void run_join_dict(const DevJoin& J, JoinScratch& S, u8 sel, GTable& out, cudaStream_t st, JoinInfo& info) {
  table_init(out, st);
  build(J, S, sel, false, st);
  k_dict_translate<<<num_sms() * 8, 256, 0, st>>>(J, S.ht, S.xlat);
  k_dict_scan<<<num_sms() * 4, 256, 0, st>>>(J, S.xlat, out.t);
  CK(cudaGetLastError());
  info.probed = J.D;
}

template <class T>
static void ensure(T*& p, size_t& cap, size_t n, const char* what) {
  if (n <= cap) return;
  if (p) CK(cudaFree(p));
  p = nullptr; cap = 0;
  p = (T*)dmalloc_or_skip(n * sizeof(T), what);
  cap = n;
}

template <class Src>
static u32 partition(const Src& src, int B, u64* okey, u32* opay, JoinScratch& S, cudaStream_t st, std::vector<u32>& starts) {
  const u32 P = 1u << B;
  const size_t smem = (size_t)P * 4;
  int bps;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_jhist<Src>, 256, smem));
  bps = std::max(1, std::min(bps, 4));
  u32 nblk = (u32)(bps * num_sms());
  ensure(S.hist, S.hist_cap, (size_t)P * nblk, "jhist");
  ensure(S.hscan, S.hscan_cap, (size_t)P * nblk, "jhist");
  ensure(S.starts, S.starts_cap, (size_t)P + 1, "jstarts");
  k_jhist<Src><<<nblk, 256, smem, st>>>(src, B, S.hist);
  size_t tb = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, S.hist, S.hscan, (int)(P * nblk), st));
  if (tb > S.tmp_cap) { if (S.tmp) CK(cudaFree(S.tmp)); S.tmp = dmalloc_or_skip(tb, "jtmp"); S.tmp_cap = tb; }
  CK(cub::DeviceScan::ExclusiveSum(S.tmp, tb, S.hist, S.hscan, (int)(P * nblk), st));
  u32 last[2];
  CK(cudaMemcpyAsync(&last[0], S.hist + (size_t)P * nblk - 1, 4, cudaMemcpyDeviceToHost, st));
  CK(cudaMemcpyAsync(&last[1], S.hscan + (size_t)P * nblk - 1, 4, cudaMemcpyDeviceToHost, st));
  k_jscatter<Src><<<nblk, 256, smem, st>>>(src, B, S.hscan, okey, opay);
  CK(cudaStreamSynchronize(st));
  u32 total = last[0] + last[1];
  k_starts<<<64, 256, 0, st>>>(S.hscan, nblk, P, total, S.starts);
  starts.resize(P + 1);
  CK(cudaMemcpyAsync(starts.data(), S.starts, (P + 1) * 4, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  return total;
}

void run_join_part(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, GTable& out, cudaStream_t st, JoinInfo& info) {
  for (int k = 0; k < 2; ++k)
    CK(cudaFuncSetAttribute(k == 0 ? (const void*)k_jhist<ProbeSrc> : (const void*)k_jscatter<ProbeSrc>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024));
  PhaseTimer pt(st);
  table_init(out, st);
  // plan: build keys expected after the filter (from the dim statistics: sel% of D)
  double nb = std::max(1.0, J.D * sel / 100.0);
  int B = std::max(1, ceil_log2(nb / (JS / 2)));
  if (B > JMAXBITS) throw Skip("join fan-out beyond single pass");
  info.partition_bits = B;
  ensure(S.bkey, S.bkey_cap, J.D, "join build partitions");
  ensure(S.bcat, S.bcat_cap, J.D, "join build partitions");
  ensure(S.pkey, S.pkey_cap, J.F, "join probe partitions");
  ensure(S.pval, S.pval_cap, J.F, "join probe partitions");
  if (bloom) CK(cudaMemsetAsync(S.bloom, 0, S.bloom_words * 4, st));
  std::vector<u32> bst, pst;
  for (int k = 0; k < 2; ++k)
    CK(cudaFuncSetAttribute(k == 0 ? (const void*)k_jhist<BuildSrc> : (const void*)k_jscatter<BuildSrc>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024));
  u32 nbuild = partition(BuildSrc{J.dkey, J.dcat, J.dp, J.D, sel}, B, S.bkey, S.bcat, S, st, bst);
  pt.mark("build_partition");
  if (bloom) {
    // Bloom filter over the partitioned build keys (contiguous, coalesced)
    build_bloom(S.bkey, nbuild, S.bloom, (u32)(S.bloom_words - 1), st);
  }
  u32 nprobe = partition(ProbeSrc{J.fcode, J.fdict, J.fval, J.F, bloom ? S.bloom : nullptr, (u32)(S.bloom_words - 1)},
                         B, S.pkey, S.pval, S, st, pst);
  pt.mark("probe_partition");
  u32 maxb = 0;
  for (size_t p = 0; p + 1 < bst.size(); ++p) maxb = std::max(maxb, bst[p + 1] - bst[p]);
  if (maxb > JS * 3 / 4) throw Skip("build partition overflow");
  std::vector<u32> items;
  for (size_t p = 0; p + 1 < pst.size(); ++p)
    for (u32 a = pst[p]; a < pst[p + 1]; a += PROBE_CHUNK) {
      items.push_back((u32)p); items.push_back(a); items.push_back(std::min(a + PROBE_CHUNK, pst[p + 1]));
    }
  u32 nitems = (u32)(items.size() / 3);
  ensure(S.items, S.items_cap, items.size() + 3, "jitems");
  ensure(S.bst, S.bst_cap, bst.size(), "jbst");
  CK(cudaMemcpyAsync(S.bst, bst.data(), bst.size() * 4, cudaMemcpyHostToDevice, st));
  CK(cudaMemsetAsync(S.stats, 0, 16, st));
  if (nitems) {
    CK(cudaMemcpyAsync(S.items, items.data(), items.size() * 4, cudaMemcpyHostToDevice, st));
    k_jpart<<<nitems, 256, 0, st>>>(S.items, S.bst, S.bkey, S.bcat, S.pkey, S.pval, out.t, (u32*)S.stats);
    CK(cudaGetLastError());
  }
  pt.mark("join_aggregate");
  u32 ovf;
  CK(cudaMemcpyAsync(&ovf, S.stats, 4, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  if (ovf) throw Skip("build partition overflow (device)");
  info.probed = nprobe;
  info.build_rows = nbuild;
  info.max_build_partition = maxb;
  pt.dump(info.phase_ms);
}

__global__ void k_bloom_build(const u64* keys, u32 n, u32* bloom, u32 bmask) {
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    u64 h = fmix64(keys[i]);
    atomicOr(&bloom[h & bmask], bloom_bits(h));
  }
}
void build_bloom(const u64* keys, u32 n, u32* bloom, u32 bmask, cudaStream_t st) {
  if (n) k_bloom_build<<<num_sms() * 8, 256, 0, st>>>(keys, n, bloom, bmask);
  CK(cudaGetLastError());
}
