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
#include "../gpu/partition.cuh"
#include "../gpu/procs.cuh"
#include "v6_join.cuh"
#include "variants.hpp"

namespace {

using CatAgg = PSmem<LSoA<256>, 256, true>;
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
// Probe side is radix-partitioned (write-combined staging, Bloom filter applied while partitioning)
// on the top B bits of fmix64(key).  The build table's home slot is also the top bits of the same
// hash, so partition p only ever touches the contiguous slice [p*cap/2^B, (p+1)*cap/2^B) of the
// table; B is chosen so that slice is <= 1 MB and partitions are processed in order -> L2-resident.
__device__ __forceinline__ bool probe_item(const DevJoin& J, u64 i, u64 end, const u32* bloom, u32 bmask, u64& key) {
  if (i >= end) return false;
  key = __ldg(J.fdict + J.fcode[i]);
  if (!bloom) return true;
  u64 h = fmix64(key);
  u32 bb = bloom_bits(h);
  return (__ldg(bloom + (h & bmask)) & bb) == bb;
}
__global__ void __launch_bounds__(STAGE_THREADS) k_jp_hist(DevJoin J, u64 chunk, int B, const u32* bloom, u32 bmask, u32* hist) {
  __shared__ u32 h[STAGE_PMAX];
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) h[i] = 0;
  __syncthreads();
  const u64 b0 = blockIdx.x * chunk, b1 = min(b0 + chunk, J.F);
  for (u64 base = b0; base < b1; base += STAGE_CAP) {
#pragma unroll
    for (int j = 0; j < STAGE_ITEMS; ++j) {
      u64 key = 0;
      bool ok = probe_item(J, base + j * blockDim.x + threadIdx.x, b1, bloom, bmask, key);
      u32 p = ok ? (u32)(fmix64(key) >> (64 - B)) : 0xffffffffu;
      u32 peers = __match_any_sync(FULL, p);
      if (ok && lane_id() == (u32)__ffs(peers) - 1) atomicAdd(&h[p], (u32)__popc(peers));
    }
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) hist[(u64)i * gridDim.x + blockIdx.x] = h[i];
}
__global__ void __launch_bounds__(STAGE_THREADS) k_jp_scatter(DevJoin J, u64 chunk, int B, const u32* bloom, u32 bmask,
                                                              const u32* hs, u64* okey, i32* oval) {
  __shared__ StageSmem St;
  for (u32 i = threadIdx.x; i < (1u << B); i += blockDim.x) St.cur[i] = hs[(u64)i * gridDim.x + blockIdx.x];
  __syncthreads();
  const u64 b0 = blockIdx.x * chunk, b1 = min(b0 + chunk, J.F);
  for (u64 base = b0; base < b1; base += STAGE_CAP) {
    bool ok[4]; u32 p[4]; u64 key[4]; i32 v[4];
#pragma unroll
    for (int j = 0; j < STAGE_ITEMS; ++j) {
      u64 i = base + j * blockDim.x + threadIdx.x;
      key[j] = 0;
      ok[j] = probe_item(J, i, b1, bloom, bmask, key[j]);
      v[j] = ok[j] ? J.fval[i] : 0;
      p[j] = ok[j] ? (u32)(fmix64(key[j]) >> (64 - B)) : 0;
    }
    stage_round(St, 1u << B, ok, p, key, v, okey, oval);
  }
}
__global__ void k_jstarts(const u32* hs, u32 nblk, u32 P, u32 total, u32* starts) {
  for (u32 p = blockIdx.x * blockDim.x + threadIdx.x; p <= P; p += gridDim.x * blockDim.x)
    starts[p] = p == P ? total : hs[(u64)p * nblk];
}
// one block per (partition, chunk) item, items in partition order
__global__ void __launch_bounds__(256) k_jpart_probe(const u32* items, JHT ht, const u64* pkey, const i32* pval, GT out) {
  __shared__ __align__(16) unsigned char sm[CatAgg::SMEM];
  CatAgg agg({out}, sm);
  agg.init();
  __syncthreads();
  const u32 a = items[2 * blockIdx.x], b = items[2 * blockIdx.x + 1];
  for (u32 base = a; base < b; base += blockDim.x) {
    u32 i = base + threadIdx.x;
    int cat = -1;
    i32 v = 0;
    if (i < b) {
      u64 key = pkey[i];
      v = pval[i];
      cat = ht_probe(ht, key, fmix64(key));
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

void run_join_part(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, GTable& out, cudaStream_t st, JoinInfo& info) {
  PhaseTimer pt(st);
  table_init(out, st);
  build(J, S, sel, bloom, st);
  pt.mark("build");
  // B: table slice per partition <= 1 MB (keys 8 B + cat 1 B per slot), fan-out <= 256 (one staged pass)
  int B = std::max(1, std::min(8, ceil_log2((double)S.ht_cap * 9.0 / (1 << 20))));
  const u32 P = 1u << B;
  info.partition_bits = B;
  ensure(S.pkey, S.pkey_cap, J.F, "join probe partitions");
  ensure(S.pval, S.pval_cap, J.F, "join probe partitions");
  int bps;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_jp_scatter, STAGE_THREADS, 0));
  const u32 nblk = (u32)(std::max(1, bps) * num_sms());
  const u64 chunk = (J.F + nblk - 1) / nblk;
  const u64 chunk_r = (chunk + STAGE_CAP - 1) / STAGE_CAP * STAGE_CAP;
  ensure(S.hist, S.hist_cap, (size_t)P * nblk, "jhist");
  ensure(S.hscan, S.hscan_cap, (size_t)P * nblk, "jhist");
  ensure(S.starts, S.starts_cap, (size_t)P + 1, "jstarts");
  const u32* bl = bloom ? S.bloom : nullptr;
  const u32 bmask = (u32)(S.bloom_words - 1);
  k_jp_hist<<<nblk, STAGE_THREADS, 0, st>>>(J, chunk_r, B, bl, bmask, S.hist);
  size_t tb = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, S.hist, S.hscan, (int)(P * nblk), st));
  if (tb > S.tmp_cap) { if (S.tmp) CK(cudaFree(S.tmp)); S.tmp = dmalloc_or_skip(tb, "jtmp"); S.tmp_cap = tb; }
  CK(cub::DeviceScan::ExclusiveSum(S.tmp, tb, S.hist, S.hscan, (int)(P * nblk), st));
  u32 last[2];
  CK(cudaMemcpyAsync(&last[0], S.hist + (size_t)P * nblk - 1, 4, cudaMemcpyDeviceToHost, st));
  CK(cudaMemcpyAsync(&last[1], S.hscan + (size_t)P * nblk - 1, 4, cudaMemcpyDeviceToHost, st));
  k_jp_scatter<<<nblk, STAGE_THREADS, 0, st>>>(J, chunk_r, B, bl, bmask, S.hscan, S.pkey, S.pval);
  CK(cudaStreamSynchronize(st));
  const u32 total = last[0] + last[1];
  k_jstarts<<<16, 256, 0, st>>>(S.hscan, nblk, P, total, S.starts);
  std::vector<u32> pst(P + 1);
  CK(cudaMemcpyAsync(pst.data(), S.starts, (P + 1) * 4, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  pt.mark("probe_partition");
  std::vector<u32> items;
  for (u32 p = 0; p < P; ++p)
    for (u32 a = pst[p]; a < pst[p + 1]; a += PROBE_CHUNK) { items.push_back(a); items.push_back(std::min(a + PROBE_CHUNK, pst[p + 1])); }
  const u32 nitems = (u32)(items.size() / 2);
  ensure(S.items, S.items_cap, items.size() + 2, "jitems");
  if (nitems) {
    CK(cudaMemcpyAsync(S.items, items.data(), items.size() * 4, cudaMemcpyHostToDevice, st));
    k_jpart_probe<<<nitems, 256, 0, st>>>(S.items, S.ht, S.pkey, S.pval, out.t);
    CK(cudaGetLastError());
  }
  pt.mark("probe_join_aggregate");
  info.probed = total;
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
