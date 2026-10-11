// Microbenchmarks that give every "% of achievable" number its denominator:
//   DRAM read / copy bandwidth, PCIe H2D/D2H (pinned + pageable), integer op throughput,
//   global atomic throughput vs working-set size (L2 vs DRAM, and a single hot address),
//   shared-memory probe and atomic cost for the AoS-64B vs SoA table layouts (bank conflicts),
//   plus an analytic Monte-Carlo bank-conflict model for the same two layouts.
#include <random>
#include "../src/gpu/procs.cuh"
#include "util.hpp"

__global__ void k_read(const int4* p, size_t n4, int* out) {
  int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x7f3a5c11) out[0] = acc;  // practically never true; keeps the loads alive
}
__global__ void k_copy(const int4* p, int4* q, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) q[i] = p[i];
}
__global__ void k_imad(int iters, int* out) {
  int a = threadIdx.x, b = blockIdx.x, c = 7, d = 3, e = 11, f = 13, g = 17, h = 19;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 16; ++j) { a = a * c + d; b = b * d + e; e = e * f + g; g = g * h + a; }
  }
  if ((a ^ b ^ e ^ g) == 0x7f3a5c11) out[0] = a;
}
__global__ void k_gatomic(unsigned long long* t, u64 mask, int iters, bool hot) {
  u64 x = splitmix64(blockIdx.x * 1024ull + threadIdx.x);
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    atomicAdd(&t[hot ? 0 : ((x >> 20) & mask)], 1ull);
  }
}
__global__ void k_gatomic_dbl(double* t, u64 mask, int iters) {
  u64 x = splitmix64(blockIdx.x * 1024ull + threadIdx.x);
  for (int i = 0; i < iters; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    atomicAdd(&t[(x >> 20) & mask], 1.0);
  }
}

// mode 0: random key probe (read)   1: random u32 atomicAdd   2: random u64 atomicAdd   3: double atomicAdd
template <class L, int S>
__global__ void k_smem(int iters, int mode, long long* cycles, u64* sink) {
  __shared__ L tab;
  for (u32 s = threadIdx.x; s < (u32)S; s += blockDim.x) {
    tab.K(s) = s; tab.CNT(s) = 0; tab.SUM(s) = 0; tab.DS(s) = 0;
  }
  __syncthreads();
  u32 x = (u32)splitmix64(blockIdx.x * 4096ull + threadIdx.x);
  u64 acc = 0;
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = x * 1664525u + 1013904223u;
    u32 s = (x >> 8) & (S - 1);
    if (mode == 0) acc += *(volatile u64*)&tab.K(s);
    else if (mode == 1) atomicAdd(&tab.CNT(s), 1u);
    else if (mode == 2) atomicAdd((unsigned long long*)&tab.SUM(s), 1ull);
    else atomicAdd(&tab.DS(s), 1.0);
  }
  long long t1 = clock64();
  __syncthreads();
  if (lane_id() == 0) atomicAdd((unsigned long long*)cycles, (unsigned long long)(t1 - t0));
  if (acc == 0x7f3a5c11) sink[0] = acc;
}

// Expected shared-memory wavefronts for a warp of 64-bit loads at random slots.  A 64-bit warp
// access is split into two half-warp phases; within a phase, distinct 8-byte words that map to the
// same bank pair serialise.  word index of key = slot * stride_words/2 (stride in 4-byte words).
static double model_wavefronts(int S, int stride_words, int trials = 200000) {
  std::mt19937_64 rng(1);
  double tot = 0;
  for (int t = 0; t < trials; ++t) {
    int wf = 0;
    for (int ph = 0; ph < 2; ++ph) {
      std::map<int, std::set<long>> bank;  // bank pair -> distinct 8B words
      for (int l = 0; l < 16; ++l) {
        long slot = rng() % S;
        long w8 = slot * stride_words / 2;
        bank[(int)(w8 % 16)].insert(w8);
      }
      size_t m = 0;
      for (auto& kv : bank) m = std::max(m, kv.second.size());
      wf += (int)m;
    }
    tot += wf;
  }
  return tot / trials;
}

void run_micro(const Args& a) {
  cudaStream_t st = 0;
  const int sms = num_sms();
  int* d_out; CK(cudaMalloc(&d_out, 64));
  auto rec = [&](const std::string& name, double val, const std::string& unit, Stat s, const Clk& pre, const Clk& post) {
    Json j; j.s("exp", "micro").s("name", name).n("value", val).s("unit", unit).n("median_ms", s.med)
        .n("p25_ms", s.p25).n("p75_ms", s.p75).n("n_iter", s.n);
    put_clocks(j, pre, post); emit(j);
    printf("micro %-28s %12.2f %s\n", name.c_str(), val, unit.c_str());
  };
  const int iters = (int)a.num("iters", 30);
  // ---- DRAM bandwidth ----
  size_t bytes = (size_t)a.num("bw_bytes", 2e9);
  int4 *p, *q; CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&q, bytes / 2)); CK(cudaMemset(p, 1, bytes));
  double best_read = 0;
  for (int bpsm : {4, 8, 16, 32}) {
    Clk c0 = clocks();
    Stat s = time_gpu([&] { k_read<<<sms * bpsm, 256>>>(p, bytes / 16, d_out); }, st, 5, iters);
    Clk c1 = clocks();
    double gbs = bytes / (s.med * 1e6);
    best_read = std::max(best_read, gbs);
    rec("dram_read_gbs_bpsm" + std::to_string(bpsm), gbs, "GB/s", s, c0, c1);
  }
  {
    Clk c0 = clocks();
    Stat s = time_gpu([&] { k_copy<<<sms * 16, 256>>>(p, q, bytes / 32); }, st, 5, iters);
    Clk c1 = clocks();
    rec("dram_copy_gbs", (bytes / 2) * 2.0 / (s.med * 1e6), "GB/s", s, c0, c1);
  }
  rec("dram_read_gbs_best", best_read, "GB/s", Stat{}, clocks(), clocks());
  // ---- PCIe ----
  {
    size_t pb = 512ull << 20;
    void* hp; CK(cudaMallocHost(&hp, pb));
    std::vector<char> pageable(pb, 1);
    memset(hp, 1, pb);
    struct { const char* n; void* h; cudaMemcpyKind k; } cases[] = {
        {"pcie_h2d_pinned_gbs", hp, cudaMemcpyHostToDevice}, {"pcie_d2h_pinned_gbs", hp, cudaMemcpyDeviceToHost},
        {"pcie_h2d_pageable_gbs", pageable.data(), cudaMemcpyHostToDevice},
        {"pcie_d2h_pageable_gbs", pageable.data(), cudaMemcpyDeviceToHost}};
    for (auto& cs : cases) {
      Clk c0 = clocks();
      Stat s = time_gpu([&] {
        if (cs.k == cudaMemcpyHostToDevice) CK(cudaMemcpyAsync(p, cs.h, pb, cs.k, st));
        else CK(cudaMemcpyAsync(cs.h, p, pb, cs.k, st));
      }, st, 3, 10);
      Clk c1 = clocks();
      rec(cs.n, pb / (s.med * 1e6), "GB/s", s, c0, c1);
    }
    CK(cudaFreeHost(hp));
  }
  // ---- integer throughput ----
  {
    int it = 4096;
    Clk c0 = clocks();
    Stat s = time_gpu([&] { k_imad<<<sms * 8, 256>>>(it, d_out); }, st, 3, iters);
    Clk c1 = clocks();
    double ops = (double)sms * 8 * 256 * it * 16 * 4 * 2;  // 4 IMAD per inner step = mul+add
    rec("int_gops", ops / (s.med * 1e6), "Gop/s", s, c0, c1);
  }
  // ---- global atomics vs working set ----
  {
    unsigned long long* t; CK(cudaMalloc(&t, 1ull << 30)); CK(cudaMemset(t, 0, 1ull << 30));
    int it = 256;
    double nops = (double)sms * 8 * 256 * it;
    for (u64 slots : {1ull << 10, 1ull << 14, 1ull << 17, 1ull << 20, 1ull << 24, 1ull << 27}) {
      Clk c0 = clocks();
      Stat s = time_gpu([&] { k_gatomic<<<sms * 8, 256>>>(t, slots - 1, it, false); }, st, 3, iters);
      Clk c1 = clocks();
      rec("gatomic_u64_random_ws" + std::to_string(slots * 8 / 1024) + "KB_gops", nops / (s.med * 1e6), "Gatomic/s", s, c0, c1);
    }
    {
      Clk c0 = clocks();
      Stat s = time_gpu([&] { k_gatomic<<<sms * 8, 256>>>(t, 0, it, true); }, st, 1, 10);
      Clk c1 = clocks();
      rec("gatomic_u64_hot_single_address_gops", nops / (s.med * 1e6), "Gatomic/s", s, c0, c1);
    }
    for (u64 slots : {1ull << 14, 1ull << 24}) {
      Clk c0 = clocks();
      Stat s = time_gpu([&] { k_gatomic_dbl<<<sms * 8, 256>>>((double*)t, slots - 1, it); }, st, 3, iters);
      Clk c1 = clocks();
      rec("gatomic_f64_random_ws" + std::to_string(slots * 8 / 1024) + "KB_gops", nops / (s.med * 1e6), "Gatomic/s", s, c0, c1);
    }
    CK(cudaFree(t));
  }
  // ---- shared-memory layouts: cycles per warp-instruction ----
  {
    long long* cyc; u64* sink; CK(cudaMalloc(&cyc, 8)); CK(cudaMalloc(&sink, 8));
    const char* modes[] = {"probe_read_u64", "atomic_u32", "atomic_u64", "atomic_f64"};
    const int it = 4096, blocks = sms * 2, threads = 256;
    for (int m = 0; m < 4; ++m) {
      for (int lay = 0; lay < 2; ++lay) {
        CK(cudaMemset(cyc, 0, 8));
        if (lay == 0) k_smem<LAoS64<512>, 512><<<blocks, threads>>>(it, m, cyc, sink);
        else k_smem<LSoA<512>, 512><<<blocks, threads>>>(it, m, cyc, sink);
        CK(cudaGetLastError());
        long long h; CK(cudaMemcpy(&h, cyc, 8, cudaMemcpyDeviceToHost));
        double per = (double)h / (blocks * threads / 32) / it;
        rec(std::string("smem_") + modes[m] + (lay == 0 ? "_aos64" : "_soa") + "_cycles_per_warp_op", per, "cycles",
            Stat{}, clocks(), clocks());
      }
    }
    CK(cudaFree(cyc)); CK(cudaFree(sink));
    rec("model_wavefronts_u64_load_aos64", model_wavefronts(512, 16), "wavefronts/warp", Stat{}, Clk{}, Clk{});
    rec("model_wavefronts_u64_load_soa", model_wavefronts(512, 2), "wavefronts/warp", Stat{}, Clk{}, Clk{});
  }
  CK(cudaFree(p)); CK(cudaFree(q)); CK(cudaFree(d_out));
}

double quick_read_bw() {
  size_t bytes = 1ull << 30;
  int4* p; int* o;
  CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 64)); CK(cudaMemset(p, 1, bytes));
  double best = 0;
  for (int bpsm : {8, 16, 32}) {
    Stat s = time_gpu([&] { k_read<<<num_sms() * bpsm, 256>>>(p, bytes / 16, o); }, 0, 3, 15);
    best = std::max(best, bytes / (s.med * 1e6));
  }
  CK(cudaFree(p)); CK(cudaFree(o));
  return best;
}
