// V5: streaming pipeline.  Compressed columns live in host memory; chunks are copied H2D on
// NS >= 3 streams (pinned memory, cudaMemcpyAsync) while earlier chunks aggregate (V3 kernel) into
// one persistent device table; the result is extracted table-slice by slice and copied D2H on the
// same rotating streams.  Every operation is bracketed by CUDA events to rebuild a timeline
// (nsys is not installed on the Kaggle image; when it is available `make nsys` captures one too).
//
// The second half is the deliberate negative result: a query whose GPU time is pinned to the
// PCIe floor, compared with a CPU engine that scans host memory directly.
#include <thread>
#include "../src/cpu/cpu.hpp"
#include "../src/gpu/frontend.cuh"
#include "../src/kernels/variants.hpp"
#include "util.hpp"

double quick_read_bw();

namespace {

// key decoder for a global (no GROUP BY) aggregate: no key column is read at all
struct KD0 {
  __device__ __forceinline__ void code4(u64, u32 o[4]) const { o[0] = o[1] = o[2] = o[3] = 0; }
  __device__ __forceinline__ u64 decode(u32) const { return 0x5eedull; }
};

struct Ev { std::string what; int stream; cudaEvent_t a, b; u64 bytes; };

struct Pipeline {
  const HostData& H;
  bool global_agg;
  int NS;
  u64 chunk;
  std::vector<cudaStream_t> st;
  std::vector<void*> dc, dc2;
  std::vector<i32*> dv;
  std::vector<u32*> dre;
  std::vector<u16*> drv;
  GTable T;
  Results R;
  ResV hostR{};  // pinned host copy of the results
  u64 hostR_cap = 0;
  u64 slices = 0;
  u32* d_slice_n = nullptr;
  u64* d_dict = nullptr;

  Pipeline(const HostData& h, bool g, int ns, u64 ch) : H(h), global_agg(g), NS(ns), chunk(ch) {
    st.resize(NS);
    for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int w = global_agg ? 0 : (H.kind == KeyKind::COMP ? 2 : code_bytes(H.kind));
    u64 pad = chunk + TILE;
    for (int s = 0; s < NS; ++s) {
      void* p = nullptr;
      if (w) CK(cudaMalloc(&p, pad * w));
      dc.push_back(p);
      p = nullptr;
      if (!global_agg && H.kind == KeyKind::COMP) CK(cudaMalloc(&p, pad * 2));
      dc2.push_back(p);
      i32* v; CK(cudaMalloc(&v, pad * 4)); dv.push_back(v);
      u32* re; CK(cudaMalloc(&re, (chunk + 64) * 4)); dre.push_back(re);
      u16* rv; CK(cudaMalloc(&rv, (chunk + 64) * 2)); drv.push_back(rv);
    }
    if (!global_agg && H.kind != KeyKind::COMP) {
      CK(cudaMalloc(&d_dict, H.dict.size() * 8));
      CK(cudaMemcpy(d_dict, H.dict.data(), H.dict.size() * 8, cudaMemcpyHostToDevice));
    }
    u64 card = global_agg ? 1 : H.card;
    T.alloc(table_capacity(card));
    slices = std::min<u64>(T.cap / 1024, 16);
    R.alloc(T.cap);  // slice-local regions: slice j owns [j*cap/slices, (j+1)*cap/slices)
    CK(cudaMalloc(&d_slice_n, slices * 4));
    hostR_cap = T.cap;
    CK(cudaMallocHost(&hostR.key, T.cap * 8)); CK(cudaMallocHost(&hostR.cnt, T.cap * 8));
    CK(cudaMallocHost(&hostR.sum, T.cap * 8)); CK(cudaMallocHost(&hostR.dsum, T.cap * 8));
    CK(cudaMallocHost(&hostR.mn, T.cap * 4)); CK(cudaMallocHost(&hostR.mx, T.cap * 4));
    CK(cudaMallocHost(&hostR.n, slices * 4));
  }
  ~Pipeline() {
    for (int s = 0; s < NS; ++s) {
      if (dc[s]) CK(cudaFree(dc[s]));
      if (dc2[s]) CK(cudaFree(dc2[s]));
      CK(cudaFree(dv[s])); CK(cudaFree(dre[s])); CK(cudaFree(drv[s]));
      CK(cudaStreamDestroy(st[s]));
    }
    T.release(); R.release(); CK(cudaFree(d_slice_n));
    if (d_dict) CK(cudaFree(d_dict));
    for (void* p : {(void*)hostR.key, (void*)hostR.cnt, (void*)hostR.sum, (void*)hostR.dsum, (void*)hostR.mn, (void*)hostR.mx, (void*)hostR.n})
      CK(cudaFreeHost(p));
  }

  template <class KD>
  void launch_chunk(int s, KD kd, u64 start, u64 len, u32 nr, u16 lo, u16 hi) {
    Cols<KD> C{kd, dv[s], dre[s], drv[s], nr, start, len, lo, hi};
    launch_agg<KD, PSmem<LSoA<DEFAULT_SLOTS>, DEFAULT_SLOTS, true>, FE_VEC_COMPACT>(C, {T.t}, st[s]);
  }

  // overlap=false: one stream, copy everything chunk-by-chunk with no concurrency (serial baseline)
  double run(u16 lo, u16 hi, bool overlap, std::vector<Ev>* tl) {
    const int ns = overlap ? NS : 1;
    cudaEvent_t base, fin;
    CK(cudaEventCreate(&base)); CK(cudaEventCreate(&fin));
    auto mark = [&](const std::string& w, int s, u64 bytes, const std::function<void()>& f) {
      if (!tl) { f(); return; }
      Ev e{w, s, nullptr, nullptr, bytes};
      CK(cudaEventCreate(&e.a)); CK(cudaEventCreate(&e.b));
      CK(cudaEventRecord(e.a, st[s])); f(); CK(cudaEventRecord(e.b, st[s]));
      tl->push_back(e);
    };
    CK(cudaEventRecord(base, st[0]));
    mark("init", 0, 0, [&] { table_init(T, st[0]); });
    cudaEvent_t ready; CK(cudaEventCreate(&ready)); CK(cudaEventRecord(ready, st[0]));
    for (int s = 1; s < ns; ++s) CK(cudaStreamWaitEvent(st[s], ready, 0));
    const int w = global_agg ? 0 : (H.kind == KeyKind::COMP ? 2 : code_bytes(H.kind));
    u64 nchunks = (H.n + chunk - 1) / chunk;
    for (u64 i = 0; i < nchunks; ++i) {
      int s = (int)(i % ns);
      u64 a = i * chunk, len = std::min(chunk, H.n - a);
      u32 r0 = (u32)(std::upper_bound(H.run_end.begin(), H.run_end.end(), (u32)a) - H.run_end.begin());
      u32 r1 = (u32)(std::upper_bound(H.run_end.begin(), H.run_end.end(), (u32)(a + len - 1)) - H.run_end.begin()) + 1;
      u64 bytes = len * (w + 4) + (u64)(r1 - r0) * 6;
      mark("h2d", s, bytes, [&] {
        if (w) CK(cudaMemcpyAsync(dc[s], H.codes.data() + a * w, len * w, cudaMemcpyHostToDevice, st[s]));
        if (dc2[s]) CK(cudaMemcpyAsync(dc2[s], H.codes2.data() + a * 2, len * 2, cudaMemcpyHostToDevice, st[s]));
        CK(cudaMemcpyAsync(dv[s], H.vals.data() + a, len * 4, cudaMemcpyHostToDevice, st[s]));
        CK(cudaMemcpyAsync(dre[s], H.run_end.data() + r0, (r1 - r0) * 4, cudaMemcpyHostToDevice, st[s]));
        CK(cudaMemcpyAsync(drv[s], H.run_val.data() + r0, (r1 - r0) * 2, cudaMemcpyHostToDevice, st[s]));
      });
      mark("kernel", s, 0, [&] {
        if (global_agg) launch_chunk(s, KD0{}, a, len, r1 - r0, lo, hi);
        else switch (H.kind) {
          case KeyKind::U8: launch_chunk(s, KD1<u8>{(const u8*)dc[s], d_dict}, a, len, r1 - r0, lo, hi); break;
          case KeyKind::U16: launch_chunk(s, KD1<u16>{(const u16*)dc[s], d_dict}, a, len, r1 - r0, lo, hi); break;
          case KeyKind::U32: launch_chunk(s, KD1<u32>{(const u32*)dc[s], d_dict}, a, len, r1 - r0, lo, hi); break;
          default: throw Skip("composite keys not wired into the streaming bench");
        }
      });
    }
    // join all streams, then slice-wise extraction + D2H, rotating over streams
    cudaEvent_t agg_done[16];
    for (int s = 0; s < ns; ++s) { CK(cudaEventCreate(&agg_done[s])); CK(cudaEventRecord(agg_done[s], st[s])); }
    for (int s = 0; s < ns; ++s) for (int t = 0; t < ns; ++t) CK(cudaStreamWaitEvent(st[s], agg_done[t], 0));
    u64 sc = T.cap / slices;
    for (u64 j = 0; j < slices; ++j) {
      int s = (int)(j % ns);
      GT sub = T.t.slice(j * sc);  // view of slots [j*sc, (j+1)*sc)
      ResV rr = R.r;
      rr.key += j * sc; rr.cnt += j * sc; rr.sum += j * sc; rr.dsum += j * sc; rr.mn += j * sc; rr.mx += j * sc;
      rr.n = d_slice_n + j;
      mark("extract", s, 0, [&] {
        CK(cudaMemsetAsync(rr.n, 0, 4, st[s]));
        extract_slice(sub, sc, rr, st[s]);
      });
      // D2H of the slice region: an upper bound (sc entries); the count arrives with it
      mark("d2h", s, sc * 36, [&] {
        CK(cudaMemcpyAsync(hostR.key + j * sc, rr.key, sc * 8, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.cnt + j * sc, rr.cnt, sc * 8, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.sum + j * sc, rr.sum, sc * 8, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.dsum + j * sc, rr.dsum, sc * 8, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.mn + j * sc, rr.mn, sc * 4, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.mx + j * sc, rr.mx, sc * 4, cudaMemcpyDeviceToHost, st[s]));
        CK(cudaMemcpyAsync(hostR.n + j, rr.n, 4, cudaMemcpyDeviceToHost, st[s]));
      });
    }
    for (int s = 0; s < ns; ++s) { CK(cudaEventRecord(agg_done[s], st[s])); CK(cudaStreamWaitEvent(st[0], agg_done[s], 0)); }
    CK(cudaEventRecord(fin, st[0]));
    CK(cudaEventSynchronize(fin));
    float ms; CK(cudaEventElapsedTime(&ms, base, fin));
    for (int s = 0; s < ns; ++s) CK(cudaEventDestroy(agg_done[s]));
    CK(cudaEventDestroy(ready));
    timeline_base = base;
    CK(cudaEventDestroy(fin));
    return ms;
  }
  cudaEvent_t timeline_base{};

  static void extract_slice(const GT& sub, u64 sc, const ResV& rr, cudaStream_t s);

  std::vector<GroupRow> gather() const {
    std::vector<GroupRow> out;
    u64 sc = T.cap / slices;
    for (u64 j = 0; j < slices; ++j)
      for (u64 i = 0; i < hostR.n[j]; ++i) {
        u64 x = j * sc + i;
        out.push_back({hostR.key[x], {hostR.cnt[x], hostR.sum[x], hostR.mn[x], hostR.mx[x], hostR.dsum[x]}});
      }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.key < b.key; });
    return out;
  }
};

__global__ void k_extract_slice(GT T, u64 cap, ResV R) {
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  for (u64 base = gw * 32; base < cap; base += nw * 32) {
    u64 s = base + lane_id();
    bool ok = s < cap && T.K(s) != EMPTY_KEY;
    u32 b = __ballot_sync(FULL, ok);
    u32 off = 0;
    if (lane_id() == 0 && b) off = atomicAdd(R.n, __popc(b));
    off = __shfl_sync(FULL, off, 0);
    if (ok) {
      u32 i = off + __popc(b & lanemask_lt());
      R.key[i] = T.K(s); R.cnt[i] = T.C(s); R.sum[i] = (i64)T.S(s);
      R.mn[i] = T.MN(s); R.mx[i] = T.MX(s); R.dsum[i] = T.D(s);
    }
  }
}
void Pipeline::extract_slice(const GT& sub, u64 sc, const ResV& rr, cudaStream_t s) {
  k_extract_slice<<<num_sms() * 2, 256, 0, s>>>(sub, sc, rr);
  CK(cudaGetLastError());
}

bool same(const std::vector<GroupRow>& a, const std::vector<GroupRow>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const Agg &x = a[i].a, &y = b[i].a;
    if (a[i].key != b[i].key || x.cnt != y.cnt || x.sum != y.sum || x.mn != y.mn || x.mx != y.mx) return false;
    if (std::fabs(x.dsum - y.dsum) / std::max(std::fabs(y.dsum), 1.0) > 1e-9) return false;
  }
  return true;
}

// CPU global aggregate (no GROUP BY): per selected run, a tight vectorisable loop over the value
// column.  This is what a CPU engine does at memory bandwidth; SUM(v*0.01) is computed as
// 0.01*SUM(v) (exact algebra, differs from per-row double accumulation only by rounding).
GroupRow cpu_global(const HostData& H, u16 lo, u16 hi, int T) {
  std::vector<Agg> part(T, Agg{0, 0, INT_MAX, INT_MIN, 0});
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      u64 a = H.n * t / T, b = H.n * (t + 1) / T;
      size_t run = std::upper_bound(H.run_end.begin(), H.run_end.end(), (u32)a) - H.run_end.begin();
      Agg g{0, 0, INT_MAX, INT_MIN, 0};
      for (u64 r = a; r < b; ++run) {
        u64 re = std::min<u64>(H.run_end[run], b);
        u16 p = H.run_val[run];
        if (p >= lo && p < hi) {
          const i32* v = H.vals.data();
          i64 s = 0; i32 mn = INT_MAX, mx = INT_MIN;
          for (u64 i = r; i < re; ++i) { s += v[i]; mn = std::min(mn, v[i]); mx = std::max(mx, v[i]); }
          g.cnt += re - r; g.sum += s; g.mn = std::min(g.mn, mn); g.mx = std::max(g.mx, mx);
        }
        r = re;
      }
      part[t] = g;
    });
  for (auto& x : th) x.join();
  Agg g{0, 0, INT_MAX, INT_MIN, 0};
  for (auto& p : part) { g.cnt += p.cnt; g.sum += p.sum; g.mn = std::min(g.mn, p.mn); g.mx = std::max(g.mx, p.mx); }
  g.dsum = (double)g.sum * 0.01;
  return {0x5eedull, g};
}

double cpu_read_bw(const HostData& H, int T) {
  // multithreaded streaming read of the value column (the CPU's own "achievable bandwidth")
  std::vector<double> ms;
  for (int it = 0; it < 7; ++it) {
    std::vector<i64> part(T);
    double t0 = now_ms();
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t)
      th.emplace_back([&, t] {
        u64 a = H.n * t / T, b = H.n * (t + 1) / T; i64 s = 0;
        for (u64 i = a; i < b; ++i) s += H.vals[i];
        part[t] = s;
      });
    for (auto& x : th) x.join();
    ms.push_back(now_ms() - t0);
    if (part[0] == 0x7f3a5c11) printf(" ");
  }
  return H.n * 4.0 / (stats(ms).med * 1e6);
}

}  // namespace

int run_stream(const Args& a) {
  const u64 rows = (u64)a.num("rows", 200e6);
  const int T = (int)std::thread::hardware_concurrency();
  const int iters = (int)a.num("iters", 30);
  std::string tag = a.get("tag", "stream");
  double gpu_bw = quick_read_bw();
  int fails = 0;
  struct Q { const char* name; u32 card; bool global; };
  auto only = a.slist("queries", "global_agg,groupby_card16,groupby_card16M");
  const bool nsys_only = a.num("nsys_only", 0) != 0;  // trace exactly one config: pinned, 3 streams
  for (Q q : {Q{"global_agg", 16, true}, Q{"groupby_card16", 16, false}, Q{"groupby_card16M", 1u << 24, false}}) {
    if (std::find(only.begin(), only.end(), q.name) == only.end()) continue;
    HostData H;
    GenSpec g; g.rows = rows; g.card = q.card; g.seed = 99;
    gen_setup(H, g);
    gen_rows_host(H, build_zipf_alias(q.card, 0.0));
    u16 lo = 0, hi = 100;
    // bytes the GPU must receive over PCIe for this query (global aggregate never ships the key column)
    double bytes = (double)H.n * (q.global ? 4 : 4 + code_bytes(H.kind)) + H.nruns() * 6.0;
    // ---- correctness reference ----
    std::vector<GroupRow> ref;
    if (q.global) {
      ref = cpu_reference(H, lo, hi);  // card 16 grouped -> fold into one global row
      Agg t{0, 0, INT_MAX, INT_MIN, 0};
      for (auto& r : ref) { t.cnt += r.a.cnt; t.sum += r.a.sum; t.mn = std::min(t.mn, r.a.mn); t.mx = std::max(t.mx, r.a.mx); t.dsum += r.a.dsum; }
      ref = {{0x5eedull, t}};
    } else {
      ref = cpu_reference(H, lo, hi);
    }
    // ---- CPU engine on host-resident data ----
    std::vector<double> cms;
    for (int i = 0; i < (nsys_only ? 0 : 1 + std::min(iters, 10)); ++i) {
      double t0 = now_ms();
      std::vector<GroupRow> r;
      if (q.global) r = {cpu_global(H, lo, hi, T)}; else r = cpu_engine(H, lo, hi, T);
      double dt = now_ms() - t0;
      if (i) cms.push_back(dt);
      if (i == 0) {
        std::sort(r.begin(), r.end(), [](auto& x, auto& y) { return x.key < y.key; });
        if (!same(r, ref)) { printf("FAIL cpu %s\n", q.name); ++fails; }
      }
    }
    Stat cpu = stats(cms);
    double cbw = nsys_only ? 0 : cpu_read_bw(H, T);
    auto emit_row = [&](const std::string& mode, const Stat& s, bool ok, double extra_ms, int ns, u64 chunk) {
      Json j;
      j.s("exp", tag).s("query", q.name).s("mode", mode).n("rows", (double)rows).n("pcie_bytes", bytes)
          .n("median_ms", s.med).n("p25_ms", s.p25).n("p75_ms", s.p75).n("iqr_ms", s.p75 - s.p25).n("n_iter", s.n)
          .b("correct", ok).n("extra_ms", extra_ms).n("streams", ns).n("chunk_rows", (double)chunk)
          .n("cpu_ms", cpu.med).n("cpu_threads", T).n("cpu_read_gbs", cbw).n("gpu_read_gbs", gpu_bw)
          .n("eff_pcie_gbs", bytes / (s.med * 1e6));
      put_clocks(j, clocks(), clocks());
      emit(j);
      printf("  %-14s %-26s %9.2f ms (iqr %.2f) %s%s\n", q.name, mode.c_str(), s.med, s.p75 - s.p25,
             ok ? "" : "INCORRECT ", extra_ms > 0 ? ("+" + std::to_string((int)extra_ms) + "ms").c_str() : "");
      fflush(stdout);
    };
    emit_row("cpu_engine_host_resident", cpu, true, 0, 0, 0);
    printf("  CPU read bandwidth (%d threads): %.1f GB/s\n", T, cbw);

    // ---- GPU, data already resident in device memory (kernel-only lower bound) ----
    if (!nsys_only) {
      DevData D; D.upload(H);
      GTable Tb; Tb.alloc(table_capacity(q.global ? 1 : q.card));
      Results R; R.alloc(Tb.cap);
      DevRef dref; dref.upload(ref);
      cudaStream_t s0; CK(cudaStreamCreate(&s0));
      auto run = [&] {
        table_init(Tb, s0);
        if (q.global) {
          Cols<KD0> C{KD0{}, D.vals, D.run_end, D.run_val, D.nruns, 0, D.n, lo, hi};
          launch_agg<KD0, PSmem<LSoA<DEFAULT_SLOTS>, DEFAULT_SLOTS, true>, FE_VEC_COMPACT>(C, {Tb.t}, s0);
        } else {
          with_kd(D, [&](auto kd) {
            auto C = make_cols(D, kd, lo, hi);
            launch_agg<decltype(kd), PSmem<LSoA<DEFAULT_SLOTS>, DEFAULT_SLOTS, true>, FE_VEC_COMPACT>(C, {Tb.t}, s0);
          });
        }
        table_extract(Tb, R, s0);
      };
      run();
      bool ok = check_results(R, dref, s0).ok;
      if (!ok) ++fails;
      Stat s = ok ? time_gpu(run, s0, 5, iters) : Stat{};
      emit_row("gpu_resident_kernel_only", s, ok, 0, 1, 0);
      dref.release(); R.release(); Tb.release(); D.release(); CK(cudaStreamDestroy(s0));
    }
    // ---- GPU streaming from host memory ----
    u64 chunk = (u64)a.num("chunk", 16e6);
    for (int pinned = nsys_only ? 1 : 0; pinned < 2; ++pinned) {
      double reg_ms = 0;
      if (pinned) {
        double t0 = now_ms();
        if (!q.global) CK(cudaHostRegister(H.codes.data(), H.codes.size(), cudaHostRegisterDefault));
        CK(cudaHostRegister(H.vals.data(), H.vals.size() * 4, cudaHostRegisterDefault));
        CK(cudaHostRegister(H.run_end.data(), H.run_end.size() * 4, cudaHostRegisterDefault));
        CK(cudaHostRegister(H.run_val.data(), H.run_val.size() * 2, cudaHostRegisterDefault));
        reg_ms = now_ms() - t0;
      }
      for (int overlap = nsys_only ? 1 : 0; overlap < 2; ++overlap) {
        for (int ns : nsys_only ? std::vector<int>{3} : overlap ? std::vector<int>{2, 3, 4} : std::vector<int>{1}) {
          Pipeline P(H, q.global, ns, chunk);
          double first = P.run(lo, hi, overlap, nullptr);
          (void)first;
          auto got = P.gather();
          bool ok = same(got, ref);
          if (!ok) { ++fails; printf("FAIL stream %s pinned=%d overlap=%d ns=%d (%zu vs %zu groups)\n", q.name, pinned, overlap, ns, got.size(), ref.size()); }
          std::vector<double> ms;
          for (int i = 0; ok && i < std::max(5, iters / 2); ++i) ms.push_back(P.run(lo, hi, overlap, nullptr));
          std::string mode = std::string(pinned ? "pinned" : "pageable") + (overlap ? "_overlap" : "_serial");
          emit_row("gpu_stream_" + mode, stats(ms), ok, pinned ? reg_ms : 0, ns, chunk);
          if (pinned && overlap && ns == 3) {
            // event timeline of one run (relative to the pipeline's base event)
            std::vector<Ev> tl;
            P.run(lo, hi, overlap, &tl);
            for (auto& e : tl) {
              float t0, t1;
              CK(cudaEventElapsedTime(&t0, P.timeline_base, e.a));
              CK(cudaEventElapsedTime(&t1, P.timeline_base, e.b));
              Json j; j.s("exp", "timeline").s("query", q.name).s("op", e.what).n("stream", e.stream).n("t0_ms", t0)
                  .n("t1_ms", t1).n("bytes", (double)e.bytes);
              emit(j);
              CK(cudaEventDestroy(e.a)); CK(cudaEventDestroy(e.b));
            }
          }
        }
      }
      if (pinned) {
        // one-shot query: the pinning cost is paid once per query
        Json j; j.s("exp", tag).s("query", q.name).s("mode", "host_register_cost").n("median_ms", reg_ms)
            .n("rows", (double)rows).n("pcie_bytes", bytes);
        emit(j);
        printf("  %-14s cudaHostRegister of the columns: %.1f ms\n", q.name, reg_ms);
        if (!q.global) CK(cudaHostUnregister(H.codes.data()));
        CK(cudaHostUnregister(H.vals.data())); CK(cudaHostUnregister(H.run_end.data())); CK(cudaHostUnregister(H.run_val.data()));
      }
    }
    // chunk-size sensitivity (pinned, 3 streams)
    if (q.card == 16 && !q.global && !nsys_only) {
      CK(cudaHostRegister(H.codes.data(), H.codes.size(), cudaHostRegisterDefault));
      CK(cudaHostRegister(H.vals.data(), H.vals.size() * 4, cudaHostRegisterDefault));
      CK(cudaHostRegister(H.run_end.data(), H.run_end.size() * 4, cudaHostRegisterDefault));
      CK(cudaHostRegister(H.run_val.data(), H.run_val.size() * 2, cudaHostRegisterDefault));
      for (u64 ch : {1ull << 20, 4ull << 20, 64ull << 20}) {
        Pipeline P(H, false, 3, ch);
        P.run(lo, hi, true, nullptr);
        bool ok = same(P.gather(), ref);
        if (!ok) ++fails;
        std::vector<double> ms;
        for (int i = 0; ok && i < 10; ++i) ms.push_back(P.run(lo, hi, true, nullptr));
        emit_row("gpu_stream_pinned_overlap_chunk", stats(ms), ok, 0, 3, ch);
      }
      CK(cudaHostUnregister(H.codes.data())); CK(cudaHostUnregister(H.vals.data()));
      CK(cudaHostUnregister(H.run_end.data())); CK(cudaHostUnregister(H.run_val.data()));
    }
  }
  printf("stream: %d failures\n", fails);
  return fails ? 1 : 0;
}
