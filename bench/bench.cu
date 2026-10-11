// bench <cmd> [key=value ...]
//   verify   [tiny=1]                 correctness of every variant on small inputs (exit 1 on any mismatch)
//   sweep    rows= cards= skews= sels= variants=main|all|v0_naive,... composite=0|1 tag=...
//   micro                             bandwidth / atomic / shared-memory microbenchmarks
//   stream, join, tpch                see stream.cu, join.cu, tpch.cu
// Results append to $RESULTS_DIR/results.jsonl (one JSON object per measurement).
#include <thread>
#include "../src/cpu/cpu.hpp"
#include "../src/kernels/variants.hpp"
#include "util.hpp"

void run_micro(const Args& a);
double quick_read_bw();
int run_stream(const Args& a);
int run_join(const Args& a);
int run_tpch(const Args& a);
int run_dump(const Args& a);
int join_verify();

static std::set<std::string> done_keys() {
  std::set<std::string> s;
  std::ifstream f(results_path());
  std::string line;
  while (std::getline(f, line)) {
    auto p = line.find("\"pkey\":\"");
    if (p == std::string::npos) continue;
    auto e = line.find('"', p + 8);
    s.insert(line.substr(p + 8, e - p - 8));
  }
  return s;
}

static std::vector<const Variant*> pick_variants(const std::string& spec) {
  static std::vector<Variant> all = all_variants();
  std::vector<const Variant*> out;
  if (spec == "main" || spec == "all") {
    for (auto& v : all) if (spec == "all" || !v.ablation) out.push_back(&v);
  } else {
    std::stringstream ss(spec); std::string t;
    while (std::getline(ss, t, ',')) {
      const Variant* v = find_variant(t);
      if (!v) { fprintf(stderr, "unknown variant %s\n", t.c_str()); exit(1); }
      out.push_back(v);
    }
  }
  return out;
}

struct Point {
  u64 rows; u32 card; double skew; bool comp;
};

static void make_data(HostData& H, const Point& p, u64 seed) {
  GenSpec g; g.rows = p.rows; g.card = p.card; g.zipf = p.skew; g.composite = p.comp; g.seed = seed;
  double t0 = now_ms();
  gen_setup(H, g);
  AliasTable A = build_zipf_alias(p.card, p.skew);
  gen_rows_host(H, A);
  printf("  generated rows=%llu card=%u skew=%.1f comp=%d in %.0f ms (runs=%zu, %.1f MB compressed)\n",
         (unsigned long long)p.rows, p.card, p.skew, p.comp, now_ms() - t0, H.nruns(), H.bytes_compressed() / 1e6);
  fflush(stdout);
}

// ------------------------------------------------------------------ verify
static int run_verify(const Args& a) {
  bool tiny = a.num("tiny", 0) != 0;
  std::vector<u64> rows = tiny ? std::vector<u64>{20011} : std::vector<u64>{1, 4099, 1000003};
  std::vector<u32> cards = tiny ? std::vector<u32>{16, 4096, 1u << 20} : std::vector<u32>{16, 256, 4096, 65536, 1u << 20, 1u << 24};
  std::vector<double> skews = tiny ? std::vector<double>{0, 1.5} : std::vector<double>{0, 1.0, 1.5};
  std::vector<int> sels = tiny ? std::vector<int>{50, 100} : std::vector<int>{0, 1, 50, 100};
  auto vars = pick_variants(a.get("variants", "all"));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int fails = 0, checks = 0;
  for (u64 n : rows) for (u32 card : cards) for (double z : skews) for (int comp = 0; comp < 2; ++comp) {
    if (n == 1 && (z != 0 || card > 256)) continue;
    HostData H; make_data(H, {n, card, z, (bool)comp}, 7);
    DevData D; D.upload(H);
    GTable T; T.alloc(table_capacity(card));
    Results R; R.alloc(std::max<u64>(card, 1));
    for (int sel : sels) {
      auto ref = cpu_reference(H, 0, (u16)sel);
      auto eng = cpu_engine(H, 0, (u16)sel, 4);
      bool eng_ok = eng.size() == ref.size();
      std::sort(eng.begin(), eng.end(), [](auto& x, auto& y) { return x.key < y.key; });
      for (size_t i = 0; eng_ok && i < ref.size(); ++i)
        eng_ok = eng[i].key == ref[i].key && eng[i].a.cnt == ref[i].a.cnt && eng[i].a.sum == ref[i].a.sum &&
                 eng[i].a.mn == ref[i].a.mn && eng[i].a.mx == ref[i].a.mx;
      ++checks;
      if (!eng_ok) { ++fails; printf("FAIL cpu_engine n=%llu card=%u z=%.1f comp=%d sel=%d\n", (unsigned long long)n, card, z, comp, sel); }
      DevRef dref; dref.upload(ref);
      for (auto* v : vars) {
        RunCtx c; c.D = &D; c.lo = 0; c.hi = (u16)sel; c.T = &T; c.R = &R; c.st = st; c.sel_est = rle_count_selected(H, 0, sel);
        try {
          v->run(c);
          CK(cudaStreamSynchronize(st));
          CK(cudaGetLastError());
          CheckResult r = check_results(R, dref, st);
          ++checks;
          if (!r.ok) {
            ++fails;
            printf("FAIL %-16s n=%llu card=%u z=%.1f comp=%d sel=%d: %s\n", v->name.c_str(), (unsigned long long)n, card, z, comp, sel, r.msg.c_str());
          }
        } catch (const Skip& s) {
          printf("skip %s: %s\n", v->name.c_str(), s.what());
        }
      }
      dref.release();
    }
    T.release(); R.release(); D.release();
    release_v4_scratch(); release_sort_scratch(); release_ablation_tables();
  }
  // checker self-test: a gate that never fails proves nothing, so corrupt one aggregate on purpose
  {
    HostData H; make_data(H, {100000, 4096, 1.0, false}, 5);
    DevData D; D.upload(H);
    GTable T; T.alloc(table_capacity(4096));
    Results R; R.alloc(4096);
    auto ref = cpu_reference(H, 0, 50);
    DevRef dref; dref.upload(ref);
    RunCtx c; c.D = &D; c.hi = 50; c.T = &T; c.R = &R; c.st = st; c.sel_est = rle_count_selected(H, 0, 50);
    run_v3_warpagg<DEFAULT_SLOTS>(c);
    bool clean = check_results(R, dref, st).ok;
    u64 one = 987654321ull;  // impossible count
    CK(cudaMemcpy(R.r.cnt + 7, &one, 8, cudaMemcpyHostToDevice));  // wrong COUNT in one group
    bool caught_cnt = !check_results(R, dref, st).ok;
    run_v3_warpagg<DEFAULT_SLOTS>(c);
    double bump; CK(cudaMemcpy(&bump, R.r.dsum + 3, 8, cudaMemcpyDeviceToHost)); bump += 1e-3;
    CK(cudaMemcpy(R.r.dsum + 3, &bump, 8, cudaMemcpyHostToDevice));  // FP sum off by 1e-3
    bool caught_dsum = !check_results(R, dref, st).ok;
    run_v3_warpagg<DEFAULT_SLOTS>(c);
    u32 fewer = (u32)ref.size() - 1;
    CK(cudaMemcpy(R.r.n, &fewer, 4, cudaMemcpyHostToDevice));  // a missing group
    bool caught_missing = !check_results(R, dref, st).ok;
    checks += 4;
    if (!(clean && caught_cnt && caught_dsum && caught_missing)) {
      ++fails;
      printf("FAIL checker self-test clean=%d cnt=%d dsum=%d missing=%d\n", clean, caught_cnt, caught_dsum, caught_missing);
    } else printf("checker self-test: injected COUNT / FP-SUM / missing-group errors all detected\n");
    dref.release(); R.release(); T.release(); D.release();
  }
  if (!tiny) fails += join_verify();
  printf("verify: %d checks, %d failures\n", checks, fails);
  Json j; j.s("exp", "verify").b("tiny", tiny).n("checks", checks).n("failures", fails); emit(j);
  return fails ? 1 : 0;
}

// ------------------------------------------------------------------ sweep
static int run_sweep(const Args& a) {
  auto rows = a.list("rows", "10M");
  auto cards = a.list("cards", "16,256,4096,65536,1048576,16777216");
  auto skews = a.list("skews", "0,0.5,1,1.5");
  auto sels = a.list("sels", "1,10,50,100");
  bool comp = a.num("composite", 0) != 0;
  int warm = (int)a.num("warmup", 5), iters = (int)a.num("iters", 30);
  int cpu_iters = (int)a.num("cpu_iters", 30);
  double cpu_budget_ms = a.num("cpu_budget_s", 20) * 1000;
  bool do_cpu = a.num("cpu", 1) != 0;
  std::string tag = a.get("tag", "sweep");
  // capacity policy: "dict" = 2 x dictionary size (default, fixed per dataset);
  // "est" = 2 x min(dictionary size, selected rows from RLE statistics) - still load <= 0.5, same for all variants
  const bool cap_est = a.get("cap", "dict") == "est";
  auto vars = pick_variants(a.get("variants", "main"));
  int threads = (int)std::thread::hardware_concurrency();
  double bw = quick_read_bw();
  printf("achievable DRAM read bandwidth (own microbenchmark): %.1f GB/s; CPU threads=%d\n", bw, threads);
  auto done = done_keys();
  cudaStream_t st; CK(cudaStreamCreate(&st));
  for (double n : rows) for (double card : cards) for (double z : skews) {
    Point p{(u64)n, (u32)card, z, comp};
    // skip whole dataset if every (sel, variant) is done
    bool all_done = true;
    auto pk = [&](const std::string& vn, int sel) {
      char b[256];
      snprintf(b, sizeof b, "%s|%s|%llu|%u|%.2f|%d|%d", tag.c_str(), vn.c_str(), (unsigned long long)p.rows, p.card, z, sel, (int)comp);
      return std::string(b);
    };
    for (double s : sels) for (auto* v : vars) if (!done.count(pk(v->name, (int)s))) all_done = false;
    if (all_done) continue;
    HostData H; make_data(H, p, 42);
    DevData D; D.upload(H);
    GTable T; Results R;
    try { T.alloc(table_capacity(p.card)); R.alloc(p.card); }
    catch (const Skip& s) { printf("skip dataset: %s\n", s.what()); continue; }
    for (double sd : sels) {
      int sel = (int)sd;
      double t0 = now_ms();
      auto ref = cpu_reference(H, 0, (u16)sel);
      double ref_ms = now_ms() - t0;
      DevRef dref; dref.upload(ref);
      // CPU engine baseline (all hardware threads), adaptive iteration count bounded by a time budget
      Stat cpu{};
      if (do_cpu) {
        std::vector<double> ms;
        double first = now_ms(); cpu_engine(H, 0, (u16)sel, threads); first = now_ms() - first;  // warmup
        int it = std::max(3, std::min(cpu_iters, (int)(cpu_budget_ms / std::max(first, 1.0))));
        for (int i = 0; i < it; ++i) { double t = now_ms(); auto r = cpu_engine(H, 0, (u16)sel, threads); ms.push_back(now_ms() - t); }
        cpu = stats(ms);
      }
      u64 sel_est = rle_count_selected(H, 0, (u16)sel);
      if (cap_est) {
        u64 want = table_capacity(std::max<u64>(1, std::min<u64>(p.card, sel_est)));
        if (want != T.cap) T.alloc(want);
      }
      printf(" sel=%d%% groups=%zu ref=%.0fms cpu_engine=%.1fms cap=%llu\n", sel, ref.size(), ref_ms, cpu.med,
             (unsigned long long)T.cap); fflush(stdout);
      for (auto* v : vars) {
        std::string key = pk(v->name, sel);
        if (done.count(key)) continue;
        RunCtx c; c.D = &D; c.lo = 0; c.hi = (u16)sel; c.T = &T; c.R = &R; c.st = st; c.sel_est = sel_est;
        Json j;
        j.s("exp", tag).s("pkey", key).s("variant", v->name).n("rows", p.rows).n("card", p.card).n("skew", z)
            .n("sel", sel).b("composite", comp).s("key_kind", key_kind_name(H.kind)).n("bytes_in", (double)H.bytes_compressed())
            .n("achievable_read_gbs", bw).n("ref_ms", ref_ms).n("cpu_ms", cpu.med).n("cpu_p25_ms", cpu.p25)
            .n("cpu_p75_ms", cpu.p75).n("cpu_iters", cpu.n).n("cpu_threads", threads).n("selected_est", (double)sel_est)
            .n("table_cap", (double)T.cap).s("cap_policy", cap_est ? "est" : "dict");
        try {
          v->run(c);  // correctness gate first: one run, diff against the reference
          CK(cudaStreamSynchronize(st));
          CheckResult r = check_results(R, dref, st);
          j.b("correct", r.ok).n("groups", (double)r.groups).n("mismatches", (double)r.mismatches)
              .n("max_rel_err_dsum", r.max_rel_dsum).n("load_factor", (double)r.groups / T.cap);
          if (!r.ok) {
            printf("  %-16s INCORRECT: %s (no timing recorded)\n", v->name.c_str(), r.msg.c_str());
          } else {
            Clk c0 = clocks();
            // first timed run tells us how long one iteration takes; very slow points keep >=30 iters but fewer warmups
            double one = now_ms(); v->run(c); CK(cudaStreamSynchronize(st)); one = now_ms() - one;
            int w = one > 1000 ? 1 : warm;
            Stat s = time_gpu([&] { v->run(c); }, st, w, iters);
            Clk c1 = clocks();
            double gbs = H.bytes_compressed() / (s.med * 1e6);
            j.n("median_ms", s.med).n("p25_ms", s.p25).n("p75_ms", s.p75).n("iqr_ms", s.p75 - s.p25).n("min_ms", s.mn)
                .n("max_ms", s.mx).n("n_iter", s.n).n("warmup", w).n("eff_gbs", gbs).n("bw_pct", 100 * gbs / bw)
                .n("speedup_vs_cpu", cpu.med / s.med).n("rows_per_s", p.rows / (s.med / 1e3));
            put_clocks(j, c0, c1);
            printf("  %-16s %9.3f ms  iqr %.3f  %6.1f%% bw  %6.1fx cpu  groups=%llu\n", v->name.c_str(), s.med,
                   s.p75 - s.p25, 100 * gbs / bw, cpu.med / s.med, (unsigned long long)r.groups);
          }
          std::ostringstream info; info << "{"; bool f = true;
          for (auto& kv : c.info) { info << (f ? "" : ",") << "\"" << kv.first << "\":" << kv.second; f = false; }
          info << "}";
          j.raw("info", info.str());
        } catch (const Skip& s) {
          j.b("correct", false).s("skipped", s.what());
          printf("  %-16s skipped: %s\n", v->name.c_str(), s.what());
        }
        emit(j);
        fflush(stdout);
        // partition / sort buffers are per variant: free them so the next variant has the memory
        release_v4_scratch(); release_sort_scratch(); release_ablation_tables();
      }
      dref.release();
    }
    T.release(); R.release(); D.release();
    release_v4_scratch(); release_sort_scratch(); release_ablation_tables();
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: bench verify|sweep|micro|stream|join|tpch [k=v...]\n"); return 1; }
  std::string cmd = argv[1];
  Args a(argc, argv, 2);
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("device: %s, %d SMs, L2 %d MB, smem/SM %zu KB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20,
         pr.sharedMemPerMultiprocessor >> 10);
  if (cmd == "verify") return run_verify(a);
  if (cmd == "sweep") return run_sweep(a);
  if (cmd == "micro") { run_micro(a); return 0; }
  if (cmd == "stream") return run_stream(a);
  if (cmd == "join") return run_join(a);
  if (cmd == "tpch") return run_tpch(a);
  if (cmd == "dump") return run_dump(a);
  fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 1;
}
