// V6 driver: bench join D=65536,1048576,16777216 F=100M sels=1,10,50,100
#include <thread>
#include "../src/kernels/v6_join.cuh"
#include "util.hpp"

namespace {
struct JV {
  const char* name;
  int kind;  // 0 np, 1 part, 2 dict
  bool bloom;
};
const JV JVARS[] = {{"v6_np", 0, false}, {"v6_np_bloom", 0, true}, {"v6_part", 1, false},
                    {"v6_part_bloom", 1, true}, {"v6_dict", 2, false}};

void run_one(const JV& v, const DevJoin& J, JoinScratch& S, u8 sel, GTable& T, cudaStream_t st, JoinInfo& info) {
  if (v.kind == 0) run_join_np(J, S, sel, v.bloom, T, st, info);
  else if (v.kind == 1) run_join_part(J, S, sel, v.bloom, T, st, info);
  else run_join_dict(J, S, sel, T, st, info);
}
}  // namespace

// small exhaustive correctness check, called from `bench verify`
int join_verify() {
  int fails = 0, checks = 0;
  cudaStream_t st; CK(cudaStreamCreate(&st));
  for (u64 D : {1024ull, 65536ull}) for (u64 F : {1ull, 1000003ull}) {
    JoinData H; gen_join(H, D, F, 3);
    DevJoin J; J.upload(H);
    JoinScratch S; S.alloc(J);
    GTable T; T.alloc(1024);
    Results R; R.alloc(1024);
    for (int sel : {0, 1, 10, 50, 100}) {
      auto ref = cpu_join_reference(H, (u8)sel);
      auto eng = cpu_join_engine(H, (u8)sel, 4);
      ++checks;
      if (eng.size() != ref.size()) { ++fails; printf("FAIL cpu_join_engine D=%llu sel=%d\n", (unsigned long long)D, sel); }
      for (size_t i = 0; i < std::min(eng.size(), ref.size()); ++i)
        if (eng[i].key != ref[i].key || eng[i].a.cnt != ref[i].a.cnt || eng[i].a.sum != ref[i].a.sum) { ++fails; break; }
      DevRef dref; dref.upload(ref);
      for (auto& v : JVARS) {
        JoinInfo info;
        try {
          run_one(v, J, S, (u8)sel, T, st, info);
          table_extract(T, R, st);
          CheckResult r = check_results(R, dref, st);
          ++checks;
          if (!r.ok) { ++fails; printf("FAIL %s D=%llu F=%llu sel=%d: %s\n", v.name, (unsigned long long)D, (unsigned long long)F, sel, r.msg.c_str()); }
        } catch (const Skip& s) { printf("skip %s: %s\n", v.name, s.what()); }
      }
      dref.release();
    }
    R.release(); T.release(); S.release(); J.release();
  }
  printf("join verify: %d checks, %d failures\n", checks, fails);
  return fails;
}

int run_join(const Args& a) {
  auto Ds = a.list("D", "65536,1048576,16777216");
  u64 F = (u64)a.num("F", 100e6);
  auto sels = a.list("sels", "1,10,50,100");
  int iters = (int)a.num("iters", 30), warm = (int)a.num("warmup", 5);
  int threads = (int)std::thread::hardware_concurrency();
  std::string tag = a.get("tag", "join");
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int fails = 0;
  for (double Dd : Ds) {
    u64 D = (u64)Dd;
    JoinData H; gen_join(H, D, F, 11);
    DevJoin J; J.upload(H);
    JoinScratch S; S.alloc(J);
    GTable T; T.alloc(1024);
    Results R; R.alloc(1024);
    for (double sd : sels) {
      u8 sel = (u8)sd;
      double t0 = now_ms();
      auto ref = cpu_join_reference(H, sel);
      double ref_ms = now_ms() - t0;
      std::vector<double> cms;
      cpu_join_engine(H, sel, threads);
      for (int i = 0; i < std::min(iters, 10); ++i) { double t = now_ms(); cpu_join_engine(H, sel, threads); cms.push_back(now_ms() - t); }
      Stat cpu = stats(cms);
      u64 matches = 0;
      for (auto& g : ref) matches += g.a.cnt;
      DevRef dref; dref.upload(ref);
      printf(" D=%llu sel=%d%% matches=%llu ref=%.0fms cpu_engine=%.1fms\n", (unsigned long long)D, sel,
             (unsigned long long)matches, ref_ms, cpu.med);
      for (auto& v : JVARS) {
        Json j;
        j.s("exp", tag).s("variant", v.name).n("D", (double)D).n("F", (double)F).n("sel", sel).n("matches", (double)matches)
            .n("cpu_ms", cpu.med).n("cpu_threads", threads).n("ref_ms", ref_ms);
        try {
          JoinInfo info;
          run_one(v, J, S, sel, T, st, info);
          table_extract(T, R, st);
          CheckResult r = check_results(R, dref, st);
          j.b("correct", r.ok).n("probed", (double)info.probed).n("partition_bits", info.partition_bits)
              .n("max_build_partition", (double)info.max_build_partition);
          if (v.bloom && info.probed >= matches && F > matches)
            j.n("bloom_fpr", (double)(info.probed - matches) / (double)(F - matches));
          if (!r.ok) { ++fails; printf("  %-14s INCORRECT %s\n", v.name, r.msg.c_str()); }
          else {
            Clk c0 = clocks();
            Stat s = time_gpu([&] { JoinInfo x; run_one(v, J, S, sel, T, st, x); table_extract(T, R, st); }, st, warm, iters);
            Clk c1 = clocks();
            j.n("median_ms", s.med).n("p25_ms", s.p25).n("p75_ms", s.p75).n("iqr_ms", s.p75 - s.p25).n("n_iter", s.n)
                .n("speedup_vs_cpu", cpu.med / s.med);
            put_clocks(j, c0, c1);
            printf("  %-14s %9.3f ms (iqr %.3f) probed=%llu %.1fx cpu\n", v.name, s.med, s.p75 - s.p25,
                   (unsigned long long)info.probed, cpu.med / s.med);
          }
        } catch (const Skip& s) {
          j.b("correct", false).s("skipped", s.what());
          printf("  %-14s skipped: %s\n", v.name, s.what());
        }
        emit(j);
        fflush(stdout);
      }
      dref.release();
    }
    R.release(); T.release(); S.release(); J.release();
  }
  return fails ? 1 : 0;
}
