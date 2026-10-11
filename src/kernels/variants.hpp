// Shared interface of every aggregation variant: run one complete query
//   SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v), SUM(v*0.01) FROM t WHERE lo <= p < hi GROUP BY k
// leaving the dense result in ctx.R.  Everything a variant launches (table clear, partition passes,
// host round-trips, extraction) is inside the timed region.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "../gpu/dev.cuh"

struct RunCtx {
  const DevData* D = nullptr;
  u16 lo = 0, hi = 100;
  GTable* T = nullptr;
  Results* R = nullptr;
  cudaStream_t st = 0;
  u64 sel_est = 0;  // selected rows from RLE statistics (not timed; a planner would have it)
  // tuning knobs
  int block = 256;
  // filled in by the variant
  std::map<std::string, double> info;
};

struct Variant {
  std::string name;
  std::function<void(RunCtx&)> run;
  bool ablation = false;  // run only in ablation sweeps
};

std::vector<Variant> all_variants();
const Variant* find_variant(const std::string& name);

// per-file registration
void run_v0_naive(RunCtx&);
void run_v0_naive_soa(RunCtx&);
void release_ablation_tables();
void run_v1_vec(RunCtx&);
void run_v1_nocompact(RunCtx&);
template <int S> void run_v2_aos(RunCtx&);
template <int S> void run_v2_soa(RunCtx&);
template <int S> void run_v3_warpagg(RunCtx&);
void run_v3_global(RunCtx&);
void run_v4_radix(RunCtx&, bool l2_mode, bool force_1pass, bool nostage = false);
void run_v4_sort(RunCtx&);
void release_v4_scratch();
void release_sort_scratch();

constexpr int DEFAULT_SLOTS = 256;  // shared-memory table slots: 256 keeps 2 blocks/SM (dev sweep, see NOTEBOOK)

// CUDA-event phase breakdown recorded into RunCtx::info as ms_<phase> (from the previous mark)
struct PhaseTimer {
  cudaStream_t st;
  std::vector<std::pair<std::string, cudaEvent_t>> ev;
  explicit PhaseTimer(cudaStream_t s) : st(s) { mark("start"); }
  void mark(const std::string& name) {
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    CK(cudaEventRecord(e, st));
    ev.push_back({name, e});
  }
  void dump(std::map<std::string, double>& info) {
    CK(cudaEventSynchronize(ev.back().second));
    for (size_t i = 1; i < ev.size(); ++i) {
      float ms;
      CK(cudaEventElapsedTime(&ms, ev[i - 1].second, ev[i].second));
      info["ms_" + ev[i].first] = ms;
    }
    for (auto& e : ev) CK(cudaEventDestroy(e.second));
    ev.clear();
  }
};
