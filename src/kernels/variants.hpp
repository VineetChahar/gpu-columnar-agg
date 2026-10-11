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
void run_v1_vec(RunCtx&);
void run_v1_nocompact(RunCtx&);
template <int S> void run_v2_aos(RunCtx&);
template <int S> void run_v2_soa(RunCtx&);
template <int S> void run_v3_warpagg(RunCtx&);
void run_v3_global(RunCtx&);
void run_v4_radix(RunCtx&, bool l2_mode, bool force_1pass);
void run_v4_sort(RunCtx&);
void release_v4_scratch();
void release_sort_scratch();

constexpr int DEFAULT_SLOTS = 1024;  // shared-memory table slots (chosen by `bench tune`, see NOTEBOOK)
