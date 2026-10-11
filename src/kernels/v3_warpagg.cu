// V3: warp-aggregated atomics.  __match_any_sync groups the lanes of a warp holding the same key,
// reduce_peers folds their aggregates in log2(group) shuffle rounds, and only the group leader
// touches the (shared or global) table: one atomic set per distinct key per warp.
#include "../gpu/frontend.cuh"
#include "variants.hpp"

template <int S>
void run_v3_warpagg(RunCtx& c) {
  table_init(*c.T, c.st);
  with_kd(*c.D, [&](auto kd) {
    auto C = make_cols(*c.D, kd, c.lo, c.hi);
    auto L = launch_agg<decltype(kd), PSmem<LSoA<S>, S, true>, FE_VEC_COMPACT>(C, {c.T->t}, c.st, c.block);
    c.info["grid"] = L.grid; c.info["occupancy"] = L.occupancy; c.info["smem"] = (double)L.smem;
    c.info["slots"] = S;
  });
  table_extract(*c.T, *c.R, c.st);
}
template void run_v3_warpagg<256>(RunCtx&);
template void run_v3_warpagg<512>(RunCtx&);
template void run_v3_warpagg<1024>(RunCtx&);

// ablation: warp aggregation straight into the global table (no shared-memory table)
void run_v3_global(RunCtx& c) {
  table_init(*c.T, c.st);
  with_kd(*c.D, [&](auto kd) {
    auto C = make_cols(*c.D, kd, c.lo, c.hi);
    auto L = launch_agg<decltype(kd), PGlobal<true>, FE_VEC_COMPACT>(C, {c.T->t}, c.st, c.block);
    c.info["grid"] = L.grid; c.info["occupancy"] = L.occupancy;
  });
  table_extract(*c.T, *c.R, c.st);
}
