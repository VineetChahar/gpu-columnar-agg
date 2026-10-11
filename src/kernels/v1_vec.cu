// V1: coalesced 16-byte vector loads (4 rows per lane, 128 rows per warp step), predicate turned
// into ballot masks, and stream compaction via __ballot_sync + __popc prefix into a per-warp ring
// buffer so the probe/atomic phase always runs with a full warp.  Table: same global table as V0.
#include "../gpu/frontend.cuh"
#include "variants.hpp"

template <int MODE>
static void run_v1_impl(RunCtx& c) {
  table_init(*c.T, c.st);
  with_kd(*c.D, [&](auto kd) {
    auto C = make_cols(*c.D, kd, c.lo, c.hi);
    auto L = launch_agg<decltype(kd), PGlobal<false>, MODE>(C, {c.T->t}, c.st, c.block);
    c.info["grid"] = L.grid; c.info["occupancy"] = L.occupancy; c.info["smem"] = (double)L.smem;
  });
  table_extract(*c.T, *c.R, c.st);
}
void run_v1_vec(RunCtx& c) { run_v1_impl<FE_VEC_COMPACT>(c); }
void run_v1_nocompact(RunCtx& c) { run_v1_impl<FE_VEC>(c); }
