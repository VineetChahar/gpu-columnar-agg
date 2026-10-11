// V0: naive global-memory hash aggregate.  One row per lane, scalar loads, linear probing with a
// read-then-64-bit-atomicCAS slot claim and one atomic per aggregate.  The fused decode front end
// is shared with every other variant, so V0 differs only in what a competent first version does.
#include "../gpu/frontend.cuh"
#include "variants.hpp"

void run_v0_naive(RunCtx& c) {
  table_init(*c.T, c.st);
  with_kd(*c.D, [&](auto kd) {
    auto C = make_cols(*c.D, kd, c.lo, c.hi);
    auto L = launch_agg<decltype(kd), PGlobal<false>, FE_SCALAR>(C, {c.T->t}, c.st, c.block);
    c.info["grid"] = L.grid; c.info["occupancy"] = L.occupancy; c.info["smem"] = (double)L.smem;
  });
  table_extract(*c.T, *c.R, c.st);
}
