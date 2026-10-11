// V2: block-private shared-memory hash tables (load factor capped at 0.5, 16-probe limit), overflow
// to the global table, block-level merge at the end.  Two slot layouts:
//   aos : 64-byte AoS buckets (a global-memory habit) -> keys land on 2 banks -> bank conflicts
//   soa : structure-of-arrays -> keys spread over all banks
#include "../gpu/frontend.cuh"
#include "variants.hpp"

template <class L, int S>
static void run_v2_impl(RunCtx& c) {
  table_init(*c.T, c.st);
  with_kd(*c.D, [&](auto kd) {
    auto C = make_cols(*c.D, kd, c.lo, c.hi);
    auto Li = launch_agg<decltype(kd), PSmem<L, S, false>, FE_VEC_COMPACT>(C, {c.T->t}, c.st, c.block);
    c.info["grid"] = Li.grid; c.info["occupancy"] = Li.occupancy; c.info["smem"] = (double)Li.smem;
    c.info["slots"] = S;
  });
  table_extract(*c.T, *c.R, c.st);
}
template <int S> void run_v2_aos(RunCtx& c) { run_v2_impl<LAoS64<S>, S>(c); }
template <int S> void run_v2_soa(RunCtx& c) { run_v2_impl<LSoA<S>, S>(c); }
template void run_v2_aos<512>(RunCtx&);
template void run_v2_soa<256>(RunCtx&);
template void run_v2_soa<512>(RunCtx&);
template void run_v2_soa<1024>(RunCtx&);
