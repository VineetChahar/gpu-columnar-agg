#include "variants.hpp"

std::vector<Variant> all_variants() {
  return {
      {"v0_naive", run_v0_naive},
      {"v0_naive_soa", run_v0_naive_soa, true},
      {"v1_vec", run_v1_vec},
      {"v1_nocompact", run_v1_nocompact, true},
      {"v2_smem_aos512", run_v2_aos<512>},
      {"v2_smem_soa512", run_v2_soa<512>},
      {"v2_smem_soa", run_v2_soa<DEFAULT_SLOTS>},
      {"v2_smem_soa1024", run_v2_soa<1024>, true},
      {"v3_warpagg", run_v3_warpagg<DEFAULT_SLOTS>},
      {"v3_warpagg1024", run_v3_warpagg<1024>, true},
      {"v3_warpagg512", run_v3_warpagg<512>, true},
      {"v3_global", run_v3_global, true},
      {"v3_latemat", run_v3_latemat, true},
      {"v4_radix_smem", [](RunCtx& c) { run_v4_radix(c, false, false); }},
      {"v4_radix_l2", [](RunCtx& c) { run_v4_radix(c, true, false); }},
      {"v4_radix_1pass", [](RunCtx& c) { run_v4_radix(c, false, true); }, true},
      {"v4_radix_nostage", [](RunCtx& c) { run_v4_radix(c, false, false, true); }, true},
      {"v4_sort", run_v4_sort},
  };
}

const Variant* find_variant(const std::string& name) {
  static std::vector<Variant> v = all_variants();
  for (auto& x : v)
    if (x.name == name) return &x;
  return nullptr;
}
