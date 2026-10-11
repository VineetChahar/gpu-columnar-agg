// V6 data model: star-schema join followed by an aggregate.
//   SELECT d.cat, COUNT(*), SUM(f.v), MIN(f.v), MAX(f.v), SUM(f.v*0.01)
//   FROM fact f JOIN dim d ON f.fk = d.key
//   WHERE d.p < sel
//   GROUP BY d.cat
// dim:  D rows, distinct random 64-bit keys, cat in [0,64), p uniform in [0,100)
// fact: F rows, fk dictionary-encoded (u32 codes into the fact's own dictionary, which is a
//       permutation of the dim keys), v int32.  The fraction of fact rows that join is ~sel%.
#pragma once
#include <vector>
#include "cpu/cpu.hpp"

struct JoinData {
  u64 D = 0, F = 0;
  std::vector<u64> dkey;
  std::vector<u8> dcat, dp;
  std::vector<u64> fdict;  // fact dictionary: code -> key
  std::vector<u32> fcode;
  std::vector<i32> fval;
};

void gen_join(JoinData& J, u64 D, u64 F, u64 seed);
std::vector<GroupRow> cpu_join_reference(const JoinData& J, u8 sel);              // single thread, std::unordered_map
std::vector<GroupRow> cpu_join_engine(const JoinData& J, u8 sel, int threads);    // partitioned build, parallel probe
