#include <algorithm>
#include <climits>
#include "cpu.hpp"

std::vector<GroupRow> cpu_reference(const HostData& D, u16 lo, u16 hi) {
  std::vector<u64> cnt(D.card, 0);
  std::vector<i64> sum(D.card, 0);
  std::vector<i32> mn(D.card, INT_MAX), mx(D.card, INT_MIN);
  std::vector<double> dsum(D.card, 0.0);
  size_t run = 0;
  for (u64 r = 0; r < D.n; ++r) {
    while (D.run_end[run] <= r) ++run;
    u16 p = D.run_val[run];
    if (p < lo || p >= hi) continue;
    u32 c = D.code(r);
    i32 v = D.vals[r];
    cnt[c]++;
    sum[c] += v;
    mn[c] = std::min(mn[c], v);
    mx[c] = std::max(mx[c], v);
    dsum[c] += vscale(v);
  }
  std::vector<GroupRow> out;
  for (u32 c = 0; c < D.card; ++c)
    if (cnt[c]) out.push_back({D.decode(c), {cnt[c], sum[c], mn[c], mx[c], dsum[c]}});
  std::sort(out.begin(), out.end(), [](const GroupRow& a, const GroupRow& b) { return a.key < b.key; });
  return out;
}
