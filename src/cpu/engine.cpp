// Multithreaded CPU baseline: hash aggregation on decoded keys, built -O3 -march=native.
#include <algorithm>
#include <atomic>
#include <climits>
#include <thread>
#include "cpu.hpp"

namespace {

struct CpuTable {
  u64 mask;
  int shift;
  std::vector<u64> keys;
  std::vector<Agg> aggs;
  explicit CpuTable(u64 cap)
      : mask(cap - 1), shift(64 - ilog2(cap)), keys(cap, EMPTY_KEY), aggs(cap, Agg{0, 0, INT_MAX, INT_MIN, 0.0}) {}
  // h must already be positioned so its top log2(cap) bits pick the home slot.
  inline Agg& get(u64 key, u64 h) {
    u64 s = h >> shift;
    while (true) {
      u64 k = keys[s];
      if (k == key) return aggs[s];
      if (k == EMPTY_KEY) { keys[s] = key; return aggs[s]; }
      s = (s + 1) & mask;
    }
  }
};

inline void upd(Agg& a, i32 v) {
  a.cnt++; a.sum += v;
  a.mn = std::min(a.mn, v); a.mx = std::max(a.mx, v);
  a.dsum += vscale(v);
}
inline void merge(Agg& a, const Agg& b) {
  a.cnt += b.cnt; a.sum += b.sum;
  a.mn = std::min(a.mn, b.mn); a.mx = std::max(a.mx, b.mx);
  a.dsum += b.dsum;
}

// Scan rows [a,b): fused RLE decode + predicate + dictionary decode, call f(key, v).
template <class KD, class F>
void scan(const HostData& D, const KD& code_of, u64 a, u64 b, u16 lo, u16 hi, F&& f) {
  size_t run = std::upper_bound(D.run_end.begin(), D.run_end.end(), (u32)a) - D.run_end.begin();
  u64 r = a;
  while (r < b) {
    u64 re = std::min<u64>(D.run_end[run], b);
    u16 p = D.run_val[run];
    if (p >= lo && p < hi)
      for (; r < re; ++r) f(D.decode(code_of(r)), D.vals[r]);
    r = re;
    ++run;
  }
}

template <class KD>
std::vector<GroupRow> engine_impl(const HostData& D, const KD& code_of, u16 lo, u16 hi, int T) {
  std::vector<GroupRow> out;
  if (D.card <= 65536) {
    // thread-local tables, then merge (card * T entries)
    std::vector<CpuTable> tabs;
    tabs.reserve(T);
    for (int t = 0; t < T; ++t) tabs.emplace_back(table_capacity(D.card));
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t)
      th.emplace_back([&, t] {
        CpuTable& tb = tabs[t];
        scan(D, code_of, D.n * t / T, D.n * (t + 1) / T, lo, hi,
             [&](u64 k, i32 v) { upd(tb.get(k, fmix64(k)), v); });
      });
    for (auto& x : th) x.join();
    for (int t = 1; t < T; ++t)
      for (size_t s = 0; s < tabs[t].keys.size(); ++s)
        if (tabs[t].keys[s] != EMPTY_KEY) {
          u64 k = tabs[t].keys[s];
          merge(tabs[0].get(k, fmix64(k)), tabs[t].aggs[s]);
        }
    for (size_t s = 0; s < tabs[0].keys.size(); ++s)
      if (tabs[0].keys[s] != EMPTY_KEY) out.push_back({tabs[0].keys[s], tabs[0].aggs[s]});
    return out;
  }
  // radix-partitioned: phase 1 scatter (key,val) into P partitions per thread, phase 2 aggregate.
  // P chosen so one partition's table (~8K groups, ~0.5 MB) stays in the core's 1 MB L2.
  int PB = 8;
  while (PB < 12 && ((u64)D.card >> PB) > 8192) ++PB;
  const int P = 1 << PB;
  struct Buf { std::vector<u64> k; std::vector<i32> v; };
  std::vector<std::vector<Buf>> bufs(T, std::vector<Buf>(P));
  u64 est = rle_count_selected(D, lo, hi);
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      auto& my = bufs[t];
      for (auto& b : my) { b.k.reserve(est / T / P * 9 / 8 + 64); b.v.reserve(est / T / P * 9 / 8 + 64); }
      scan(D, code_of, D.n * t / T, D.n * (t + 1) / T, lo, hi, [&](u64 k, i32 v) {
        auto& b = my[fmix64(k) >> (64 - PB)];
        b.k.push_back(k); b.v.push_back(v);
      });
    });
  for (auto& x : th) x.join();
  th.clear();
  std::atomic<int> next{0};
  std::vector<std::vector<GroupRow>> outs(T);
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      int p;
      while ((p = next.fetch_add(1)) < P) {
        u64 rows = 0;
        for (int s = 0; s < T; ++s) rows += bufs[s][p].k.size();
        if (!rows) continue;
        u64 groups = std::min<u64>((u64)D.card / P * 2 + 64, rows);
        CpuTable tb(next_pow2(2 * groups));
        for (int s = 0; s < T; ++s) {
          const auto& b = bufs[s][p];
          for (size_t i = 0; i < b.k.size(); ++i) upd(tb.get(b.k[i], fmix64(b.k[i]) << PB), b.v[i]);
        }
        for (size_t s = 0; s < tb.keys.size(); ++s)
          if (tb.keys[s] != EMPTY_KEY) outs[t].push_back({tb.keys[s], tb.aggs[s]});
      }
    });
  for (auto& x : th) x.join();
  for (auto& o : outs) out.insert(out.end(), o.begin(), o.end());
  return out;
}

}  // namespace

std::vector<GroupRow> cpu_engine(const HostData& D, u16 lo, u16 hi, int T) {
  switch (D.kind) {
    case KeyKind::U8: { const u8* c = D.codes.data(); return engine_impl(D, [c](u64 r) { return (u32)c[r]; }, lo, hi, T); }
    case KeyKind::U16: { auto c = reinterpret_cast<const u16*>(D.codes.data()); return engine_impl(D, [c](u64 r) { return (u32)c[r]; }, lo, hi, T); }
    case KeyKind::U32: { auto c = reinterpret_cast<const u32*>(D.codes.data()); return engine_impl(D, [c](u64 r) { return c[r]; }, lo, hi, T); }
    default: {
      auto c1 = reinterpret_cast<const u16*>(D.codes.data());
      auto c2 = reinterpret_cast<const u16*>(D.codes2.data());
      int sh = D.log_card2;
      return engine_impl(D, [c1, c2, sh](u64 r) { return (u32(c1[r]) << sh) | c2[r]; }, lo, hi, T);
    }
  }
}
