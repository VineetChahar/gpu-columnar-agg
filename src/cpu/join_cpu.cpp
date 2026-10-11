#include <algorithm>
#include <climits>
#include <thread>
#include <unordered_map>
#include "../join.hpp"

void gen_join(JoinData& J, u64 D, u64 F, u64 seed) {
  J.D = D; J.F = F;
  J.dkey.resize(D); J.dcat.resize(D); J.dp.resize(D);
  for (u64 i = 0; i < D; ++i) {
    J.dkey[i] = splitmix64(i ^ (seed * 0x2545F4914F6CDD1DULL));
    if (J.dkey[i] == EMPTY_KEY) J.dkey[i] = 0;
    u64 x = splitmix64(row_seed(seed, 5, i));
    J.dcat[i] = (u8)(x % 64);
    J.dp[i] = (u8)((x >> 32) % 100);
  }
  // fact dictionary = the dim keys under an odd-multiplier permutation (D is a power of two)
  J.fdict.resize(D);
  for (u64 c = 0; c < D; ++c) J.fdict[c] = J.dkey[(c * 0x9E3779B1ull) & (D - 1)];
  J.fcode.resize(F); J.fval.resize(F);
  unsigned T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      for (u64 r = F * t / T; r < F * (t + 1) / T; ++r) {
        J.fcode[r] = (u32)(splitmix64(row_seed(seed, 6, r)) & (D - 1));
        J.fval[r] = sample_val(row_seed(seed, 7, r));
      }
    });
  for (auto& x : th) x.join();
}

static void upd(Agg& a, i32 v) {
  a.cnt++; a.sum += v; a.mn = std::min(a.mn, v); a.mx = std::max(a.mx, v); a.dsum += vscale(v);
}
static std::vector<GroupRow> to_rows(const Agg* a) {
  std::vector<GroupRow> out;
  for (int c = 0; c < 64; ++c) if (a[c].cnt) out.push_back({(u64)c, a[c]});
  return out;
}

std::vector<GroupRow> cpu_join_reference(const JoinData& J, u8 sel) {
  std::unordered_map<u64, u8> ht;
  ht.reserve(J.D);
  for (u64 i = 0; i < J.D; ++i) if (J.dp[i] < sel) ht.emplace(J.dkey[i], J.dcat[i]);
  Agg a[64];
  for (auto& x : a) x = Agg{0, 0, INT_MAX, INT_MIN, 0.0};
  for (u64 r = 0; r < J.F; ++r) {
    auto it = ht.find(J.fdict[J.fcode[r]]);
    if (it != ht.end()) upd(a[it->second], J.fval[r]);
  }
  return to_rows(a);
}

// T disjoint open-addressing sub-tables (partition = top bits of the hash), built in parallel;
// probe in parallel over fact chunks with thread-local aggregates.
std::vector<GroupRow> cpu_join_engine(const JoinData& J, u8 sel, int T) {
  int PB = 0; while ((1 << PB) < T) ++PB;
  const int P = 1 << PB;
  u64 per = std::max<u64>(16, next_pow2(J.D * 2 / P + 64));
  std::vector<std::vector<u64>> keys(P, std::vector<u64>(per, EMPTY_KEY));
  std::vector<std::vector<u8>> cats(P, std::vector<u8>(per));
  const int sh = 64 - ilog2(per);
  auto home = [&](u64 h) { return PB ? (h << PB) >> sh : h >> sh; };
  std::vector<std::thread> th;
  for (int p = 0; p < P; ++p)
    th.emplace_back([&, p] {
      auto& k = keys[p]; auto& c = cats[p];
      for (u64 i = 0; i < J.D; ++i) {
        if (J.dp[i] >= sel) continue;
        u64 h = fmix64(J.dkey[i]);
        if (PB && (int)(h >> (64 - PB)) != p) continue;
        u64 s = home(h);
        while (k[s] != EMPTY_KEY) s = (s + 1) & (per - 1);
        k[s] = J.dkey[i]; c[s] = J.dcat[i];
      }
    });
  for (auto& x : th) x.join();
  th.clear();
  std::vector<std::vector<Agg>> part(T, std::vector<Agg>(64, Agg{0, 0, INT_MAX, INT_MIN, 0.0}));
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      auto& a = part[t];
      for (u64 r = J.F * t / T; r < J.F * (t + 1) / T; ++r) {
        u64 key = J.fdict[J.fcode[r]];
        u64 h = fmix64(key);
        int p = PB ? (int)(h >> (64 - PB)) : 0;
        const auto& k = keys[p];
        u64 s = home(h);
        while (true) {
          u64 kk = k[s];
          if (kk == key) { upd(a[cats[p][s]], J.fval[r]); break; }
          if (kk == EMPTY_KEY) break;
          s = (s + 1) & (per - 1);
        }
      }
    });
  for (auto& x : th) x.join();
  Agg tot[64];
  for (int c = 0; c < 64; ++c) {
    tot[c] = Agg{0, 0, INT_MAX, INT_MIN, 0.0};
    for (int t = 0; t < T; ++t) {
      const Agg& b = part[t][c];
      tot[c].cnt += b.cnt; tot[c].sum += b.sum; tot[c].mn = std::min(tot[c].mn, b.mn);
      tot[c].mx = std::max(tot[c].mx, b.mx); tot[c].dsum += b.dsum;
    }
  }
  return to_rows(tot);
}
