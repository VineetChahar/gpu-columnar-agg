#include <algorithm>
#include <cmath>
#include <thread>
#include "data.hpp"

AliasTable build_zipf_alias(u32 card, double alpha) {
  // Vose alias method over p_k ~ 1/(k+1)^alpha.  O(card) build, O(1) sample.
  AliasTable A;
  A.prob.assign(card, 1.0f);
  A.alias.resize(card);
  for (u32 i = 0; i < card; ++i) A.alias[i] = i;
  if (alpha == 0.0) return A;
  std::vector<double> p(card);
  double tot = 0;
  for (u32 k = 0; k < card; ++k) tot += (p[k] = std::pow((double)k + 1.0, -alpha));
  std::vector<u32> small, large;
  for (u32 k = 0; k < card; ++k) {
    p[k] = p[k] / tot * card;
    (p[k] < 1.0 ? small : large).push_back(k);
  }
  while (!small.empty() && !large.empty()) {
    u32 s = small.back(); small.pop_back();
    u32 l = large.back();
    A.prob[s] = (float)p[s];
    A.alias[s] = l;
    p[l] = (p[l] + p[s]) - 1.0;
    if (p[l] < 1.0) { large.pop_back(); small.push_back(l); }
  }
  for (u32 k : large) A.prob[k] = 1.0f;
  for (u32 k : small) A.prob[k] = 1.0f;
  return A;
}

void gen_setup(HostData& D, const GenSpec& s) {
  D.spec = s;
  D.n = s.rows;
  D.card = s.card;
  if (s.composite) {
    int lg = ilog2(s.card);
    D.card1 = 1u << (lg - lg / 2);
    D.card2 = 1u << (lg / 2);
    D.log_card2 = lg / 2;
    if ((u64)D.card1 * D.card2 != s.card || D.card1 > 65536 || D.card2 > 65536) {
      fprintf(stderr, "composite card must be a power of two <= 2^32\n"); exit(1);
    }
    D.kind = KeyKind::COMP;
  } else {
    D.kind = s.card <= 256 ? KeyKind::U8 : s.card <= 65536 ? KeyKind::U16 : KeyKind::U32;
  }
  // dictionaries: random, distinct values (splitmix64 / fmix32 are bijections).
  if (D.kind == KeyKind::COMP) {
    D.dict1.resize(D.card1); D.dict2.resize(D.card2);
    for (u32 c = 0; c < D.card1; ++c) D.dict1[c] = fmix32(c ^ (u32)s.seed ^ 0x1234567u);
    for (u32 c = 0; c < D.card2; ++c) D.dict2[c] = fmix32(c ^ (u32)(s.seed >> 7) ^ 0x7654321u);
    for (u32 c = 0; c < D.card1; ++c)
      if (D.dict1[c] == 0xffffffffu) D.dict1[c] = 0x7fffffffu;  // keep EMPTY_KEY unreachable
  } else {
    D.dict.resize(s.card);
    for (u32 c = 0; c < s.card; ++c) {
      D.dict[c] = splitmix64((u64)c ^ (s.seed * 0x51ed27f1ULL));
      if (D.dict[c] == EMPTY_KEY) { fprintf(stderr, "dictionary hit EMPTY_KEY\n"); exit(1); }
    }
  }
  // RLE predicate column: runs of length uniform in [1, 2*mean_run-1], value uniform in [0,100).
  D.run_end.clear(); D.run_val.clear();
  D.run_end.reserve(s.rows / s.mean_run + 16);
  D.run_val.reserve(s.rows / s.mean_run + 16);
  u64 pos = 0, i = 0;
  while (pos < s.rows) {
    u64 x = splitmix64(row_seed(s.seed, 3, i++));
    u64 len = 1 + (x % (2 * s.mean_run - 1));
    pos = std::min<u64>(pos + len, s.rows);
    D.run_end.push_back((u32)pos);
    D.run_val.push_back((u16)((x >> 32) % 100));
  }
  int w = code_bytes(D.kind);
  if (D.kind == KeyKind::COMP) { D.codes.resize(D.n * 2); D.codes2.resize(D.n * 2); }
  else { D.codes.resize(D.n * w); D.codes2.clear(); }
  D.vals.resize(D.n);
}

void gen_rows_host(HostData& D, const AliasTable& A) {
  unsigned T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  bool uni = D.spec.zipf == 0.0;
  for (unsigned t = 0; t < T; ++t) {
    th.emplace_back([&, t] {
      u64 a = D.n * t / T, b = D.n * (t + 1) / T;
      for (u64 r = a; r < b; ++r) {
        u32 c = sample_code(row_seed(D.spec.seed, 1, r), D.card, A.prob.data(), A.alias.data(), uni);
        switch (D.kind) {
          case KeyKind::U8: D.codes[r] = (u8)c; break;
          case KeyKind::U16: reinterpret_cast<u16*>(D.codes.data())[r] = (u16)c; break;
          case KeyKind::U32: reinterpret_cast<u32*>(D.codes.data())[r] = c; break;
          case KeyKind::COMP:
            reinterpret_cast<u16*>(D.codes.data())[r] = (u16)(c >> D.log_card2);
            reinterpret_cast<u16*>(D.codes2.data())[r] = (u16)(c & (D.card2 - 1));
            break;
        }
        D.vals[r] = sample_val(row_seed(D.spec.seed, 2, r));
      }
    });
  }
  for (auto& x : th) x.join();
}
