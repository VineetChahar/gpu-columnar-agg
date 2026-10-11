// Shared types, hashing and RNG used by host (CPU engine / generator) and device code.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr u64 EMPTY_KEY = ~0ull;

// murmur3 finalizer: the one hash every hash table / partitioner in the repo uses.
HD u64 fmix64(u64 k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}
// bijective 32-bit mixer (murmur3 fmix32) — used for composite-key dictionaries so values stay distinct.
HD u32 fmix32(u32 h) {
  h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h;
}
// counter-based RNG: row i of column c is splitmix64(seed ^ (c << 56) ^ i), identical on host and device.
HD u64 splitmix64(u64 x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// The floating-point aggregate: SUM(v * 0.01) accumulated in double (think "price in dollars").
HD double vscale(i32 v) { return (double)v * 0.01; }

struct Agg {
  u64 cnt;
  i64 sum;
  i32 mn, mx;
  double dsum;
};

enum class KeyKind : int { U8 = 0, U16 = 1, U32 = 2, COMP = 3 };
inline const char* key_kind_name(KeyKind k) {
  switch (k) { case KeyKind::U8: return "u8"; case KeyKind::U16: return "u16";
               case KeyKind::U32: return "u32"; default: return "comp16x16"; }
}
inline int code_bytes(KeyKind k) {
  switch (k) { case KeyKind::U8: return 1; case KeyKind::U16: return 2;
               case KeyKind::U32: return 4; default: return 4; /* two u16 columns */ }
}
inline int ilog2(u64 x) { int r = 0; while ((1ull << (r + 1)) <= x) ++r; return r; }
inline u64 next_pow2(u64 x) { u64 p = 1; while (p < x) p <<= 1; return p; }

// Global hash-table capacity rule, identical for every variant (load factor <= 0.5 by construction).
inline u64 table_capacity(u64 card) { return next_pow2(card * 2 < 1024 ? 1024 : card * 2); }
