// Device-side data model: compressed columns, key decoders, global hash table, result buffers.
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <string>
#include "../common.hpp"
#include "../data.hpp"

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                                  \
    }                                                                                           \
  } while (0)

constexpr unsigned FULL = 0xffffffffu;
constexpr int TILE = 4096;  // rows per warp work unit; device arrays are padded to a multiple of this

// Thrown when a variant cannot run at a sweep point (e.g. partition buffers do not fit in 15 GB).
struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };
void* dmalloc_or_skip(size_t bytes, const char* what);

__device__ __forceinline__ u32 lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ u32 lanemask_lt() { return (1u << lane_id()) - 1u; }

// ---------------- key decoders (dictionary decode fused into the scan) ----------------
template <class CodeT>
struct KD1 {
  const CodeT* c;
  const u64* d;
  __device__ __forceinline__ u32 code(u64 r) const { return c[r]; }
  __device__ __forceinline__ void code4(u64 r, u32 o[4]) const;
  __device__ __forceinline__ u64 decode(u32 k) const { return __ldg(d + k); }
};
template <>
__device__ __forceinline__ void KD1<u8>::code4(u64 r, u32 o[4]) const {
  u32 w = *reinterpret_cast<const u32*>(c + r);
  o[0] = w & 0xff; o[1] = (w >> 8) & 0xff; o[2] = (w >> 16) & 0xff; o[3] = w >> 24;
}
template <>
__device__ __forceinline__ void KD1<u16>::code4(u64 r, u32 o[4]) const {
  uint2 w = *reinterpret_cast<const uint2*>(c + r);
  o[0] = w.x & 0xffff; o[1] = w.x >> 16; o[2] = w.y & 0xffff; o[3] = w.y >> 16;
}
template <>
__device__ __forceinline__ void KD1<u32>::code4(u64 r, u32 o[4]) const {
  uint4 w = *reinterpret_cast<const uint4*>(c + r);
  o[0] = w.x; o[1] = w.y; o[2] = w.z; o[3] = w.w;
}
// composite key: two u16 dictionary-encoded columns; dense code = c1 << sh | c2
struct KD2 {
  const u16* c1;
  const u16* c2;
  const u32* d1;
  const u32* d2;
  int sh;
  u32 m2;
  __device__ __forceinline__ u32 code(u64 r) const { return (u32(c1[r]) << sh) | c2[r]; }
  __device__ __forceinline__ void code4(u64 r, u32 o[4]) const {
    uint2 a = *reinterpret_cast<const uint2*>(c1 + r);
    uint2 b = *reinterpret_cast<const uint2*>(c2 + r);
    o[0] = ((a.x & 0xffff) << sh) | (b.x & 0xffff);
    o[1] = ((a.x >> 16) << sh) | (b.x >> 16);
    o[2] = ((a.y & 0xffff) << sh) | (b.y & 0xffff);
    o[3] = ((a.y >> 16) << sh) | (b.y >> 16);
  }
  __device__ __forceinline__ u64 decode(u32 k) const {
    return (u64(__ldg(d1 + (k >> sh))) << 32) | __ldg(d2 + (k & m2));
  }
};

// A (possibly chunked) view of the compressed columns.  Arrays are indexed by local row r;
// run_end holds absolute row numbers, row0 is the absolute row of local row 0.
template <class KD>
struct Cols {
  KD kd;
  const i32* vals;
  const u32* run_end;
  const u16* run_val;
  u32 nruns;
  u64 row0;
  u64 n;
  u16 lo, hi;
};

// Owning device copy of a HostData.
struct DevData {
  KeyKind kind{};
  u64 n = 0, n_pad = 0;
  u32 card = 0, card1 = 0, card2 = 0;
  int sh = 0;
  void* codes = nullptr;
  void* codes2 = nullptr;
  u64* dict = nullptr;
  u32* dict1 = nullptr;
  u32* dict2 = nullptr;
  i32* vals = nullptr;
  u32* run_end = nullptr;
  u16* run_val = nullptr;
  u32 nruns = 0;
  size_t bytes_in = 0;  // compressed bytes a full scan reads
  void upload(const HostData& H);
  // rows [r0, r1) only (multi-GPU split); run ends are rebased to the slice, dictionaries are copied whole
  void upload_range(const HostData& H, u64 r0, u64 r1);
  void release();
};

template <class F>
void with_kd(const DevData& D, F&& f) {
  switch (D.kind) {
    case KeyKind::U8: f(KD1<u8>{(const u8*)D.codes, D.dict}); break;
    case KeyKind::U16: f(KD1<u16>{(const u16*)D.codes, D.dict}); break;
    case KeyKind::U32: f(KD1<u32>{(const u32*)D.codes, D.dict}); break;
    case KeyKind::COMP:
      f(KD2{(const u16*)D.codes, (const u16*)D.codes2, D.dict1, D.dict2, D.sh, D.card2 - 1});
      break;
  }
}
template <class KD>
Cols<KD> make_cols(const DevData& D, KD kd, u16 lo, u16 hi) {
  return Cols<KD>{kd, D.vals, D.run_end, D.run_val, D.nruns, 0, D.n, lo, hi};
}
// view of rows [r0, r0 + n) (r0 a multiple of TILE so vector loads stay aligned)
template <class T>
KD1<T> kd_offset(KD1<T> k, u64 r0) { k.c += r0; return k; }
inline KD2 kd_offset(KD2 k, u64 r0) { k.c1 += r0; k.c2 += r0; return k; }
template <class KD>
Cols<KD> make_cols_range(const DevData& D, KD kd, u16 lo, u16 hi, u64 r0, u64 n) {
  return Cols<KD>{kd_offset(kd, r0), D.vals + r0, D.run_end, D.run_val, D.nruns, r0, n, lo, hi};
}

// ---------------- global open-addressing hash table ----------------
// Two layouts behind one view (index multipliers m8/m4 for 8- and 4-byte fields):
//   AoS (default): one 64-byte aligned slot {key, cnt, sum, dsum, mn, mx} -> the five atomics of an
//                  update touch two 32-byte sectors of ONE line instead of six different lines.
//   SoA (ablation): six separate arrays.
struct GT {
  u64* keys;
  u64* cnt;
  u64* sum;  // int64 stored as two's complement u64 so atomicAdd(ull) applies
  double* dsum;
  i32* mn;
  i32* mx;
  u32 m8, m4;  // element stride of 8-byte / 4-byte fields (SoA 1,1; AoS 8,16)
  u32 shift;   // slot = hash >> shift  (top log2(cap) bits)
  u32 mask;
  __host__ __device__ __forceinline__ u64& K(u64 s) const { return keys[s * m8]; }
  __host__ __device__ __forceinline__ u64& C(u64 s) const { return cnt[s * m8]; }
  __host__ __device__ __forceinline__ u64& S(u64 s) const { return sum[s * m8]; }
  __host__ __device__ __forceinline__ double& D(u64 s) const { return dsum[s * m8]; }
  __host__ __device__ __forceinline__ i32& MN(u64 s) const { return mn[s * m4]; }
  __host__ __device__ __forceinline__ i32& MX(u64 s) const { return mx[s * m4]; }
  // view of slots [off, ...)
  GT slice(u64 off) const {
    GT g = *this;
    g.keys += off * m8; g.cnt += off * m8; g.sum += off * m8; g.dsum += off * m8; g.mn += off * m4; g.mx += off * m4;
    return g;
  }
};
struct GTable {
  GT t{};
  u64 cap = 0;
  bool aos = true;
  void* base = nullptr;
  void alloc(u64 capacity, bool aos_layout = true);
  void release();
  size_t bytes() const { return aos ? cap * 64 : cap * 40; }
};

__device__ __forceinline__ u32 gt_slot(const GT& T, u64 key, u64 h) {
  u32 s = (u32)(h >> T.shift);
  while (true) {
    u64 k = T.K(s);
    if (k == key) return s;
    if (k == EMPTY_KEY) {
      u64 old = atomicCAS((unsigned long long*)&T.K(s), (unsigned long long)EMPTY_KEY, (unsigned long long)key);
      if (old == EMPTY_KEY || old == key) return s;
    }
    s = (s + 1) & T.mask;
  }
}
__device__ __forceinline__ void gt_add(const GT& T, u32 s, u64 c, i64 sum, i32 mn, i32 mx, double ds) {
  atomicAdd((unsigned long long*)&T.C(s), (unsigned long long)c);
  atomicAdd((unsigned long long*)&T.S(s), (unsigned long long)sum);
  atomicAdd(&T.D(s), ds);
  atomicMin(&T.MN(s), mn);
  atomicMax(&T.MX(s), mx);
}

// ---------------- dense result buffers (what every variant must produce) ----------------
struct ResV {
  u64* key;
  u64* cnt;
  i64* sum;
  i32* mn;
  i32* mx;
  double* dsum;
  u32* n;
};
struct Results {
  ResV r{};
  u64 cap = 0;
  void alloc(u64 capacity);
  void release();
  u32 count() const;
};

void table_init(const GTable& T, cudaStream_t st);
void table_extract(const GTable& T, const Results& R, cudaStream_t st);
// insert every row of a result set into T (hash on the decoded key) - merges partial results
void table_merge_results(const GTable& T, const Results& R, u64 n, cudaStream_t st);

// ---------------- validation against the CPU reference ----------------
struct GroupRow;
struct DevRef {
  ResV r{};
  u64 n = 0;
  void upload(const std::vector<GroupRow>& ref);
  void release();
};
struct CheckResult {
  bool ok;
  u64 groups;
  u64 mismatches;
  double max_rel_dsum;
  std::string msg;
};
CheckResult check_results(const Results& R, const DevRef& ref, cudaStream_t st);

int num_sms();
