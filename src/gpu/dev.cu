#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include "../cpu/cpu.hpp"
#include "dev.cuh"

void* dmalloc_or_skip(size_t bytes, const char* what) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) {
    cudaGetLastError();  // clear sticky-free allocation error
    throw Skip(std::string("oom:") + what);
  }
  return p;
}

int num_sms() {
  thread_local int n = 0;  // per host thread = per GPU
  if (!n) { int d; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d)); }
  return n;
}

template <class T>
static T* upload_padded(const T* src, size_t n, size_t n_pad) {
  T* d = nullptr;
  CK(cudaMalloc(&d, n_pad * sizeof(T)));
  CK(cudaMemset(d, 0, n_pad * sizeof(T)));
  if (n) CK(cudaMemcpy(d, src, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

void DevData::upload(const HostData& H) { upload_range(H, 0, H.n); }
void DevData::upload_range(const HostData& H, u64 r0, u64 r1) {
  release();
  kind = H.kind; n = r1 - r0; card = H.card; card1 = H.card1; card2 = H.card2; sh = H.log_card2;
  n_pad = (n + TILE - 1) / TILE * TILE + TILE;
  if (kind == KeyKind::COMP) {
    codes = upload_padded((const u16*)H.codes.data() + r0, n, n_pad);
    codes2 = upload_padded((const u16*)H.codes2.data() + r0, n, n_pad);
    dict1 = upload_padded(H.dict1.data(), H.dict1.size(), H.dict1.size());
    dict2 = upload_padded(H.dict2.data(), H.dict2.size(), H.dict2.size());
  } else {
    int w = code_bytes(kind);
    codes = upload_padded(H.codes.data() + r0 * w, n * w, n_pad * w);
    dict = upload_padded(H.dict.data(), H.dict.size(), H.dict.size());
  }
  vals = upload_padded(H.vals.data() + r0, n, n_pad);
  std::vector<u32> re;
  std::vector<u16> rv;
  if (r0 == 0 && r1 == H.n) { re = H.run_end; rv = H.run_val; }
  else
    for (size_t i = std::upper_bound(H.run_end.begin(), H.run_end.end(), (u32)r0) - H.run_end.begin();
         i < H.run_end.size(); ++i) {
      re.push_back((u32)(std::min<u64>(H.run_end[i], r1) - r0));
      rv.push_back(H.run_val[i]);
      if (H.run_end[i] >= r1) break;
    }
  nruns = (u32)re.size();
  run_end = upload_padded(re.data(), nruns, nruns + 32);
  run_val = upload_padded(rv.data(), nruns, nruns + 32);
  bytes_in = H.bytes_compressed() * n / std::max<u64>(H.n, 1);
}
void DevData::release() {
  for (void* p : {codes, codes2, (void*)dict, (void*)dict1, (void*)dict2, (void*)vals, (void*)run_end, (void*)run_val})
    if (p) CK(cudaFree(p));
  codes = codes2 = nullptr; dict = nullptr; dict1 = dict2 = nullptr; vals = nullptr; run_end = nullptr; run_val = nullptr;
}

void GTable::alloc(u64 capacity, bool aos_layout) {
  release();
  cap = capacity;
  aos = aos_layout;
  t.shift = 64 - ilog2(cap);
  t.mask = (u32)(cap - 1);
  if (aos) {
    base = dmalloc_or_skip(cap * 64, "table");
    u64* b = (u64*)base;
    t.keys = b; t.cnt = b + 1; t.sum = b + 2; t.dsum = (double*)(b + 3);
    t.mn = (i32*)(b + 4); t.mx = (i32*)(b + 4) + 1;
    t.m8 = 8; t.m4 = 16;
  } else {
    base = dmalloc_or_skip(cap * 40, "table");
    char* b = (char*)base;
    t.keys = (u64*)b; t.cnt = (u64*)(b + cap * 8); t.sum = (u64*)(b + cap * 16); t.dsum = (double*)(b + cap * 24);
    t.mn = (i32*)(b + cap * 32); t.mx = (i32*)(b + cap * 36);
    t.m8 = 1; t.m4 = 1;
  }
}
void GTable::release() {
  if (!cap) return;
  CK(cudaFree(base));
  base = nullptr;
  cap = 0;
}

static void alloc_res(ResV& r, u64 cap) {
  r.key = (u64*)dmalloc_or_skip(cap * 8, "results");
  r.cnt = (u64*)dmalloc_or_skip(cap * 8, "results");
  r.sum = (i64*)dmalloc_or_skip(cap * 8, "results");
  r.dsum = (double*)dmalloc_or_skip(cap * 8, "results");
  r.mn = (i32*)dmalloc_or_skip(cap * 4, "results");
  r.mx = (i32*)dmalloc_or_skip(cap * 4, "results");
  r.n = (u32*)dmalloc_or_skip(4, "results");
}
static void free_res(ResV& r) {
  for (void* p : {(void*)r.key, (void*)r.cnt, (void*)r.sum, (void*)r.dsum, (void*)r.mn, (void*)r.mx, (void*)r.n})
    if (p) CK(cudaFree(p));
  r = ResV{};
}
void Results::alloc(u64 capacity) { release(); cap = capacity; alloc_res(r, cap); }
void Results::release() { if (cap) free_res(r); cap = 0; }
u32 Results::count() const { u32 n; CK(cudaMemcpy(&n, r.n, 4, cudaMemcpyDeviceToHost)); return n; }

__global__ void k_table_init(GT T, u64 cap) {
  for (u64 s = blockIdx.x * (u64)blockDim.x + threadIdx.x; s < cap; s += (u64)gridDim.x * blockDim.x) {
    T.K(s) = EMPTY_KEY; T.C(s) = 0; T.S(s) = 0; T.D(s) = 0.0; T.MN(s) = INT_MAX; T.MX(s) = INT_MIN;
  }
}
void table_init(const GTable& T, cudaStream_t st) {
  k_table_init<<<num_sms() * 8, 256, 0, st>>>(T.t, T.cap);
  CK(cudaGetLastError());
}

// Compact non-empty slots into dense results (warp-aggregated append).
__global__ void k_extract(GT T, u64 cap, ResV R) {
  const u64 gw = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5, nw = ((u64)gridDim.x * blockDim.x) >> 5;
  for (u64 base = gw * 32; base < cap; base += nw * 32) {
    u64 s = base + lane_id();
    bool ok = s < cap && T.K(s) != EMPTY_KEY;
    u32 b = __ballot_sync(FULL, ok);
    u32 off = 0;
    if (lane_id() == 0 && b) off = atomicAdd(R.n, __popc(b));
    off = __shfl_sync(FULL, off, 0);
    if (ok) {
      u32 i = off + __popc(b & lanemask_lt());
      R.key[i] = T.K(s); R.cnt[i] = T.C(s); R.sum[i] = (i64)T.S(s);
      R.mn[i] = T.MN(s); R.mx[i] = T.MX(s); R.dsum[i] = T.D(s);
    }
  }
}
void table_extract(const GTable& T, const Results& R, cudaStream_t st) {
  CK(cudaMemsetAsync(R.r.n, 0, 4, st));
  k_extract<<<num_sms() * 8, 256, 0, st>>>(T.t, T.cap, R.r);
  CK(cudaGetLastError());
}

__global__ void k_merge_results(GT T, ResV R, u64 n) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
    u64 key = R.key[i];
    u32 s = gt_slot(T, key, fmix64(key));
    gt_add(T, s, R.cnt[i], R.sum[i], R.mn[i], R.mx[i], R.dsum[i]);
  }
}
void table_merge_results(const GTable& T, const Results& R, u64 n, cudaStream_t st) {
  if (!n) return;
  k_merge_results<<<num_sms() * 8, 256, 0, st>>>(T.t, R.r, n);
  CK(cudaGetLastError());
}

void DevRef::upload(const std::vector<GroupRow>& ref) {
  release();
  n = ref.size();
  u64 c = n ? n : 1;
  alloc_res(r, c);
  std::vector<u64> k(c), cn(c); std::vector<i64> s(c); std::vector<i32> mn(c), mx(c); std::vector<double> ds(c);
  for (u64 i = 0; i < n; ++i) {
    k[i] = ref[i].key; cn[i] = ref[i].a.cnt; s[i] = ref[i].a.sum; mn[i] = ref[i].a.mn; mx[i] = ref[i].a.mx; ds[i] = ref[i].a.dsum;
  }
  CK(cudaMemcpy(r.key, k.data(), c * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.cnt, cn.data(), c * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.sum, s.data(), c * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dsum, ds.data(), c * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.mn, mn.data(), c * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.mx, mx.data(), c * 4, cudaMemcpyHostToDevice));
}
void DevRef::release() { if (r.key) free_res(r); n = 0; }

__global__ void k_iota(u32* p, u64 n) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) p[i] = (u32)i;
}
// out[0] = mismatches, out[1] = bits of max relative dsum error (non-negative doubles order like u64)
__global__ void k_compare(const u64* skeys, const u32* perm, ResV R, ResV F, u64 n, unsigned long long* out) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
    u32 j = perm[i];
    bool bad = skeys[i] != F.key[i] || (u64)R.cnt[j] != F.cnt[i] || R.sum[j] != F.sum[i] || R.mn[j] != F.mn[i] ||
               R.mx[j] != F.mx[i];
    double a = R.dsum[j], b = F.dsum[i];
    double rel = fabs(a - b) / fmax(fabs(b), 1.0);
    if (rel > 1e-9) bad = true;
    if (bad) atomicAdd(&out[0], 1ull);
    atomicMax(&out[1], (unsigned long long)__double_as_longlong(rel));
  }
}

CheckResult check_results(const Results& R, const DevRef& ref, cudaStream_t st) {
  CK(cudaStreamSynchronize(st));
  CheckResult c{true, 0, 0, 0.0, ""};
  u64 n = R.count();
  c.groups = n;
  if (n != ref.n) {
    c.ok = false; c.mismatches = (n > ref.n ? n - ref.n : ref.n - n);
    c.msg = "group count " + std::to_string(n) + " != ref " + std::to_string(ref.n);
    return c;
  }
  if (!n) return c;
  u64 *skeys; u32 *perm, *sperm; void* tmp = nullptr; size_t tb = 0; unsigned long long* out;
  CK(cudaMalloc(&skeys, n * 8)); CK(cudaMalloc(&perm, n * 4)); CK(cudaMalloc(&sperm, n * 4)); CK(cudaMalloc(&out, 16));
  CK(cudaMemset(out, 0, 16));
  k_iota<<<256, 256, 0, st>>>(perm, n);
  CK(cub::DeviceRadixSort::SortPairs(tmp, tb, R.r.key, skeys, perm, sperm, (int)n, 0, 64, st));
  CK(cudaMalloc(&tmp, tb));
  CK(cub::DeviceRadixSort::SortPairs(tmp, tb, R.r.key, skeys, perm, sperm, (int)n, 0, 64, st));
  k_compare<<<512, 256, 0, st>>>(skeys, sperm, R.r, ref.r, n, out);
  CK(cudaGetLastError());
  unsigned long long h[2];
  CK(cudaMemcpyAsync(h, out, 16, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  c.mismatches = h[0];
  long long bits = (long long)h[1];
  memcpy(&c.max_rel_dsum, &bits, 8);
  c.ok = h[0] == 0;
  if (!c.ok) c.msg = std::to_string(h[0]) + " mismatching groups";
  CK(cudaFree(skeys)); CK(cudaFree(perm)); CK(cudaFree(sperm)); CK(cudaFree(out)); CK(cudaFree(tmp));
  return c;
}
