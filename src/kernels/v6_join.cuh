#pragma once
#include "../gpu/dev.cuh"
#include "../join.hpp"

struct DevJoin {
  u64 D = 0, F = 0;
  u64* dkey = nullptr;
  u8* dcat = nullptr;
  u8* dp = nullptr;
  u64* fdict = nullptr;
  u32* fcode = nullptr;
  i32* fval = nullptr;
  void upload(const JoinData& H);
  void release();
};
struct JHT {  // global build table: key -> cat
  u64* keys = nullptr;
  u8* cats = nullptr;
  u32 shift = 0, mask = 0;
};
struct JoinScratch {
  JHT ht;
  u64 ht_cap = 0;
  u32* bloom = nullptr;
  u64 bloom_words = 0;
  unsigned long long* stats = nullptr;
  u8* xlat = nullptr;
  u64* bkey = nullptr; size_t bkey_cap = 0;
  u32* bcat = nullptr; size_t bcat_cap = 0;
  u64* pkey = nullptr; size_t pkey_cap = 0;
  i32* pval = nullptr; size_t pval_cap = 0;
  u32* hist = nullptr; size_t hist_cap = 0;
  u32* hscan = nullptr; size_t hscan_cap = 0;
  u32* starts = nullptr; size_t starts_cap = 0;
  u32* items = nullptr; size_t items_cap = 0;
  u32* bst = nullptr; size_t bst_cap = 0;
  void* tmp = nullptr; size_t tmp_cap = 0;
  void alloc(const DevJoin& J);
  void release();
};
#include <map>
#include <string>
struct JoinInfo {
  std::map<std::string, double> phase_ms;
  u64 probed = 0, matched = 0, build_rows = 0, max_build_partition = 0;
  int partition_bits = 0;
};
void run_join_np(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, GTable& out, cudaStream_t st, JoinInfo& info);
void run_join_part(const DevJoin& J, JoinScratch& S, u8 sel, bool bloom, GTable& out, cudaStream_t st, JoinInfo& info);
void run_join_dict(const DevJoin& J, JoinScratch& S, u8 sel, GTable& out, cudaStream_t st, JoinInfo& info);
void build_bloom(const u64* keys, u32 n, u32* bloom, u32 bmask, cudaStream_t st);
