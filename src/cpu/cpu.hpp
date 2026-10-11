#pragma once
#include <vector>
#include "../data.hpp"

struct GroupRow {
  u64 key;
  Agg a;
};

// Single-threaded reference.  Deliberately a *different algorithm* from every GPU variant:
// it aggregates by dense dictionary code into plain arrays (no hashing), then decodes.
// Output sorted by key.
std::vector<GroupRow> cpu_reference(const HostData& D, u16 lo, u16 hi);

// Multithreaded baseline engine (same algorithm class as the GPU: hash aggregation on decoded
// keys).  Thread-local tables for small cardinality, radix-partitioned two-phase otherwise.
std::vector<GroupRow> cpu_engine(const HostData& D, u16 lo, u16 hi, int threads);

// Hash join + aggregate baselines (see join.hpp for the data model).
