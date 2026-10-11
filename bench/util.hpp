// Bench utilities: JSON-lines emitter, timing statistics, NVML clock sampling, argument parsing.
#pragma once
#include <cuda_runtime.h>
#include <nvml.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../src/gpu/dev.cuh"

// ---------------- JSON ----------------
struct Json {
  std::ostringstream o;
  bool first = true;
  Json() { o << "{"; }
  Json& key(const std::string& k) { o << (first ? "" : ",") << "\"" << k << "\":"; first = false; return *this; }
  Json& s(const std::string& k, const std::string& v) {
    key(k); o << "\"";
    for (char c : v) { if (c == '"' || c == '\\') o << '\\'; o << (c == '\n' ? ' ' : c); }
    o << "\""; return *this;
  }
  Json& n(const std::string& k, double v) {
    key(k);
    if (std::isfinite(v)) { char b[64]; snprintf(b, sizeof b, "%.9g", v); o << b; } else o << "null";
    return *this;
  }
  Json& b(const std::string& k, bool v) { key(k); o << (v ? "true" : "false"); return *this; }
  Json& raw(const std::string& k, const std::string& v) { key(k); o << v; return *this; }
  std::string str() { return o.str() + "}"; }
};
inline std::string results_path() {
  const char* d = getenv("RESULTS_DIR");
  return std::string(d ? d : ".") + "/results.jsonl";
}
inline void emit(Json& j) {
  std::string line = j.str();
  std::ofstream f(results_path(), std::ios::app);
  f << line << "\n";
}

// ---------------- stats ----------------
struct Stat {
  double med = 0, p25 = 0, p75 = 0, mn = 0, mx = 0;
  int n = 0;
};
inline Stat stats(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  auto q = [&](double f) {
    double i = f * (v.size() - 1);
    size_t a = (size_t)std::floor(i), b = std::min(a + 1, v.size() - 1);
    return v[a] + (v[b] - v[a]) * (i - a);
  };
  s.med = q(0.5); s.p25 = q(0.25); s.p75 = q(0.75); s.mn = v.front(); s.mx = v.back(); s.n = (int)v.size();
  return s;
}
// CUDA-event timing: `warm` untimed runs, then `iters` timed runs, each bracketed by events.
inline Stat time_gpu(const std::function<void()>& f, cudaStream_t st, int warm, int iters) {
  for (int i = 0; i < warm; ++i) f();
  CK(cudaStreamSynchronize(st));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<double> ms;
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(a, st));
    f();
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b)); ms.push_back(t);
  }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return stats(ms);
}
inline double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------- NVML ----------------
struct Clk { unsigned sm = 0, mem = 0, temp = 0, power_mw = 0; unsigned long long throttle = 0; };
inline nvmlDevice_t nvml_dev() {
  static nvmlDevice_t h{};
  static bool init = false;
  if (!init) {
    init = true;
    if (nvmlInit_v2() != NVML_SUCCESS) return h;
    int d; CK(cudaGetDevice(&d));
    char bus[64]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, d));
    nvmlDeviceGetHandleByPciBusId_v2(bus, &h);
  }
  return h;
}
inline Clk clocks() {
  Clk c;
  nvmlDevice_t h = nvml_dev();
  if (!h) return c;
  nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, &c.sm);
  nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM, &c.mem);
  nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU, &c.temp);
  nvmlDeviceGetPowerUsage(h, &c.power_mw);
  nvmlDeviceGetCurrentClocksThrottleReasons(h, &c.throttle);
  return c;
}
inline void put_clocks(Json& j, const Clk& pre, const Clk& post) {
  j.n("sm_mhz_pre", pre.sm).n("sm_mhz_post", post.sm).n("mem_mhz_pre", pre.mem).n("mem_mhz_post", post.mem)
   .n("temp_c_pre", pre.temp).n("temp_c_post", post.temp).n("power_w_post", post.power_mw / 1000.0)
   .n("throttle_post", (double)post.throttle);
}

// ---------------- args: key=value ----------------
struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv, int start) {
    for (int i = start; i < argc; ++i) {
      std::string a = argv[i];
      auto p = a.find('=');
      if (p == std::string::npos) kv[a] = "1"; else kv[a.substr(0, p)] = a.substr(p + 1);
    }
  }
  std::string get(const std::string& k, const std::string& d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
  double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : parse_num(it->second); }
  static double parse_num(std::string s) {
    double m = 1;
    if (!s.empty() && (s.back() == 'M' || s.back() == 'm')) { m = 1e6; s.pop_back(); }
    else if (!s.empty() && (s.back() == 'K' || s.back() == 'k')) { m = 1e3; s.pop_back(); }
    else if (!s.empty() && (s.back() == 'G' || s.back() == 'g')) { m = 1e9; s.pop_back(); }
    return std::stod(s) * m;
  }
  std::vector<double> list(const std::string& k, const std::string& d) const {
    std::vector<double> out; std::stringstream ss(get(k, d)); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(parse_num(t));
    return out;
  }
  std::vector<std::string> slist(const std::string& k, const std::string& d) const {
    std::vector<std::string> out; std::stringstream ss(get(k, d)); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
    return out;
  }
};
