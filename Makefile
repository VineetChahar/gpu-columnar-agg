# One-command entry points (run on a CUDA machine; see scripts/kaggle_run.py for the Kaggle path).
BUILD ?= build
BENCH = $(BUILD)/bench
SAN ?= compute-sanitizer

.PHONY: build verify bench smoke micro clean

build:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release > /dev/null
	cmake --build $(BUILD) -j4

smoke:
	nvcc -O3 -arch=sm_75 -o /tmp/devprop src/devprop.cu && /tmp/devprop

# correctness gate: every variant vs the CPU reference + memcheck + racecheck
verify: build
	$(BENCH) verify
	$(SAN) --tool memcheck --error-exitcode 1 --leak-check full $(BENCH) verify tiny=1
	$(SAN) --tool racecheck --error-exitcode 1 --racecheck-report all $(BENCH) verify tiny=1

micro: build
	$(BENCH) micro

clean:
	rm -rf $(BUILD)

# probes recorded in NOTEBOOK.md: hardware counters (expected to be blocked on Kaggle) and nsys availability
probe-profilers: build
	-ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed --launch-count 1 $(BENCH) micro iters=1 bw_bytes=1e8 2>&1 | tail -15
	-which nsys || (apt-get update -qq >/dev/null 2>&1; apt-cache search nsight-systems | head -5)

tpch-data:
	python3 scripts/tpch_prep.py /kaggle/tmp/tpch --sf 10

tpch: build tpch-data
	$(BENCH) tpch dir=/kaggle/tmp/tpch
