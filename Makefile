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

# ---- full benchmark, split into sessions that fit a Kaggle run (each appends to results.jsonl) ----
CARDS = 16,256,4096,65536,1048576,16777216
ABL = v0_naive,v0_naive_soa,v1_vec,v1_nocompact,v3_warpagg,v3_global,v4_radix_smem,v4_radix_1pass,v2_smem_soa1024,v3_warpagg512,v3_warpagg1024

bench-part1: build
	$(BENCH) micro
	$(BENCH) sweep rows=10M,100M cards=$(CARDS) skews=0,0.5,1,1.5 sels=1,10,50,100 variants=main tag=sweep
	$(BENCH) sweep rows=100M cards=$(CARDS) skews=0,1.5 sels=10,100 variants=$(ABL) tag=ablation cpu=0
	$(BENCH) sweep rows=100M cards=$(CARDS) skews=0,1 sels=100 variants=main composite=1 tag=sweep

bench-part2: build
	$(BENCH) sweep rows=500M cards=$(CARDS) skews=0,0.5,1,1.5 sels=1,10,50,100 variants=main tag=sweep cpu_budget_s=15

bench-part3: build
	$(BENCH) stream rows=500M tag=stream
	$(BENCH) join D=65536,1048576,16777216 F=100M tag=join
	python3 scripts/tpch_prep.py /kaggle/tmp/tpch --sf 10
	$(BENCH) tpch dir=/kaggle/tmp/tpch
	python3 scripts/cudf_sweep.py --rows 10M,100M

bench: bench-part1 bench-part2 bench-part3

# nsys is not on the Kaggle image but installs from NVIDIA's apt repo; trace one V5 pipeline
nsys: build
	which nsys || (apt-get install -y -qq nsight-systems-2024.5.1 >/dev/null 2>&1 || apt-get install -y -qq nsight-systems-2024.4.2 >/dev/null 2>&1); \
	NSYS=$$(which nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1); echo "nsys: $$NSYS"; \
	$$NSYS profile -t cuda -o $${RESULTS_DIR:-.}/v5_stream -f true $(BENCH) stream rows=100M iters=2 queries=groupby_card16 tag=nsys_stream && \
	$$NSYS stats --report cuda_gpu_trace --format csv -o $${RESULTS_DIR:-.}/v5_stream $${RESULTS_DIR:-.}/v5_stream.nsys-rep >/dev/null && ls -la $${RESULTS_DIR:-.} | grep v5_stream
