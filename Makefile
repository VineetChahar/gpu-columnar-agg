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
ABL = v0_naive,v0_naive_soa,v1_vec,v1_nocompact,v3_warpagg,v3_global,v4_radix_smem,v4_radix_nostage,v4_radix_1pass,v2_smem_soa1024,v3_warpagg512,v3_warpagg1024

bench-part1a: build
	$(BENCH) micro
	$(BENCH) sweep rows=10M,100M cards=$(CARDS) skews=0,0.5,1,1.5 sels=1,10,50,100 variants=main tag=sweep

bench-part1b: build
	$(MAKE) nsys
	$(BENCH) sweep rows=100M cards=$(CARDS) skews=0,1.5 sels=10,100 variants=$(ABL) tag=ablation cpu=0
	$(BENCH) sweep rows=100M cards=$(CARDS) skews=0,1 sels=100 variants=main composite=1 tag=sweep
	$(BENCH) stream rows=200M tag=stream
	$(BENCH) join D=65536,1048576,16777216 F=100M tag=join
	python3 scripts/tpch_prep.py /kaggle/tmp/tpch --sf 10
	$(BENCH) tpch dir=/kaggle/tmp/tpch
	python3 scripts/cudf_sweep.py --rows 10M,100M

bench-part2a: build
	$(BENCH) sweep rows=500M cards=16,256,4096 skews=0,0.5,1,1.5 sels=1,10,50,100 variants=main tag=sweep cpu_budget_s=15

bench-part2b: build
	$(BENCH) sweep rows=500M cards=65536,1048576,16777216 skews=0,0.5,1,1.5 sels=1,10,50,100 variants=main tag=sweep cpu_budget_s=15

bench-part1: bench-part1a bench-part1b
bench-part2: bench-part2a bench-part2b
# hash-table capacity from estimated groups instead of dictionary size (same for every variant)
bench-part3: build
	$(BENCH) verify variants=v3_latemat,v3_warpagg
	$(BENCH) sweep rows=10M,100M cards=65536,1048576,16777216 skews=0,1.5 sels=1,10,50,100 variants=main cap=est tag=capest
	$(BENCH) sweep rows=100M cards=$(CARDS) skews=0,1.5 sels=1,10,50,100 variants=v3_warpagg,v3_latemat tag=latemat cpu=0

# sanitizers for the late-materialisation front end (added after the main verify run); sel 1% exercises the skipped loads
verify-latemat-san: build
	$(SAN) --tool memcheck --error-exitcode 1 --leak-check full $(BENCH) verify tiny=1 sels=1,10,100 variants=v3_latemat
	$(SAN) --tool racecheck --error-exitcode 1 --racecheck-report all $(BENCH) verify tiny=1 sels=1,10,100 variants=v3_latemat

# ---- follow-up: fixes for the gaps recorded in RESULTS (OOM at 500M, cap=est at 500M, full Q6, multi-GPU) ----
# correctness gates for the new code paths, incl. a forced multi-chunk plan (chunk=4096 rows) and the
# multi-GPU split + merge on small inputs; sanitizers on both
verify-followup: build
	$(BENCH) verify variants=v4_sort,v4_sort_chunked
	$(BENCH) verify variants=v4_sort_chunked chunk=4096
	$(SAN) --tool memcheck --error-exitcode 1 --leak-check full $(BENCH) verify tiny=1 sels=1,50,100 variants=v4_sort_chunked chunk=4096
	$(SAN) --tool racecheck --error-exitcode 1 --racecheck-report all $(BENCH) verify tiny=1 sels=1,50,100 variants=v4_sort_chunked chunk=4096
	$(BENCH) mgpu rows=1000003,20000000 cards=16,65536,1048576 skews=0,1.5 sels=1,100 gpus=2,1 variants=v0_naive,v1_vec,v2_smem_soa,v3_warpagg,v4_radix_l2,v4_sort,v4_sort_chunked iters=3 warmup=1 tag=mgpu_verify
	$(SAN) --tool memcheck --error-exitcode 1 $(BENCH) mgpu rows=100000 cards=4096 skews=0 sels=50 gpus=2 variants=v3_warpagg,v4_sort_chunked iters=1 warmup=0 tag=mgpu_san

# chunked sort at 500M (where v4_sort runs out of memory) + chunking overhead at 100M; full TPC-H Q6
bench-part4: verify-followup
	$(BENCH) sweep rows=500M cards=65536,1048576,16777216 skews=0,0.5,1,1.5 sels=1,10,50,100 variants=v4_sort_chunked tag=chunked cpu=0
	$(BENCH) sweep rows=100M cards=65536,1048576,16777216 skews=0,1.5 sels=10,100 variants=v4_sort,v4_sort_chunked tag=chunked cpu=0
	$(BENCH) sweep rows=100M cards=65536,1048576,16777216 skews=0,1.5 sels=10,100 variants=v4_sort_chunked chunk=16777216 tag=chunked16M cpu=0
	python3 scripts/tpch_prep.py /kaggle/tmp/tpch --sf 10
	$(BENCH) tpch dir=/kaggle/tmp/tpch only=q6full

# capacity policy at 500M (hash variants are the fallback there) + row-split multi-GPU (T4 x2)
bench-part5: build
	$(BENCH) mgpu rows=100M,500M cards=4096,65536,1048576,16777216 skews=0,1.5 sels=1,100 gpus=2,1 variants=v1_vec,v3_warpagg,v4_radix_l2,v4_sort,v4_sort_chunked tag=mgpu
	$(BENCH) sweep rows=500M cards=1048576,16777216 skews=0,1.5 sels=1,10,50,100 variants=main cap=est tag=capest cpu=0

# multi-GPU re-run after sizing the merge table by the partial groups that arrive (was dictionary-sized);
# correctness gate + memcheck first, then the same grid as bench-part5 (rows with the same key replace the old ones)
bench-part5b: build
	$(BENCH) mgpu rows=1000003,20000000 cards=16,65536,1048576 skews=0,1.5 sels=1,100 gpus=2 variants=v1_vec,v3_warpagg,v4_sort,v4_sort_chunked iters=2 warmup=1 tag=mgpu_verify2
	$(SAN) --tool memcheck --error-exitcode 1 $(BENCH) mgpu rows=100000 cards=4096 skews=0 sels=50 gpus=2 variants=v3_warpagg,v4_sort_chunked iters=1 warmup=0 tag=mgpu_san2
	$(BENCH) mgpu rows=100M,500M cards=4096,65536,1048576,16777216 skews=0,1.5 sels=1,100 gpus=2,1 variants=v1_vec,v3_warpagg,v4_radix_l2,v4_sort,v4_sort_chunked tag=mgpu

bench: bench-part1 bench-part2 bench-part3

# nsys is not on the Kaggle image but installs from NVIDIA's apt repo; trace one V5 pipeline
nsys: build
	which nsys || (apt-get install -y -qq nsight-systems-2024.5.1 >/dev/null 2>&1 || apt-get install -y -qq nsight-systems-2024.4.2 >/dev/null 2>&1); \
	NSYS=$$(which nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1); echo "nsys: $$NSYS"; \
	$$NSYS profile -t cuda -o $${RESULTS_DIR:-.}/v5_stream -f true $(BENCH) stream rows=100M iters=6 queries=groupby_card16 nsys_only=1 tag=nsys_stream && \
	$$NSYS stats --report cuda_gpu_trace --format csv -o $${RESULTS_DIR:-.}/v5_stream $${RESULTS_DIR:-.}/v5_stream.nsys-rep >/dev/null && ls -la $${RESULTS_DIR:-.} | grep v5_stream
