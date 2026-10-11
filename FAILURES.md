# FAILURES

Three things that did not help, why, and what the numbers were. All runs are on one T4, 100M rows
unless stated. Raw rows are in `results.json` (tags `ablation`, `join`), and the dev-run numbers are in
NOTEBOOK.md.

## 1. Stream compaction before the atomics (V1's ballot/popc densification)

**Idea.** At partial selectivity only some lanes hold a surviving row, so the probe/atomic phase runs
with idle lanes. V1 compacts survivors through a per-warp shared ring buffer (`__ballot_sync`
masks, `__popc` prefix offsets) and only drains it 32 entries at a time, so every probe instruction
runs full-width.

**Measured** (`v1_vec` vs `v1_nocompact`, which has the same vector loads but no compaction):

| uniform, sel 10% | 16 | 256 | 4K | 64K | 1M | 16M |
|---|---|---|---|---|---|---|
| no compaction | 10.4 ms | 7.5 | 9.0 | 14.4 | 26.4 | 55.6 |
| compaction | 9.8 | 7.1 | 8.1 | 13.2 | 25.3 | 54.4 |

That is 4–11% at 10% selectivity and nothing at 100% (85.99 vs 85.99 ms at 16 groups), and at
Zipf 1.5 / 100% the two are identical to 0.01 ms.

**Why.** Compaction helps only when the probe phase is the bottleneck *and* lanes are idle. At low
selectivity the probe phase is tiny: 10% of rows reach it, and the kernel is a scan of codes, values
and RLE runs. Densifying the 10% saves divergence in a phase that was already cheap. At high
selectivity lanes are already busy, and the cost is the atomics themselves (contention or random DRAM
sectors), which compaction does not reduce. Compaction also adds a shared-memory round trip per row
and 2 KB of shared memory per warp. That 16 KB/block is part of why the smem-table variants run at
2 blocks/SM instead of 3. What did help at low selectivity was the other half of V1: 16-byte vector
loads (1.4–1.8× over V0 at 1%). The fix that should matter there is not loading unselected rows at all
(`v3_latemat`, measured in RESULTS.md).

## 2. Bloom-filter prefilter on the probe side (V6)

**Idea.** Build a blocked Bloom filter (one 32-bit word per key, 4 bits, 16 bits/key, so ≤ 2 MB and
L2-resident) from the filtered build side. Check it before probing the hash table, so non-matching
fact rows skip the probe. In the partitioned join, apply it while partitioning, so non-matching rows
are never even written out.

**Measured** (100M fact rows; FPR 0.0005–0.19%, so the filter itself works):

| D | sel | non-partitioned | + Bloom | partitioned | partitioned + Bloom |
|---|---|---|---|---|---|
| 1M | 1% | 89.8 ms | 83.9 (−7%) | 101.6 | 136.5 (+34%) |
| 16M | 1% | 105.6 | 103.0 (−2%) | 178.2 | 207.1 (+16%) |
| 1M | 100% | 157.4 | 193.0 (+23%) | 121.4 | 180.1 (+48%) |
| 16M | 100% | 217.4 | 297.9 (+37%) | 258.1 | 417.5 (+62%) |

At best 7%, even at 1% selectivity, where 99% of probes are skipped. Everywhere else it is a loss.

**Why.** The probe that the filter avoids is not where the time goes. The fact key is
dictionary-encoded, so every row first does `key = fact_dict[code]`, a random 8-byte gather from a
D-entry dictionary (8–128 MB, so DRAM-resident for D ≥ 1M). The Bloom check needs the key, so it
happens *after* that gather. The filter can only remove the second random access (the hash-table
probe) and it adds a third (the filter word, an L2 hit). In the partitioned join both passes
(histogram, scatter) must evaluate the filter, so the gather and the filter run twice. At 100%
selectivity nothing is filtered and all of it is overhead. The approach that does pay off attacks the
gather itself. Join the *dictionary* once (D probes instead of F), then scan with a 1-byte
code → category lookup: 13.8 ms vs 83.9 ms at D = 1M / 1% (`v6_dict`).

## 3. Naive radix-partition scatter (V4 pass 1, and the first V6 partitioned join)

**Idea.** The classic two-kernel partition: histogram, then exclusive scan, then scatter, with each
lane writing its row straight to `out[cursor[p]++]` (cursor in shared memory, warp-aggregated with
`__match_any_sync`). Then aggregate each partition in a shared-memory table.

**Measured.** Phase timers (`info.ms_pass1`) showed pass 1 at **217.8 ms for 100M rows into 256
partitions**, about 25× its byte count (2 × 0.6 GB read for the histogram and scatter passes +
1.2 GB written ≈ 2.4 GB ≈ 9 ms at the measured 278 GB/s). The two-pass 16M-group plan spent 333 + 197 ms partitioning before aggregating anything. The
first V6 partitioned join (partition *both* sides into ≤ 2K-key shared-memory build tables, up to 2^13
partitions) lost to the plain join at every point (71 vs 28 ms at D = 1M, dev run).

**Why.** Every row is written as its own 8-byte (key) and 4-byte (value) store to a different
partition, so each store is a partial 32-byte sector. With ~160 blocks × 256 partitions ≈ 40K open
write streams, the partially written lines exceed the 4 MB L2 and are evicted half-filled. On this
ECC-enabled part (nvidia-smi reports ECC counters), a partial-sector write-back most likely becomes a
DRAM read-modify-write. That is inferred, not measured, since there are no counters on Kaggle. Each
store fills at most 8/32 of a sector.

**What recovered part of it, and what did not.** Write-combining: each block counting-sorts a
2048-row round by partition in shared memory, then writes each partition's run contiguously
(`src/gpu/partition.cuh`). That cut pass 1 to 96.7 ms (2.3×) and pass 2 from 197 to 80 ms. It is
still ~10× the byte count, because with 256 partitions a 2048-row round gives runs of ~8 rows (64 B),
still below a 128-byte line. Getting further needs deeper per-block buffering (several rounds per
flush), which is what `cub`'s onesweep radix sort does. That is why plain `v4_sort` beats every
partitioned hash variant above 64K groups (68 vs 142 ms at 1M groups). The partitioned hash approach
pays off only in the 64K–1M band, in its single-pass L2 mode.

## Smaller things that did not pay off (recorded, not one of the three)

- **AoS global-table slots under contention**: 1.8–2.4× slower than SoA at ≤ 16 groups or Zipf 1.5,
  because five atomics on one hot slot serialise on one L2 line. Kept anyway, because V3 removes
  the contention and AoS is 2–3× better once the table is DRAM-resident (RESULTS.md, V0).
- **Bigger shared-memory tables** (1024 slots): fewer overflows, but 1 block/SM. They lose 1.2–1.4× at 16
  groups and under skew, and win only at exactly 256 uniform groups.
- **Single-pass 4096-way partitioning** (`v4_radix_1pass`, unstaged): 394 ms vs 260 ms for two unstaged
  64-way passes (`v4_radix_nostage`) at 1M groups. Open write streams grow with fan-out, so splitting
  the fan-out across passes is cheaper.
