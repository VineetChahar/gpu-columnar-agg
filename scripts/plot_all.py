#!/usr/bin/env python3
"""Regenerate every plot and the summary tables from results.json.

usage: plot_all.py [results.json ...]     (default: merge runs/*/results.json -> results.json)
Later runs override earlier ones for the same measurement key.  Output: plots/*.png, plots/tables.md
"""
import glob, json, math, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "plots")
os.makedirs(OUT, exist_ok=True)

# categorical slots in fixed order (validated reference palette, light mode); baselines are neutral ink
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAIN = ["v0_naive", "v1_vec", "v2_smem_soa", "v3_warpagg", "v4_radix_smem", "v4_radix_l2", "v4_sort"]
COLOR = {v: SLOT[i] for i, v in enumerate(MAIN)}
COLOR.update({"v2_smem_aos512": SLOT[7], "v2_smem_soa512": SLOT[2], "v1_nocompact": SLOT[1], "v3_global": SLOT[3],
              "v4_radix_1pass": SLOT[4], "v3_warpagg256": SLOT[3], "v3_warpagg512": SLOT[3], "v2_smem_soa256": SLOT[2]})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"v4_sort_chunked": "V4 sort, chunked (dashed)", "v0_naive": "V0 naive global", "v1_vec": "V1 vector+compaction", "v2_smem_soa": "V2 smem table (SoA)",
         "v3_warpagg": "V3 warp-aggregated", "v4_radix_smem": "V4 radix→smem", "v4_radix_l2": "V4 radix→L2",
         "v4_sort": "V4 sort+reduce", "cpu": "CPU engine (4 thr)", "cudf": "cuDF"}
CARDS = [16, 256, 4096, 65536, 1048576, 16777216]
CARD_LBL = {16: "16", 256: "256", 4096: "4K", 65536: "64K", 1048576: "1M", 16777216: "16M"}
SKEWS = [0.0, 0.5, 1.0, 1.5]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def load():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "runs", "*", "results.json")))
    rows, seen = [], {}
    for f in files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        run = os.path.basename(os.path.dirname(f))
        for r in data:
            r["_run"] = run
            k = mkey(r)
            if k in seen:
                rows[seen[k]] = r
            else:
                seen[k] = len(rows)
                rows.append(r)
    if not sys.argv[1:]:
        json.dump([{k: v for k, v in r.items()} for r in rows], open(os.path.join(ROOT, "results.json"), "w"), indent=0)
    return rows


def mkey(r):
    e = r.get("exp", "")
    if "pkey" in r:
        return ("pk", r["pkey"])
    if e == "micro":
        return (e, r["name"])
    if e == "timeline":
        return (e, r["_run"], r["query"], r["op"], r["stream"], r["t0_ms"])
    if e == "cudf":
        return (e, r["rows"], r["card"], r["skew"], r["sel"])
    if "stream" in e:
        return (e, r["query"], r["mode"], r.get("streams"), r.get("chunk_rows"))
    if "join" in e:
        return (e, r["variant"], r["D"], r["F"], r["sel"])
    if e == "tpch":
        return (e, r["query"], r["variant"])
    if e == "tpch_external":
        return (e, r["engine"], r["query"])
    return (e, json.dumps(r, sort_keys=True))


R = load()
sweep = [r for r in R if r.get("exp") in ("sweep", "sweep_big") and r.get("correct") and "median_ms" in r]
chunked = [r for r in R if r.get("exp") == "chunked" and r.get("correct") and "median_ms" in r]
mgpu = [r for r in R if r.get("exp") == "mgpu" and r.get("correct") and "median_ms" in r]
cudf = [r for r in R if r.get("exp") == "cudf" and r.get("correct")]


def pick(rows, **kw):
    return [r for r in rows if all((abs(r.get(k, -1) - v) < 1e-9 if isinstance(v, float) else r.get(k) == v) for k, v in kw.items())]


def series(rows, variant, rows_n, sel, skew, comp=False, field="median_ms"):
    xs, ys = [], []
    for c in CARDS:
        m = [r for r in rows if r["variant"] == variant and r["rows"] == rows_n and r["sel"] == sel
             and abs(r["skew"] - skew) < 1e-9 and r["card"] == c and r.get("composite", False) == comp]
        if m:
            xs.append(c); ys.append(m[-1][field])
    return xs, ys


def cpu_series(rows, rows_n, sel, skew, comp=False):
    xs, ys = [], []
    for c in CARDS:
        m = [r for r in rows if r["rows"] == rows_n and r["sel"] == sel and abs(r["skew"] - skew) < 1e-9
             and r["card"] == c and r.get("composite", False) == comp and r.get("cpu_ms")]
        if m:
            xs.append(c); ys.append(m[-1]["cpu_ms"])
    return xs, ys


def fmt_card_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(CARDS)
    ax.set_xticklabels([CARD_LBL[c] for c in CARDS])
    ax.minorticks_off()


def crossover(rows_n, sel, field="median_ms", fname=None, comp=False, variants=MAIN, title_extra=""):
    have = [r for r in sweep if r["rows"] == rows_n and r["sel"] == sel and r.get("composite", False) == comp]
    if not have:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, sk in zip(axes, SKEWS):
        for v in variants:
            xs, ys = series(sweep, v, rows_n, sel, sk, comp, field)
            if xs:
                ax.plot(xs, ys, color=COLOR[v], lw=2, marker="o", ms=4, label=LABEL.get(v, v))
        if field == "median_ms":
            xs, ys = cpu_series(sweep, rows_n, sel, sk, comp)
            if xs:
                ax.plot(xs, ys, color=INK2, lw=1.5, ls="--", label=LABEL["cpu"])
            cx = [(r["card"], r["median_ms"]) for r in cudf if r["rows"] == rows_n and r["sel"] == sel and abs(r["skew"] - sk) < 1e-9]
            if cx and not comp:
                cx.sort()
                ax.plot([c for c, _ in cx], [t for _, t in cx], color=INK, lw=1.2, ls=":", marker="s", ms=3, label=LABEL["cudf"])
            ax.set_yscale("log")
        fmt_card_axis(ax)
        ax.set_title(f"skew {'uniform' if sk == 0 else f'Zipf α={sk}'}", color=INK, fontsize=9)
        ax.set_xlabel("group cardinality (dictionary size)")
    axes[0].set_ylabel({"median_ms": "time per query (ms, log)", "bw_pct": "% of measured DRAM read BW"}[field])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"{rows_n / 1e6:.0f}M rows, selectivity {sel}%{title_extra}", y=1.13, color=INK, fontsize=10)
    fname = fname or f"crossover_{field}_{int(rows_n / 1e6)}M_sel{sel}{'_comp' if comp else ''}.png"
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    return fname


def winners(rows_n, sel):
    """table: best variant per (card, skew) + its speedup over the CPU engine"""
    out = ["| skew \\ card | " + " | ".join(CARD_LBL[c] for c in CARDS) + " |", "|---" * (len(CARDS) + 1) + "|"]
    for sk in SKEWS:
        cells = []
        for c in CARDS:
            m = [r for r in sweep if r["rows"] == rows_n and r["sel"] == sel and abs(r["skew"] - sk) < 1e-9
                 and r["card"] == c and not r.get("composite") and r["variant"] in MAIN]
            if not m:
                cells.append("–"); continue
            b = min(m, key=lambda r: r["median_ms"])
            cells.append(f"{b['variant'].replace('_', ' ')} {b['median_ms']:.1f} ms ({b.get('speedup_vs_cpu', float('nan')):.0f}×)")
        out.append(f"| {'uniform' if sk == 0 else sk} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table(rows_n, sel, skew, variants, field="median_ms", fmt="{:.2f}", comp=False):
    out = ["| variant | " + " | ".join(CARD_LBL[c] for c in CARDS) + " |", "|---" * (len(CARDS) + 1) + "|"]
    for v in variants:
        cells = []
        for c in CARDS:
            m = [r for r in sweep if r["variant"] == v and r["rows"] == rows_n and r["sel"] == sel
                 and abs(r["skew"] - skew) < 1e-9 and r["card"] == c and r.get("composite", False) == comp]
            cells.append(fmt.format(m[-1][field]) if m and field in m[-1] else "–")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    xs, ys = cpu_series(sweep, rows_n, sel, skew, comp)
    if xs and field == "median_ms":
        d = dict(zip(xs, ys))
        out.append("| cpu engine | " + " | ".join(fmt.format(d[c]) if c in d else "–" for c in CARDS) + " |")
    return "\n".join(out)


md = ["# Generated tables (scripts/plot_all.py)\n"]
for rows_n in sorted({r["rows"] for r in sweep}):
    for sel in sorted({r["sel"] for r in sweep if r["rows"] == rows_n}):
        crossover(rows_n, sel)
        crossover(rows_n, sel, "bw_pct")
        md.append(f"\n## Winners, {rows_n / 1e6:.0f}M rows, sel {sel}%\n\n" + winners(rows_n, sel))
        for sk in SKEWS:
            md.append(f"\n### time (ms), {rows_n / 1e6:.0f}M rows, sel {sel}%, skew {sk}\n\n" +
                      table(rows_n, sel, sk, MAIN + ["v2_smem_aos512", "v2_smem_soa512", "v1_nocompact", "v3_global", "v4_radix_1pass"]))
    if any(r.get("composite") for r in sweep if r["rows"] == rows_n):
        crossover(rows_n, 100, comp=True, title_extra=" — composite (2×u16) keys")


# ---------------- selectivity & scale ----------------
def sel_plot(rows_n, card, skew=0.0):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for v in MAIN:
        pts = sorted((r["sel"], r["median_ms"]) for r in sweep if r["variant"] == v and r["rows"] == rows_n and r["card"] == card
                     and abs(r["skew"] - skew) < 1e-9 and not r.get("composite"))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[v], lw=2, marker="o", ms=4, label=LABEL[v])
    pts = sorted({(r["sel"], r["cpu_ms"]) for r in sweep if r["rows"] == rows_n and r["card"] == card and abs(r["skew"] - skew) < 1e-9 and not r.get("composite")})
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=INK2, ls="--", lw=1.5, label=LABEL["cpu"])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 50, 100]); ax.set_xticklabels(["1%", "10%", "50%", "100%"])
    ax.set_xlabel("predicate selectivity"); ax.set_ylabel("time (ms, log)")
    ax.set_title(f"{rows_n / 1e6:.0f}M rows, card {CARD_LBL[card]}, {'uniform' if skew == 0 else f'Zipf {skew}'}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fn = f"selectivity_{int(rows_n / 1e6)}M_card{card}_skew{skew}.png"
    fig.savefig(os.path.join(OUT, fn)); plt.close(fig)


for rows_n in sorted({r["rows"] for r in sweep}):
    for card in (256, 65536, 16777216):
        sel_plot(rows_n, card)

if len({r["rows"] for r in sweep}) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, card in zip(axes, (256, 65536, 16777216)):
        for v in MAIN:
            pts = sorted((r["rows"], r["rows_per_s"] / 1e9) for r in sweep if r["variant"] == v and r["card"] == card and r["sel"] == 100
                         and r["skew"] == 0 and not r.get("composite"))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[v], lw=2, marker="o", ms=4, label=LABEL[v])
        # chunked sort: same hue as V4 sort (same algorithm), dashed - it exists where v4_sort ran out of memory
        pts = sorted((r["rows"], r["rows_per_s"] / 1e9) for r in chunked if r["variant"] == "v4_sort_chunked" and r["card"] == card
                     and r["sel"] == 100 and r["skew"] == 0)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR["v4_sort"], lw=2, ls="--", marker="s", ms=4,
                    label=LABEL["v4_sort_chunked"])
        ax.set_xscale("log"); ax.set_title(f"card {CARD_LBL[card]}, uniform, sel 100%", fontsize=9)
        ax.set_xlabel("rows")
    axes[0].set_ylabel("throughput (G rows/s)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.savefig(os.path.join(OUT, "scale.png")); plt.close(fig)

# ---------------- micro ----------------
micro = {r["name"]: r for r in R if r.get("exp") == "micro"}
if micro:
    md.append("\n## Microbenchmarks\n\n| name | value | unit |\n|---|---|---|")
    for k, r in sorted(micro.items()):
        md.append(f"| {k} | {r['value']:.3f} | {r['unit']} |")

# ---------------- roofline ----------------
# x = aggregate updates (selected rows) per byte of *analytic* DRAM traffic of that variant,
# y = updates/s.  The diagonal is the measured DRAM read bandwidth: a point on it is bandwidth-bound,
# a point far below it is bound by something else (atomic latency/serialisation, host round trips).
L2_BYTES = 4 << 20
def dram_traffic(r):
    n, sel_rows = r["rows"], r.get("selected_est", r["rows"])
    w = {"u8": 1, "u16": 2, "u32": 4}.get(r.get("key_kind"), 4)
    card, cap = r["card"], r.get("table_cap", 2 * r["card"])
    t = r["bytes_in"]                                            # compulsory scan
    dict_bytes = card * 8
    gather = sel_rows * 32 if dict_bytes > L2_BYTES else dict_bytes  # random 32 B sector per decode when dict > L2
    table = cap * 64
    v = r["variant"]
    if v == "v4_sort":
        bits = max(1, int(math.log2(card)))
        passes = math.ceil(bits / 8)
        t += sel_rows * 8 + passes * sel_rows * 16 + sel_rows * 8   # compact write, onesweep passes (r+w 8 B), reduce read
        t += min(card, sel_rows) * (8 + 32)                          # unique codes + aggregates out
        t += min(card, sel_rows) * 32 if dict_bytes > L2_BYTES else dict_bytes  # decode uniques only
        return t
    t += gather + table + cap * 8                                   # clear the table, read keys at extract
    if table > L2_BYTES:
        # V0-V3: a 256-slot shared table cannot hold >>256 keys, so (uniform keys) ~every row updates a
        # random DRAM-resident slot.  V4: each partition's slice of the table is L2/smem-resident, so each
        # slot is fetched and written back about once.
        groups_touch = min(sel_rows, card) if v.startswith("v4_radix") else sel_rows
        t += groups_touch * 128                                     # read+write one 64 B slot per update
    info = r.get("info") or {}
    passes = info.get("passes", 0)
    if v.startswith("v4_radix") and passes:
        t += r["bytes_in"] + gather                                 # histogram pass re-reads + re-decodes
        t += passes * sel_rows * 12 * 2                             # partition write + read back
    return t

if micro and sweep:
    bw = micro.get("dram_read_gbs_best", {}).get("value")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [10 ** (i / 20) for i in range(-80, 1)]
    ax.plot(xs, [bw * 1e9 * x for x in xs], color=INK2, lw=1.2)
    ax.text(2e-3, bw * 1e9 * 2e-3 * 1.5, f"DRAM read {bw:.0f} GB/s (measured)", color=INK2, fontsize=7, rotation=31)
    for name, lbl in [("gatomic_u64_random_ws1048576KB_gops", "random global atomics, DRAM-resident (÷5 per update)")]:
        if name in micro:
            y = micro[name]["value"] * 1e9 / 5
            ax.axhline(y, color=INK2, lw=0.8, ls=":")
            ax.text(xs[0] * 1.3, y * 1.12, lbl, color=INK2, fontsize=7)
    ref_rows = [r for r in sweep if r["sel"] == 100 and r["skew"] == 0 and not r.get("composite")]
    big = max((r["rows"] for r in ref_rows), default=0)
    for v in MAIN:
        for c, mk in ((4096, "s"), (1048576, "o"), (16777216, "^")):
            m = [r for r in ref_rows if r["variant"] == v and r["card"] == c and r["rows"] == big]
            if not m:
                continue
            r = m[-1]
            u = r.get("selected_est", r["rows"])
            ax.scatter([u / dram_traffic(r)], [u / (r["median_ms"] / 1e3)], color=COLOR[v], marker=mk, s=46, zorder=3,
                       edgecolor="white", linewidth=1)
    for v in MAIN:
        ax.scatter([], [], color=COLOR[v], label=LABEL[v])
    for lbl, mk in (("card 4K", "s"), ("card 1M", "o"), ("card 16M", "^")):
        ax.scatter([], [], color=INK2, marker=mk, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("updates per byte of analytic DRAM traffic"); ax.set_ylabel("aggregate updates / s")
    ax.set_title(f"Roofline: {big / 1e6:.0f}M rows, uniform, sel 100%", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(OUT, "roofline.png")); plt.close(fig)
    md.append("\n## Roofline points (100M, uniform, sel 100%)\n\n| variant | card | analytic GB | GB/s achieved | % of DRAM BW |\n|---|---|---|---|---|")
    for v in MAIN:
        for c in (4096, 1048576, 16777216):
            m = [r for r in ref_rows if r["variant"] == v and r["card"] == c and r["rows"] == big]
            if m:
                r = m[-1]; tb = dram_traffic(r); g = tb / (r["median_ms"] / 1e3) / 1e9
                md.append(f"| {v} | {CARD_LBL[c]} | {tb / 1e9:.2f} | {g:.0f} | {100 * g / bw:.0f}% |")

# ---------------- V5 timeline + PCIe wall ----------------
tl = [r for r in R if r.get("exp") == "timeline" and r.get("_run") == max((x["_run"] for x in R if x.get("exp") == "timeline"), default="")]
if tl:
    for q in sorted({r["query"] for r in tl}):
        ev = [r for r in tl if r["query"] == q]
        last_run = max(r["_run"] for r in ev)
        ev = [r for r in ev if r["_run"] == last_run]
        fig, ax = plt.subplots(figsize=(10, 2.4))
        col = {"h2d": SLOT[0], "kernel": SLOT[1], "extract": SLOT[2], "d2h": SLOT[6], "init": INK2}
        for r in ev:
            ax.barh(r["stream"], r["t1_ms"] - r["t0_ms"], left=r["t0_ms"], color=col[r["op"]], height=0.6, edgecolor="white", linewidth=0.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=op) for op, c in col.items()], ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, 1.35), fontsize=7)
        ax.set_yticks(sorted({r["stream"] for r in ev})); ax.set_yticklabels([f"stream {s}" for s in sorted({r["stream"] for r in ev})])
        ax.set_xlabel("ms since pipeline start"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_title(f"V5 event timeline, {q}", fontsize=9, pad=24)
        fig.savefig(os.path.join(OUT, f"timeline_{q}.png")); plt.close(fig)
st = [r for r in R if r.get("exp") == "stream" and "median_ms" in r]
if st:
    md.append("\n## V5 streaming / PCIe wall\n\n| query | mode | streams | chunk rows | ms | eff. PCIe GB/s | CPU engine ms | correct |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(st, key=lambda r: (r["query"], r["mode"], r.get("streams") or 0, r.get("chunk_rows") or 0)):
        md.append(f"| {r['query']} | {r['mode']} | {r.get('streams', '')} | {r.get('chunk_rows', '')} | {r['median_ms']:.1f} | "
                  f"{r.get('eff_pcie_gbs', float('nan')):.1f} | {r.get('cpu_ms', float('nan')):.1f} | {r.get('correct', '')} |")
    qs = sorted({r["query"] for r in st})
    fig, axes = plt.subplots(1, len(qs), figsize=(4.0 * len(qs) + 2, 3.4), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    axes = axes if len(qs) > 1 else [axes]
    order = ["gpu_resident_kernel_only", "gpu_stream_pinned_overlap", "gpu_stream_pinned_serial", "gpu_stream_pageable_overlap",
             "gpu_stream_pageable_serial", "cpu_engine_host_resident"]
    for ax, q in zip(axes, qs):
        vals = []
        for m in order:
            c = [r for r in st if r["query"] == q and r["mode"] == m and (r.get("streams") in (None, 0, 1, 3))]
            if c:
                vals.append((m, c[-1]["median_ms"], c[-1].get("extra_ms", 0) if "pinned" in m else 0))
        ys = range(len(vals))
        base = min(v[1] for v in vals) * 0.5
        ax.barh(list(ys), [v[1] - base for v in vals], left=base,
                color=[SLOT[0] if "gpu" in v[0] else INK2 for v in vals], height=0.6)
        ax.barh(list(ys), [v[2] for v in vals], left=[v[1] for v in vals], color=SLOT[1], height=0.6,
                label="+ cudaHostRegister (one-shot query)")
        pc = [r for r in st if r["query"] == q and "pcie_bytes" in r]
        h2d = micro.get("pcie_h2d_pinned_gbs", {}).get("value") if micro else None
        if pc and h2d:
            floor = pc[0]["pcie_bytes"] / (h2d * 1e6)
            ax.axvline(floor, color=SLOT[7], lw=1.2, ls="--", label=f"PCIe floor ({h2d:.1f} GB/s pinned)")
        ax.set_yticks(list(ys)); ax.set_yticklabels([v[0].replace("gpu_", "").replace("_", " ") for v in vals], fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("ms (log)"); ax.set_title(q.replace("_", " "), fontsize=9)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(left=base, right=max(v[1] + v[2] for v in vals) * 2.2)
        for yy, v in zip(ys, vals):
            ax.text((v[1] + v[2]) * 1.06, yy, f"{v[1]:.0f}" + (f" (+{v[2]:.0f})" if v[2] else ""), va="center", fontsize=7, color=INK)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.savefig(os.path.join(OUT, "pcie_wall.png")); plt.close(fig)

# ---------------- V6 join ----------------
jn = [r for r in R if r.get("exp") == "join" and r.get("correct") and "median_ms" in r]
if jn:
    JV = ["v6_np", "v6_np_bloom", "v6_part", "v6_part_bloom", "v6_dict"]
    Ds = sorted({r["D"] for r in jn})
    fig, axes = plt.subplots(1, len(Ds), figsize=(4.2 * len(Ds), 3.4), sharey=True)
    axes = axes if len(Ds) > 1 else [axes]
    for ax, D in zip(axes, Ds):
        for i, v in enumerate(JV):
            pts = sorted((r["sel"], r["median_ms"]) for r in jn if r["variant"] == v and r["D"] == D)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOT[i], lw=2, marker="o", ms=4, label=v)
        pts = sorted({(r["sel"], r["cpu_ms"]) for r in jn if r["D"] == D})
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=INK2, ls="--", lw=1.5, label="CPU engine (4 thr)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 50, 100]); ax.set_xticklabels(["1%", "10%", "50%", "100%"])
        ax.set_title(f"build side D = {D:,}", fontsize=9); ax.set_xlabel("join selectivity")
    axes[0].set_ylabel("time (ms, log)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(os.path.join(OUT, "join.png")); plt.close(fig)
    md.append("\n## V6 join (ms; probed = rows that reached the hash table)\n\n| D | sel | variant | ms | probed | bloom FPR | CPU ms | speedup |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(jn, key=lambda r: (r["D"], r["sel"], r["variant"])):
        md.append(f"| {r['D']:,} | {r['sel']} | {r['variant']} | {r['median_ms']:.2f} | {r.get('probed', 0):,.0f} | "
                  f"{r.get('bloom_fpr', float('nan')):.4f} | {r['cpu_ms']:.1f} | {r.get('speedup_vs_cpu', float('nan')):.1f} |")

# ---------------- follow-up: chunked sort at 500M, cap=est at 500M, multi-GPU ----------------
big = [r for r in sweep if r["rows"] == 500_000_000]
if chunked:
    md.append("\n## Chunked sort at 500M rows vs the best variant that fit before\n\n"
              "| groups | skew | sel | best of main sweep | ms | v4_sort_chunked ms | chunks | speedup |\n|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in chunked if r["rows"] == 500_000_000), key=lambda r: (r["card"], r["skew"], r["sel"])):
        b = [x for x in big if x["card"] == r["card"] and x["skew"] == r["skew"] and x["sel"] == r["sel"] and not x.get("composite")]
        if not b:
            continue
        b = min(b, key=lambda x: x["median_ms"])
        ch = (r.get("info") or {}).get("chunks", float("nan"))
        md.append(f"| {CARD_LBL[r['card']]} | {r['skew']} | {r['sel']}% | {b['variant']} | {b['median_ms']:.1f} | "
                  f"{r['median_ms']:.1f} | {ch:.0f} | {b['median_ms'] / r['median_ms']:.2f}x |")
    md.append("\n### Chunking overhead at 100M rows (same session: v4_sort vs chunked, auto plan and forced 16M-row chunks)\n\n"
              "| groups | skew | sel | v4_sort | chunked (auto) | chunked (16M-row chunks) |\n|---|---|---|---|---|---|")
    c16 = [r for r in R if r.get("exp") == "chunked16M" and r.get("correct") and "median_ms" in r]
    for r in sorted((r for r in chunked if r["rows"] == 100_000_000 and r["variant"] == "v4_sort"), key=lambda r: (r["card"], r["skew"], r["sel"])):
        a = [x for x in chunked if x["rows"] == r["rows"] and x["variant"] == "v4_sort_chunked" and x["card"] == r["card"]
             and x["skew"] == r["skew"] and x["sel"] == r["sel"]]
        f = [x for x in c16 if x["card"] == r["card"] and x["skew"] == r["skew"] and x["sel"] == r["sel"]]
        md.append(f"| {CARD_LBL[r['card']]} | {r['skew']} | {r['sel']}% | {r['median_ms']:.1f} | "
                  f"{a[0]['median_ms']:.1f} | " + (f"{f[0]['median_ms']:.1f} ({f[0]['info'].get('chunks', 0):.0f} chunks)" if f else "-") + " |"
                  if a else f"| {CARD_LBL[r['card']]} | {r['skew']} | {r['sel']}% | {r['median_ms']:.1f} | - | - |")

capbig = [r for r in R if r.get("exp") == "capest" and r.get("rows") == 500_000_000 and r.get("correct") and "median_ms" in r]
if capbig:
    md.append("\n## cap=est at 500M rows (hash variants; best per point)\n\n"
              "| groups | skew | sel | cap=dict best | ms | cap=est best | ms | table cap dict -> est |\n|---|---|---|---|---|---|---|---|")
    for key in sorted({(r["card"], r["skew"], r["sel"]) for r in capbig}):
        e = min((r for r in capbig if (r["card"], r["skew"], r["sel"]) == key), key=lambda r: r["median_ms"])
        d = [x for x in big if (x["card"], x["skew"], x["sel"]) == key and not x.get("composite")]
        if not d:
            continue
        d = min(d, key=lambda r: r["median_ms"])
        md.append(f"| {CARD_LBL[key[0]]} | {key[1]} | {key[2]}% | {d['variant']} | {d['median_ms']:.1f} | {e['variant']} | "
                  f"{e['median_ms']:.1f} | {d['table_cap']:,.0f} -> {e['table_cap']:,.0f} |")

if mgpu:
    md.append("\n## Multi-GPU (2 x T4, rows split, partials merged on GPU 0)\n\n"
              "| rows | groups | skew | sel | variant | 1 GPU ms | 2 GPU ms | speedup | slowest GPU | transfer | merge | partial groups |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    one = {(r["rows"], r["card"], r["skew"], r["sel"], r["variant"]): r for r in mgpu if r["gpus"] == 1}
    for r in sorted((r for r in mgpu if r["gpus"] == 2), key=lambda r: (r["rows"], r["card"], r["skew"], r["sel"], r["median_ms"])):
        o = one.get((r["rows"], r["card"], r["skew"], r["sel"], r["variant"]))
        md.append(f"| {r['rows'] / 1e6:.0f}M | {CARD_LBL.get(r['card'], r['card'])} | {r['skew']} | {r['sel']}% | {r['variant']} | "
                  + (f"{o['median_ms']:.1f}" if o else "OOM/skip") + f" | {r['median_ms']:.1f} | "
                  + (f"{o['median_ms'] / r['median_ms']:.2f}x" if o else "-")
                  + f" | {r['ms_slowest_gpu']:.1f} | {r['ms_transfer']:.1f} | {r['ms_merge']:.1f} | {r.get('partial_groups', 0):,.0f} |")
    # plot: best 1-GPU vs best 2-GPU per cardinality at 500M / sel 100%, and the 2-GPU combine cost
    rows_big = max(r["rows"] for r in mgpu)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, skew in zip(axes, (0.0, 1.5)):
        for G, style in ((1, dict(color=INK2, ls="--", marker="o")), (2, dict(color=SLOT[0], ls="-", marker="o"))):
            xs, ys = [], []
            for c in CARDS:
                m = [r for r in mgpu if r["gpus"] == G and r["rows"] == rows_big and r["card"] == c and r["skew"] == skew and r["sel"] == 100]
                if m:
                    xs.append(c); ys.append(min(r["median_ms"] for r in m))
            if xs:
                ax.plot(xs, ys, lw=2, ms=4, label=f"best variant, {G} GPU" + ("s" if G > 1 else ""), **style)
        xs, ys = [], []
        for c in CARDS:
            m = [r for r in mgpu if r["gpus"] == 2 and r["rows"] == rows_big and r["card"] == c and r["skew"] == skew and r["sel"] == 100]
            if m:
                b = min(m, key=lambda r: r["median_ms"]); xs.append(c); ys.append(b["ms_transfer"] + b["ms_merge"])
        if xs:
            ax.plot(xs, ys, color=SLOT[1], lw=2, ls=":", marker="^", ms=4, label="2 GPUs: transfer + merge part")
        ax.set_xscale("log"); ax.set_yscale("log"); fmt_card_axis(ax)
        ax.set_title(f"{rows_big / 1e6:.0f}M rows, {'uniform' if skew == 0 else f'Zipf {skew}'}, sel 100%", fontsize=9)
        ax.set_xlabel("groups")
    axes[0].set_ylabel("time (ms, log)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(os.path.join(OUT, "mgpu.png")); plt.close(fig)

# ---------------- TPC-H ----------------
tp = [r for r in R if r.get("exp") == "tpch" and r.get("correct")]
tx = [r for r in R if r.get("exp") == "tpch_external"]
if tp:
    md.append("\n## TPC-H SF-10 lineitem\n\n| query | engine | ms | % DRAM BW | speedup vs CPU engine |\n|---|---|---|---|---|")
    for r in sorted(tp, key=lambda r: (r["query"], r["median_ms"])):
        md.append(f"| {r['query']} | {r['variant']} | {r['median_ms']:.2f} | {r['bw_pct']:.0f}% | {r['speedup_vs_cpu']:.1f} |")
    for q in sorted({r["query"] for r in tp}):
        md.append(f"| {q} | CPU engine (4 thr) | {[r for r in tp if r['query'] == q][0]['cpu_ms']:.1f} | | 1.0 |")
    for r in tx:
        md.append(f"| {r['query']} | {r['engine']} (external) | {r['median_ms']:.2f} | | |")

open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
print("wrote", len(os.listdir(OUT)), "files to", OUT)

# ---------------- nsys trace of the V5 pipeline ----------------
def nsys_timeline():
    import csv
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "v5_stream_cuda_gpu_trace.csv")))
    if not files:
        return None
    rows = list(csv.DictReader(open(files[-1])))
    ops = []
    for r in rows:
        n = r["Name"]
        kind = "h2d" if "Host-to-Device" in n else "d2h" if "Device-to-Host" in n else "kernel" if "k_agg" in n else \
               "extract" if "extract" in n else None
        if kind:
            s = int(r["Start (ns)"]) / 1e6
            ops.append((s, s + int(r["Duration (ns)"]) / 1e6, kind, r["Strm"], float(r["Bytes (MB)"] or 0)))
    ops.sort()
    # split into episodes (pipeline runs) separated by idle gaps > 2 ms; keep the ones using >= 3 streams
    eps, cur, end = [], [], -1e18
    for o in ops:
        if cur and o[0] - end > 2.0:
            eps.append(cur); cur = []
        cur.append(o); end = max(end, o[1])
    if cur:
        eps.append(cur)
    def ovfrac(ep, a_kind, b_kind):
        A = [(a, b) for a, b, k, *_ in ep if k == a_kind]
        Bv = [(a, b) for a, b, k, *_ in ep if k == b_kind]
        tot = sum(b - a for a, b in A)
        ov = 0.0
        for a, b in A:
            for c, d in Bv:
                ov += max(0.0, min(b, d) - max(a, c))
        return ov / tot if tot else 0.0
    cand = [e for e in eps if len({o[3] for o in e if o[2] == "h2d"}) >= 2]
    if not cand:
        return None
    # the pinned + overlapped pipeline is the episode with the highest H2D throughput
    ep = max(cand, key=lambda e: sum(o[4] for o in e if o[2] == "h2d") / (max(o[1] for o in e) - e[0][0]))
    k_hidden = ovfrac(ep, "kernel", "h2d")
    c_hidden = ovfrac(ep, "h2d", "kernel")
    t0 = ep[0][0]
    win = [o for o in ep if o[0] - t0 < 30.0]
    streams = sorted({o[3] for o in win}, key=int)
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 2.6))
    col = {"h2d": SLOT[0], "kernel": SLOT[1], "extract": SLOT[2], "d2h": SLOT[6]}
    for a, b, k, s, _ in win:
        ax.barh(streams.index(s), b - a, left=a - t0, color=col[k], height=0.6, edgecolor="white", linewidth=0.4)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in col.items()], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.34), fontsize=7)
    ax.set_yticks(range(len(streams))); ax.set_yticklabels([f"stream {i}" for i in range(len(streams))])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("ms since pipeline start (nsys CUDA trace, first 30 ms)")
    ax.set_title(f"V5 under nsys: {100 * k_hidden:.0f}% of kernel time runs under an H2D copy; "
                 f"copies busy {100 * c_hidden:.0f}% concurrent with kernels", fontsize=9, pad=22)
    fig.savefig(os.path.join(OUT, "nsys_timeline.png")); plt.close(fig)
    return k_hidden, c_hidden


ovl = nsys_timeline()
if ovl is not None:
    with open(os.path.join(OUT, "tables.md"), "a") as f:
        f.write(f"\n## nsys (3-stream V5 pipeline)\n\nkernel time hidden under H2D copies: {100 * ovl[0]:.1f}%; "
                f"H2D time with a kernel running concurrently: {100 * ovl[1]:.1f}%\n")
    print(f"nsys: kernels hidden {100 * ovl[0]:.1f}%, copies overlapped {100 * ovl[1]:.1f}%")
