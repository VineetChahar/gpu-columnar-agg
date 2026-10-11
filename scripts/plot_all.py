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
LABEL = {"v0_naive": "V0 naive global", "v1_vec": "V1 vector+compaction", "v2_smem_soa": "V2 smem table (SoA)",
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

# ---------------- roofline (atomic-update roofline) ----------------
# x: aggregate updates per byte of compulsory input traffic; y: achieved updates/s.
# roofs: measured DRAM read BW (diagonal) and measured atomic throughputs (flat).
if micro and sweep:
    bw = micro.get("dram_read_gbs_best", {}).get("value")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [10 ** (i / 20) for i in range(-60, 21)]
    if bw:
        ax.plot(xs, [bw * 1e9 * x for x in xs], color=INK2, lw=1.2)
        ax.text(0.02, bw * 1e9 * 0.02 * 1.4, f"DRAM read {bw:.0f} GB/s", color=INK2, fontsize=7, rotation=33)
    for name, lbl in [("gatomic_u64_random_ws128KB_gops", "global atomics, L2-resident"),
                      ("gatomic_u64_random_ws1048576KB_gops", "global atomics, DRAM working set"),
                      ("gatomic_u64_hot_single_address_gops", "single hot address")]:
        if name in micro:
            y = micro[name]["value"] * 1e9 / 5  # 5 atomics per aggregate update
            ax.axhline(y, color=INK2, lw=0.8, ls=":")
            ax.text(xs[0] * 1.2, y * 1.08, lbl + " (/5 atomics)", color=INK2, fontsize=7)
    ref_rows = [r for r in sweep if r["sel"] == 100 and r["skew"] == 0 and not r.get("composite")]
    big = max((r["rows"] for r in ref_rows), default=0)
    for v in MAIN:
        for c, mk in ((65536, "o"), (16777216, "^")):
            m = [r for r in ref_rows if r["variant"] == v and r["card"] == c and r["rows"] == big]
            if not m:
                continue
            r = m[-1]
            sel_rows = r.get("selected_est", r["rows"])
            x = sel_rows / r["bytes_in"]
            y = sel_rows / (r["median_ms"] / 1e3)
            ax.scatter([x], [y], color=COLOR[v], marker=mk, s=40, zorder=3, edgecolor="white", linewidth=1)
    for v in MAIN:
        ax.scatter([], [], color=COLOR[v], label=LABEL[v])
    ax.scatter([], [], color=INK2, marker="o", label="card 64K"); ax.scatter([], [], color=INK2, marker="^", label="card 16M")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("aggregate updates per input byte (analytic)"); ax.set_ylabel("aggregate updates / s")
    ax.set_title(f"Roofline: {big / 1e6:.0f}M rows, uniform, sel 100%", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(OUT, "roofline.png")); plt.close(fig)

# ---------------- V5 timeline + PCIe wall ----------------
tl = [r for r in R if r.get("exp") == "timeline"]
if tl:
    for q in sorted({r["query"] for r in tl}):
        ev = [r for r in tl if r["query"] == q]
        last_run = max(r["_run"] for r in ev)
        ev = [r for r in ev if r["_run"] == last_run]
        fig, ax = plt.subplots(figsize=(10, 2.4))
        col = {"h2d": SLOT[0], "kernel": SLOT[1], "extract": SLOT[2], "d2h": SLOT[6], "init": INK2}
        for r in ev:
            ax.barh(r["stream"], r["t1_ms"] - r["t0_ms"], left=r["t0_ms"], color=col[r["op"]], height=0.6, edgecolor="white", linewidth=0.5)
        for op, c in col.items():
            ax.barh([], [], color=c, label=op)
        ax.set_yticks(sorted({r["stream"] for r in ev})); ax.set_yticklabels([f"stream {s}" for s in sorted({r["stream"] for r in ev})])
        ax.set_xlabel("ms since pipeline start"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35), fontsize=7)
        ax.set_title(f"V5 event timeline, {q}", fontsize=9, pad=24)
        fig.savefig(os.path.join(OUT, f"timeline_{q}.png")); plt.close(fig)
st = [r for r in R if "stream" in str(r.get("exp", "")) and r.get("exp") != "timeline" and "median_ms" in r]
if st:
    md.append("\n## V5 streaming / PCIe wall\n\n| query | mode | streams | chunk rows | ms | eff. PCIe GB/s | CPU engine ms | correct |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(st, key=lambda r: (r["query"], r["mode"], r.get("streams") or 0, r.get("chunk_rows") or 0)):
        md.append(f"| {r['query']} | {r['mode']} | {r.get('streams', '')} | {r.get('chunk_rows', '')} | {r['median_ms']:.1f} | "
                  f"{r.get('eff_pcie_gbs', float('nan')):.1f} | {r.get('cpu_ms', float('nan')):.1f} | {r.get('correct', '')} |")
    qs = sorted({r["query"] for r in st})
    fig, axes = plt.subplots(1, len(qs), figsize=(4.2 * len(qs), 3.6), sharey=False)
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
        ax.barh(list(ys), [v[1] for v in vals], color=[SLOT[0] if "gpu" in v[0] else INK2 for v in vals], height=0.6)
        ax.barh(list(ys), [v[2] for v in vals], left=[v[1] for v in vals], color=SLOT[1], height=0.6,
                label="+ cudaHostRegister (one-shot query)")
        pc = [r for r in st if r["query"] == q and "pcie_bytes" in r]
        h2d = micro.get("pcie_h2d_pinned_gbs", {}).get("value") if micro else None
        if pc and h2d:
            floor = pc[0]["pcie_bytes"] / (h2d * 1e6)
            ax.axvline(floor, color=SLOT[7], lw=1.2, ls="--", label=f"PCIe floor ({h2d:.1f} GB/s pinned)")
        ax.set_yticks(list(ys)); ax.set_yticklabels([v[0].replace("gpu_", "").replace("_", " ") for v in vals], fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("ms"); ax.set_title(q, fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
    fig.savefig(os.path.join(OUT, "pcie_wall.png")); plt.close(fig)

# ---------------- V6 join ----------------
jn = [r for r in R if "join" in str(r.get("exp", "")) and r.get("correct") and "median_ms" in r]
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

# ---------------- TPC-H ----------------
tp = [r for r in R if r.get("exp") == "tpch" and r.get("correct")]
tx = [r for r in R if r.get("exp") == "tpch_external"]
if tp:
    md.append("\n## TPC-H SF-10 lineitem\n\n| query | engine | ms | % DRAM BW | speedup vs CPU engine |\n|---|---|---|---|---|")
    for r in sorted(tp, key=lambda r: (r["query"], r["median_ms"])):
        md.append(f"| {r['query']} | {r['variant']} | {r['median_ms']:.2f} | {r['bw_pct']:.0f}% | {r['speedup_vs_cpu']:.1f} |")
    for q in sorted({r["query"] for r in tp}):
        md.append(f"| {q} | CPU engine (4 thr) | {tp[0]['cpu_ms'] if tp[0]['query'] == q else [r for r in tp if r['query'] == q][0]['cpu_ms']:.1f} | | 1.0 |")
    for r in tx:
        md.append(f"| {r['query']} | {r['engine']} (external) | {r['median_ms']:.2f} | | |")

open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
print("wrote", len(os.listdir(OUT)), "files to", OUT)
