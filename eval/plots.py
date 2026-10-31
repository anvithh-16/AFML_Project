"""
Turn eval results into the report's figures + tables.

    python eval/plots.py --results_dir results --stage instruct --out_dir results/figures

Reads results/variant-<X>_<stage>.csv (from run_eval.py) and results/efficiency.csv
(from efficiency.py). Whatever variants exist get plotted; missing ones are skipped,
so you can run this as soon as Variant A is done and re-run when B and C land.

Figures (PNG, 200 dpi):
  fig1_needle.png          T1 accuracy vs distance  <- the headline figure
  fig2_all_tests.png       T1-T4 accuracy vs distance, one panel each
  fig3_stale.png           T3: how often the model gives the OLD (overwritten) value
  fig4_selective.png       T2: accuracy by number of distractor facts
  fig5_corruption.png      T5: accuracy drop from injected text
  fig6_efficiency.png      T6: measured latency and theoretical attention cost
Tables (Markdown, paste into the report):
  results_table.md         accuracy per test x regime x variant, with 95% intervals
"""
import argparse
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed colour + marker per model, never re-ordered (palette validated for colour-blind safety)
STYLE = {
    "A": dict(color="#2a78d6", marker="o", label="A: local attention"),
    "B": dict(color="#eb6834", marker="s", label="B: + always-on memory"),
    "C": dict(color="#1baf7a", marker="^", label="C: + learned gated memory"),
    "full_attn_4096": dict(color="#eda100", marker="D", label="Full attention (4,096)"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REACH = 2040          # receptive field of the 8-layer, W=256 backbone
CHANCE = 0.25
TEST_NAMES = {"t1": "T1 Needle", "t2": "T2 Selective memory", "t3": "T3 Contradiction",
              "t4": "T4 Multi-hop", "t5": "T5 Corruption", "t1c": "T1 control (inside window)"}
WINDOW = 256

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2, "lines.markersize": 6,
})


def wilson(k, n, z=1.96):
    """95% interval for a proportion. With 100 questions, +-~9 points: differences
    smaller than that are not evidence of anything."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def load(results_dir, stage):
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"variant-*_{stage}.csv"))):
        v = re.search(r"variant-([A-Za-z0-9]+)_", os.path.basename(path)).group(1)
        df = pd.read_csv(path)
        df["model"] = v
        frames.append(df)
    if not frames:
        raise SystemExit(f"no results/variant-*_{stage}.csv found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def by_bin(df, col="correct"):
    g = df.groupby("bin")[col].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    ci = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


def distance_axes(ax, ylabel="Accuracy", chance=True, xmin=200):
    ax.axvspan(REACH, 4096, color="#f0efec", zorder=0)
    ax.axvline(REACH, color=INK2, lw=1, ls=":")
    ax.text(REACH + 60, 1.01, "beyond 2,040: no path for A without memory", fontsize=8, color=INK2,
            va="bottom")
    if chance:
        ax.axhline(CHANCE, color=INK2, lw=1, ls="--")
        ax.text(xmin + 50, CHANCE + 0.015, "chance (25%)", fontsize=8, color=INK2)
    if xmin < WINDOW:
        ax.axvline(WINDOW, color=INK2, lw=1, ls=":")
        ax.text(WINDOW + 30, 1.01, "256-token window", fontsize=8, color=INK2, va="bottom")
    ax.set_xlim(xmin, 4096)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Distance from fact to question (tokens)")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def label_ends(ax, ends, gap=0.04, fontsize=9):
    """Direct labels at line ends, in text ink. Lines ending within `gap` share one label."""
    groups = []
    for y, x, v in sorted(ends):
        if groups and abs(y - groups[-1][0]) <= gap:
            groups[-1][2].append(v)
        else:
            groups.append([y, x, [v]])
    for y, x, vs in groups:
        ax.annotate(", ".join(vs), (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=fontsize, color=INK)


def line_by_model(ax, df, col="correct"):
    ends = []
    for v in [m for m in STYLE if m in df["model"].unique()]:
        g = by_bin(df[df["model"] == v], col)
        st = STYLE[v]
        ax.fill_between(g["bin"], g["lo"], g["hi"], color=st["color"], alpha=0.12, lw=0)
        ax.plot(g["bin"], g["rate"], color=st["color"], marker=st["marker"], label=st["label"])
        ends.append((g["rate"].iloc[-1], g["bin"].iloc[-1], v))
    label_ends(ax, ends)


def fig_needle(df, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    has_control = (df["test"] == "t1c").any()      # control points: needles inside the window
    line_by_model(ax, df[df["test"].isin(["t1", "t1c"])])
    distance_axes(ax, xmin=0 if has_control else 200)
    ax.set_title("T1 Needle: can the model recall a fact from far back?", loc="left", fontsize=11, pad=36)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_needle.png"), dpi=200)
    plt.close(fig)


def fig_all(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, ["t1", "t2", "t3", "t4"]):
        line_by_model(ax, df[df["test"] == t])
        distance_axes(ax)
        ax.set_title(TEST_NAMES[t], loc="left", fontsize=10, pad=16)
        ax.label_outer()
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_all_tests.png"), dpi=200)
    plt.close(fig)


def fig_stale(df, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    line_by_model(ax, df[df["test"] == "t3"], col="picked_stale")
    distance_axes(ax, ylabel="Picked the OLD value", chance=False)
    ax.set_title("T3 Contradiction: how often the stale value wins", loc="left", fontsize=11, pad=18)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_stale.png"), dpi=200)
    plt.close(fig)


def grouped_bars(ax, table, groups, ylabel):
    models = [m for m in STYLE if m in table.columns]
    w = 0.8 / max(1, len(models))
    for i, v in enumerate(models):
        xs = [j + (i - (len(models) - 1) / 2) * w for j in range(len(groups))]
        ax.bar(xs, table[v].values, width=w - 0.02, color=STYLE[v]["color"], label=STYLE[v]["label"])
        for x, y in zip(xs, table[v].values):
            ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 3 if y >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)


def fig_selective(df, out):
    d = df[df["test"] == "t2"].copy()
    d["k"] = d["variant_info"].astype(int)
    tab = d.groupby(["k", "model"])["correct"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, tab, [f"{k} distractors" for k in tab.index], "Accuracy")
    ax.axhline(CHANCE, color=INK2, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_title("T2 Selective memory: accuracy as distractor facts are added", loc="left", fontsize=11, pad=24)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_selective.png"), dpi=200)
    plt.close(fig)


def corruption_drops(df):
    """Accuracy drop = clean - corrupted, on matched pairs (same fact, same filler)."""
    d = df[df["test"] == "t5"].copy()
    acc = d.groupby(["model", "variant_info"])["correct"].mean().unstack()
    out = pd.DataFrame({"irrelevant text": acc["clean"] - acc["irrelevant"],
                        "near-miss name": acc["clean"] - acc["near_miss"]}).T
    inj = d[d["variant_info"] == "near_miss"].groupby("model")["picked_injected"].mean()
    return out, acc, inj


def fig_corruption(df, out):
    drops, _, _ = corruption_drops(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, drops, list(drops.index), "Accuracy drop vs clean")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title("T5 Corruption: how much injected text hurts", loc="left", fontsize=11, pad=24)
    lo, hi = min(0, drops.min().min()), max(0.05, drops.max().max())
    ax.set_ylim(lo * 1.4, hi * 1.25)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_corruption.png"), dpi=200)
    plt.close(fig)


def fig_efficiency(eff, out):
    e = eff[(eff["batch"] == 1) & (eff["ms_per_seq"] != "OOM")].copy()
    e["ms_per_seq"] = e["ms_per_seq"].astype(float)
    # two separate charts side by side: never two y-scales on one axis
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for v in [m for m in STYLE if m in e["model"].unique()]:
        g = e[e["model"] == v].sort_values("seq_len")
        st = STYLE[v]
        a1.plot(g["seq_len"], g["ms_per_seq"], color=st["color"], marker=st["marker"], label=st["label"])
        a2.plot(g["seq_len"], g["attn_gflops"], color=st["color"], marker=st["marker"], label=st["label"])
    for a, col in ((a1, "ms_per_seq"), (a2, "attn_gflops")):
        ends = []
        for v in [m for m in STYLE if m in e["model"].unique()]:
            g = e[e["model"] == v].sort_values("seq_len")
            ends.append((float(g[col].iloc[-1]), g["seq_len"].iloc[-1], v))
        ends.sort()
        groups = []                     # lines ending within 4% of each other share one label
        for y, x, v in ends:
            if groups and abs(y - groups[-1][0]) <= 0.04 * max(abs(y), 1e-9):
                groups[-1][2].append(v)
            else:
                groups.append([y, x, [v]])
        for y, x, vs in groups:
            names = ", ".join("full attn" if v == "full_attn_4096" else v for v in vs)
            a.annotate(names, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                       fontsize=8, color=INK)
        a.set_xlim(right=e["seq_len"].max() * 1.22)
    a1.set_title("Measured latency (this implementation)", loc="left", fontsize=10)
    a1.set_ylabel("ms per sequence (batch 1)")
    a2.set_title("Theoretical attention cost", loc="left", fontsize=10)
    a2.set_ylabel("Attention GFLOPs per sequence")
    for a in (a1, a2):
        a.set_xlabel("Sequence length (tokens)")
        a.set_ylim(bottom=0)
    a1.legend(fontsize=8)
    fig.text(0.01, 0.01, f"Measured on {e['device'].iloc[0]}. The sliding window is a mask over full attention, "
             "so measured cost does not shrink with the window.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "fig6_efficiency.png"), dpi=200)
    plt.close(fig)


def write_tables(df, eff, out):
    lines = ["# Results (choice accuracy, chance = 25%, 95% Wilson interval)\n"]
    models = [m for m in STYLE if m in df["model"].unique()]
    lines.append("| Test | Regime | " + " | ".join(models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    order = ["t1c", "t1", "t2", "t3", "t4", "t5"]
    for t in sorted(df["test"].unique(), key=lambda x: order.index(x) if x in order else 99):
        for regime in ["control", "weak", "blind"]:
            cells = []
            for v in models:
                s = df[(df["test"] == t) & (df["regime"] == regime) & (df["model"] == v)]["correct"]
                if len(s) == 0:
                    cells.append("-")
                    continue
                lo, hi = wilson(s.sum(), len(s))
                cells.append(f"{s.mean():.1%} [{lo:.0%}-{hi:.0%}] (n={len(s)})")
            if any(c != "-" for c in cells):
                lines.append(f"| {TEST_NAMES.get(t, t)} | {regime} | " + " | ".join(cells) + " |")
    lines.append("\ncontrol = 60-200 tokens back (inside the 256-token window); "
                 "weak = 300-2,000 tokens back (A's theoretical multi-layer reach); "
                 "blind = 2,500-3,900 tokens back (beyond A's 2,040-token reach).\n")

    t3 = df[df["test"] == "t3"]
    if len(t3):
        lines.append("## T3 stale-value rate\n\n| Model | Picked old value |\n|---|---|")
        for v in models:
            s = t3[t3["model"] == v]["picked_stale"]
            if len(s):
                lines.append(f"| {v} | {s.mean():.1%} |")
    if (df["test"] == "t5").any():
        drops, acc, inj = corruption_drops(df)
        lines.append("\n## T5 corruption\n\n| Model | Clean | Irrelevant | Near-miss | Picked near-miss value |")
        lines.append("|---|---|---|---|---|")
        for v in models:
            if v in acc.index:
                lines.append(f"| {v} | {acc.loc[v, 'clean']:.1%} | {acc.loc[v, 'irrelevant']:.1%} | "
                             f"{acc.loc[v, 'near_miss']:.1%} | {inj.get(v, float('nan')):.1%} |")
    if eff is not None:
        lines.append("\n## T6 efficiency (batch 1)\n")
        cols = ["model", "seq_len", "ms_per_seq", "tokens_per_sec", "peak_mem_mb", "attn_gflops",
                "kv_cache_mb", "mem_state_mb"]
        e = eff[eff["batch"] == 1][cols]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in e.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(os.path.join(out, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--stage", default="instruct", help="pretrain | instruct | random")
    ap.add_argument("--out_dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.results_dir, args.stage)
    print("models:", sorted(df["model"].unique()), "| tests:", sorted(df["test"].unique()), "| rows:", len(df))
    tests = set(df["test"])
    if "t1" in tests:
        fig_needle(df, args.out_dir)
    if tests & {"t1", "t2", "t3", "t4"}:
        fig_all(df, args.out_dir)
    if "t3" in tests:
        fig_stale(df, args.out_dir)
    if "t2" in tests:
        fig_selective(df, args.out_dir)
    if "t5" in tests:
        fig_corruption(df, args.out_dir)
    eff_path = os.path.join(args.results_dir, "efficiency.csv")
    eff = pd.read_csv(eff_path) if os.path.exists(eff_path) else None
    if eff is not None:
        fig_efficiency(eff, args.out_dir)
    write_tables(df, eff, args.out_dir)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
