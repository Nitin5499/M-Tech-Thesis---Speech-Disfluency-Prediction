"""
Stage 4 -- Plot baseline vs enhanced CV accuracy across all 5 LLMs x 2 tasks.

Reads results/table_multi_llm_comparison.csv (produced by 19_run_glmm_multi_llm.R)
and writes results/multi_llm_comparison.png: a 1x2 grouped bar chart (filler,
reparandum), baseline vs enhanced, direct-labeled with the accuracy gain and
significance marker.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLOR_BASELINE = "#2a78d6"
COLOR_ENHANCED = "#eb6834"
TEXT_PRIMARY = "#1a1a1a"
TEXT_MUTED = "#6b6b6b"
GRID_COLOR = "#e4e4e4"

df = pd.read_csv("results/table_multi_llm_comparison.csv")

# keep display order = run order (param-count-ish), not alphabetical
MODEL_ORDER = ["gpt2", "dgpt2", "gpt2m", "pyt160", "opt125"]
LABELS = {
    "gpt2": "GPT-2\n(124M)",
    "dgpt2": "DistilGPT-2\n(82M)",
    "gpt2m": "GPT-2-medium\n(355M)",
    "pyt160": "Pythia\n(160M)",
    "opt125": "OPT\n(125M)",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
fig.patch.set_facecolor("#fcfcfb")

TASKS = [("filler", "Filler vs. fluent"), ("reparandum", "Reparandum vs. fluent")]

for ax, (task, title) in zip(axes, TASKS):
    sub = df[df["task"] == task].set_index("tag").loc[MODEL_ORDER]
    x = range(len(MODEL_ORDER))
    width = 0.36
    gap = 0.02

    bars_base = ax.bar(
        [i - width / 2 - gap for i in x], sub["acc_baseline"], width,
        color=COLOR_BASELINE, label="Baseline (10 features)",
        edgecolor="#fcfcfb", linewidth=1,
    )
    bars_enh = ax.bar(
        [i + width / 2 + gap for i in x], sub["acc_enhanced"], width,
        color=COLOR_ENHANCED, label="+ LLM surprisal",
        edgecolor="#fcfcfb", linewidth=1,
    )

    # y-range: zoom to where the actual separation is, per-panel
    lo = min(sub["acc_baseline"].min(), sub["acc_enhanced"].min())
    hi = max(sub["acc_baseline"].max(), sub["acc_enhanced"].max())
    pad = max(0.5, (hi - lo) * 0.6)
    ax.set_ylim(lo - pad, hi + pad * 1.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[t] for t in MODEL_ORDER], fontsize=9, color=TEXT_PRIMARY)
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_MUTED)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.set_facecolor("#fcfcfb")

    # direct labels: accuracy diff + significance marker, above the enhanced bar
    for i, tag in enumerate(MODEL_ORDER):
        row = sub.loc[tag]
        diff = row["acc_diff"]
        sig = row["significant"]
        marker = "*" if str(sig).strip().upper() == "TRUE" else "n.s."
        label = f"{diff:+.2f} {marker}"
        bar_top = max(row["acc_baseline"], row["acc_enhanced"])
        ax.text(
            i, bar_top + pad * 0.25, label,
            ha="center", va="bottom", fontsize=8.5,
            color=TEXT_PRIMARY if marker == "*" else TEXT_MUTED,
            fontweight="bold" if marker == "*" else "normal",
        )

axes[0].set_ylabel("5-fold CV accuracy (axis truncated)", fontsize=10, color=TEXT_PRIMARY)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
    ncol=2, frameon=False, fontsize=10,
)

fig.suptitle(
    "LLM surprisal helps filler prediction; reparandum accuracy barely changes",
    fontsize=13, color=TEXT_PRIMARY, y=1.12, fontweight="bold",
)
fig.text(
    0.5, 1.065,
    "10-feature baseline vs. baseline + model's prev/foll surprisal · * = likelihood-ratio test p < .05 (tests fit, not accuracy)",
    ha="center", fontsize=9, color=TEXT_MUTED,
)

plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig("results/multi_llm_comparison.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print("Saved -> results/multi_llm_comparison.png")