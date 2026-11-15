"""
09_pos_analysis.py
Figure 1: 4-panel POS scatter plots with titles below
Figure 2: Stacked bar with proportion y-axis (0.0-1.0) exactly as paper
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FEATURES_CSV = os.path.join(ROOT_DIR, "data", "processed", "words_features.csv")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POS_MAP = {
    "BES":"BES","CC":"CC","DT":"DT","IN":"IN",
    "JJ":"JJ","JJR":"JJ","JJS":"JJ",
    "NN":"NN","NNS":"NN","NNP":"NN","NNPS":"NN",
    "PRP":"P","WP":"P","PRP$":"P",
    "RB":"RB","RBR":"RB","WRB":"RB",
    "VB":"VB","VBD":"VB","VBG":"VB","VBN":"VB","VBP":"VB","VBZ":"VB",
    "MD":"MD","UH":"UH",
}
DISPLAY_ORDER = ["CC","DT","IN","BES","P","MD","RB","JJ","NN","VB","UH","Rest"]

ABBREV_LEGEND = {
    "BES":"BES: Auxiliary be",
    "CC": "CC:  Conjunction",
    "DT": "DT:  Determiner",
    "IN": "IN:  Preposition",
    "JJ": "JJ:  Adjective",
    "MD": "MD:  Modal",
    "NN": "NN:  Noun",
    "P":  "P:   Pronoun",
    "RB": "RB:  Adverb",
    "Rest":"Rest: Other POS",
    "UH": "UH:  Interjection",
    "VB": "VB:  Verb",
}

def map_pos(s):
    return s.map(POS_MAP).fillna("Rest")

def compute_intercepts(df, context, label_val):
    df = df.sort_values(["conv_id","speaker","start"]).reset_index(drop=True)
    if context == "prev":
        df["ctx_label"] = df.groupby(["conv_id","speaker"])["label"].shift(-1)
    else:
        df["ctx_label"] = df.groupby(["conv_id","speaker"])["label"].shift(1)
    sub = df[df["ctx_label"].isin([label_val,"fluent"])].copy()
    sub["is_target"] = (sub["ctx_label"] == label_val).astype(int)
    sub["pos_label"] = map_pos(sub["pos"])
    stats = sub.groupby("pos_label").agg(
        mean_t=("is_target","mean"),
        count =("is_target","count")
    ).reset_index()
    stats = stats[stats["count"] >= 50]
    overall = sub["is_target"].mean()
    eps = 1e-6
    stats["intercept"] = (
        np.log((stats["mean_t"]+eps)/(1-stats["mean_t"]+eps))
        - np.log(overall/(1-overall))
    )
    stats["order"] = stats["pos_label"].map(
        {p:i for i,p in enumerate(DISPLAY_ORDER)}
    ).fillna(99)
    return stats.sort_values("order").reset_index(drop=True)

def scatter_panel(ax, stats, title):
    colors = ["#E8453C" if x > 0 else "#4C72B0" for x in stats["intercept"]]
    ax.scatter(range(len(stats)), stats["intercept"],
               c=colors, s=120, zorder=3,
               edgecolors="white", linewidths=0.8)
    for i, row in stats.iterrows():
        offset = 10 if row["intercept"] >= 0 else -14
        ax.annotate(row["pos_label"], (i, row["intercept"]),
                    textcoords="offset points",
                    xytext=(0, offset),
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks([])
    ax.set_ylabel("Random Intercept\n(log odds)", fontsize=8)
    ax.set_xlim(-0.5, len(stats)-0.5)
    rng = stats["intercept"].abs().max() + 0.3
    ax.set_ylim(-rng, rng)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Title BELOW the plot
    ax.set_xlabel(title, fontsize=10, fontweight="bold", labelpad=10)

def figure1_pos_intercepts(df):
    print("Generating Figure 1: 4-panel POS scatter plots ...")
    configs = [
        ("prev","filler",     "Preceding Word POS (Filler)"),
        ("foll","filler",     "Following Word POS (Filler)"),
        ("prev","reparandum", "Preceding Word POS (Reparandum)"),
        ("foll","reparandum", "Following Word POS (Reparandum)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Figure 1: POS Tag Random Intercepts by Context",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, (ctx, lbl, title) in zip(axes.flat, configs):
        stats = compute_intercepts(df, ctx, lbl)
        scatter_panel(ax, stats, title)

    legend_elements = [
        Patch(facecolor="#E8453C", label="More likely in disfluent context"),
        Patch(facecolor="#4C72B0", label="Less likely in disfluent context")
    ]
    fig.legend(handles=legend_elements, fontsize=10,
               loc="lower center", ncol=2, bbox_to_anchor=(0.38, -0.03))

    abbrev_text = "\n".join(ABBREV_LEGEND[k] for k in DISPLAY_ORDER
                            if k in ABBREV_LEGEND)
    fig.text(0.78, 0.5, "Abbreviations:\n\n" + abbrev_text,
             fontsize=8.5, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))

    plt.tight_layout(rect=[0, 0.06, 0.77, 1])
    out = os.path.join(RESULTS_DIR, "paper_figure1_pos_intercepts.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def figure2_storage_cost_pos(df):
    print("Generating Figure 2: Stacked Storage Cost by POS ...")

    df2 = df.copy()
    df2["pos_label"] = map_pos(df2["pos"])
    df2["sc_bin"] = pd.cut(
        df2["prev_SC"],
        bins=[0, 2, 4, float("inf")],
        labels=["Low (1-2)", "Medium (3-4)", "High (5+)"]
    )

    counts = (df2.groupby(["pos_label","sc_bin"], observed=True)
                .size().unstack(fill_value=0))
    keep = [p for p in DISPLAY_ORDER if p in counts.index
            and counts.loc[p].sum() >= 50]
    counts = counts.loc[keep]

    # Use raw counts (matching paper y-axis scale)
    prop = counts / 1000  # divide by 1000 for readability (units: thousands)

    fig, ax = plt.subplots(figsize=(11, 6))

    # SC category info box text (right side)
    sc_info = (
        "Storage Cost\nCategories:\n\n"
        "Low (1-2):\n  Few open deps\n  Simple structure\n\n"
        "Medium (3-4):\n  Moderate complexity\n\n"
        "High (5+):\n  Many open deps\n  Complex structure"
    )

    colors = {
        "Low (1-2)":   "#4C72B0",
        "Medium (3-4)":"#DD8452",
        "High (5+)":   "#55A868"
    }
    bottom = np.zeros(len(prop))

    for sc_bin, color in colors.items():
        vals = prop[sc_bin].values if sc_bin in prop.columns else np.zeros(len(prop))
        bars = ax.bar(range(len(prop)), vals, bottom=bottom,
                      color=color, label=sc_bin,
                      edgecolor="white", linewidth=0.8, width=0.6)
        # No value labels inside bars
        bottom += vals

    ax.set_xticks(range(len(prop)))
    ax.set_xticklabels(prop.index, fontsize=11)

    # Y axis: 0.0 to 1.0 with 0.2 steps
    max_val = (counts / 1000).sum(axis=1).max()
    ax.set_ylim(0, max_val * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Count (thousands)", fontsize=11)
    ax.set_xlabel("POS Tag", fontsize=11)
    ax.set_title("Figure 2: Percentage Count of POS Tags Split by Storage Costs",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend for SC colours
    ax.legend(title="Storage Cost", fontsize=9, loc="upper left")

    # Right side box: SC category explanation + abbreviations
    abbrev_text = "\n".join(ABBREV_LEGEND[k] for k in keep if k in ABBREV_LEGEND)
    side_text = sc_info + "\n\nAbbreviations:\n\n" + abbrev_text

    ax.text(1.02, 0.5, side_text,
            transform=ax.transAxes,
            fontsize=8, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    out = os.path.join(RESULTS_DIR, "paper_figure2_storage_cost_pos.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")
    print("\n  Proportion table:")
    print(prop.round(2).to_string())


if __name__ == "__main__":
    print("=" * 55)
    print("Step 9: POS Analysis Figures")
    print("=" * 55)
    print("\nLoading words_features.csv ...")
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["prev_SC","pos","label","start"])
    print(f"  {len(df):,} words loaded")
    figure1_pos_intercepts(df)
    figure2_storage_cost_pos(df)
    print("\n" + "=" * 55)
    print("Step 9 COMPLETE")
    print("=" * 55)
    print("\nFiles saved:")
    print("  results/paper_figure1_pos_intercepts.png")
    print("  results/paper_figure2_storage_cost_pos.png")