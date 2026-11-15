"""
08_figures.py
Reproduces Figures from Dammalapati et al. (2021).
Fixed version - corrected Figure 2 label overlap and Figure 4 negative IC bars.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FEATURES_CSV = os.path.join(ROOT_DIR, "data", "processed", "words_features.csv")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def figure1_elongation(df):
    print("Generating Figure 1: Elongation by label ...")
    labels = ["fluent", "filler", "reparandum"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    means  = [df[df["label"]==l]["elongation"].mean() for l in labels]
    stds   = [df[df["label"]==l]["elongation"].std()  for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color=colors, width=0.5,
                  yerr=stds, capsize=5, edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1,
               alpha=0.7, label="Baseline (elongation = 1.0)")
    ax.set_ylabel("Mean Elongation Ratio", fontsize=12)
    ax.set_xlabel("Word Label", fontsize=12)
    ax.set_title("Figure 1: Word Elongation by Disfluency Label", fontsize=13)
    ax.set_ylim(0, max(means) * 1.4)
    ax.legend(fontsize=10)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure1_elongation_by_label.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def figure2_function_words(df):
    """
    FIXED:
    - Labels now placed only when proportion > 5% to avoid overlap
    - Filler bar shows single '100%' label centered cleanly
    - Content word label moved to avoid collision
    """
    print("Generating Figure 2: Function word proportion by label ...")

    labels        = ["fluent", "filler", "reparandum"]
    func_colors   = ["#3A6FB0", "#C8603A", "#3A8A58"]
    content_colors= ["#7BADD4", "#E8A882", "#7AC49A"]

    func_props    = [1 - df[df["label"]==l]["is_content"].mean() for l in labels]
    content_props = [    df[df["label"]==l]["is_content"].mean() for l in labels]

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x, func_props,    color=func_colors,    width=0.5, label="Function words")
    ax.bar(x, content_props, color=content_colors, width=0.5,
           bottom=func_props, label="Content words")

    # Fixed labels — only show if proportion > 3%
    for i, (fp, cp) in enumerate(zip(func_props, content_props)):
        # Function word label
        if fp > 0.03:
            ax.text(i, fp / 2, f"{fp*100:.1f}%",
                    ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        # Content word label — only show if there is enough space
        if cp > 0.03:
            ax.text(i, fp + cp / 2, f"{cp*100:.1f}%",
                    ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("Word Label", fontsize=12)
    ax.set_title("Figure 2: Function vs Content Word Proportion by Label", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure2_function_words.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def figure3_surprisal(df):
    print("Generating Figure 3: Surprisal by label ...")
    labels = ["fluent", "filler", "reparandum"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat, title in zip(
        axes,
        ["prev_lex_surp", "prev_syn_surp"],
        ["Lexical Surprisal", "Syntactic Surprisal"]
    ):
        means = [df[df["label"]==l][feat].mean() for l in labels]
        sems  = [df[df["label"]==l][feat].sem()  for l in labels]
        bars  = ax.bar(labels, means, color=colors, width=0.5,
                       yerr=sems, capsize=5, edgecolor="white")
        ax.set_title(f"{title} by Label", fontsize=12)
        ax.set_ylabel("Mean Surprisal (bits)", fontsize=11)
        ax.set_ylim(0, max(means) * 1.2)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Figure 3: Surprisal Values by Disfluency Label", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure3_surprisal_by_label.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def figure4_dlt(df):
    """
    FIXED:
    - Use SEM (standard error of mean) instead of STD for error bars
      so bars don't go negative — STD was too large relative to mean
    - Y axis set to start at 0
    - Added note explaining IC distribution
    """
    print("Generating Figure 4: DLT costs by label ...")
    labels = ["fluent", "filler", "reparandum"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, feat, title in zip(
        axes,
        ["prev_IC", "prev_SC"],
        ["Integration Cost (IC)", "Storage Cost (SC)"]
    ):
        means = [df[df["label"]==l][feat].mean() for l in labels]
        # Use SEM instead of STD — avoids negative bars
        sems  = [df[df["label"]==l][feat].sem()  for l in labels]

        bars = ax.bar(labels, means, color=colors, width=0.5,
                      yerr=sems, capsize=5, edgecolor="white")
        ax.set_title(f"{title} by Label", fontsize=12)
        ax.set_ylabel("Mean Cost", fontsize=11)
        ax.set_ylim(0, max(means) * 1.3)  # Force y-axis to start at 0

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(sems) * 0.1,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Figure 4: DLT Costs by Disfluency Label\n"
                 "(Error bars = Standard Error of Mean)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure4_dlt_by_label.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


if __name__ == "__main__":

    print("=" * 55)
    print("Step 8: Generate Figures (Fixed Version)")
    print("=" * 55)

    print("\nLoading words_features.csv ...")
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["elongation","duration","prev_lex_surp"])
    print(f"  {len(df):,} words loaded")

    figure1_elongation(df)
    figure2_function_words(df)
    figure3_surprisal(df)
    figure4_dlt(df)

    print("\n" + "=" * 55)
    print("Step 8 COMPLETE — All figures saved to results/")
    print("=" * 55)
    print("\nFiles saved:")
    print("  results/figure1_elongation_by_label.png")
    print("  results/figure2_function_words.png")
    print("  results/figure3_surprisal_by_label.png")
    print("  results/figure4_dlt_by_label.png")