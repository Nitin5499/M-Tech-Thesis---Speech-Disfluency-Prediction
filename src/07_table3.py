"""
07_table3.py
Reproduces Table 3 from Dammalapati et al. (2021).

Table 3 analyzes:
1. Elongation ratio of words preceding disfluencies
2. Proportion of function words preceding disfluencies
3. Comparison across filler, reparandum, and fluent contexts

INPUT:  data/processed/words_features.csv
OUTPUT: results/table3_elongation.csv
"""

import os
import pandas as pd
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FEATURES_CSV = os.path.join(ROOT_DIR, "data", "processed", "words_features.csv")
OUTPUT_CSV   = os.path.join(ROOT_DIR, "results", "table3_elongation.csv")
OUTPUT_TXT   = os.path.join(ROOT_DIR, "results", "table3_elongation.txt")

os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)


def compute_table3(df):
    print("Computing Table 3 statistics ...")

    # ── Elongation analysis ───────────────────────────────────
    # Paper: words immediately BEFORE a disfluency are elongated
    # We look at prev_dur relative to mean word duration

    results = []

    for label in ["filler", "reparandum", "fluent"]:
        subset = df[df["label"] == label]
        n      = len(subset)

        # Elongation ratio
        elong_mean = subset["elongation"].mean()
        elong_std  = subset["elongation"].std()
        elong_med  = subset["elongation"].median()

        # Duration stats
        dur_mean = subset["duration"].mean()
        dur_std  = subset["duration"].std()

        # Function word proportion
        func_prop = 1 - subset["is_content"].mean()

        # Lexical surprisal
        lex_surp_mean = subset["prev_lex_surp"].mean()

        # Syntactic surprisal
        syn_surp_mean = subset["prev_syn_surp"].mean()

        # DLT costs
        ic_mean = subset["prev_IC"].mean()
        sc_mean = subset["prev_SC"].mean()

        results.append({
            "label":          label,
            "n_words":        n,
            "mean_duration":  round(dur_mean, 4),
            "std_duration":   round(dur_std, 4),
            "mean_elongation":round(elong_mean, 4),
            "std_elongation": round(elong_std, 4),
            "med_elongation": round(elong_med, 4),
            "func_word_prop": round(func_prop, 4),
            "mean_lex_surp":  round(lex_surp_mean, 4),
            "mean_syn_surp":  round(syn_surp_mean, 4),
            "mean_IC":        round(ic_mean, 4),
            "mean_SC":        round(sc_mean, 4),
        })

    table3 = pd.DataFrame(results)
    return table3


def print_table3(table3):
    print("\n" + "=" * 65)
    print("Table 3: Elongation and Feature Statistics by Label")
    print("=" * 65)
    print(f"\n{'Label':<12} {'N':>8} {'Mean Dur':>10} {'Mean Elong':>12} {'Func%':>8} {'Mean LexS':>10}")
    print("-" * 65)
    for _, row in table3.iterrows():
        print(f"{row['label']:<12} {row['n_words']:>8,} "
              f"{row['mean_duration']:>10.4f} "
              f"{row['mean_elongation']:>12.4f} "
              f"{row['func_word_prop']*100:>7.1f}% "
              f"{row['mean_lex_surp']:>10.4f}")
    print("-" * 65)

    print("\nPaper Table 3 targets:")
    print("  Fillers have higher elongation than fluent words")
    print("  Reparandums have higher elongation than fluent words")
    print("  Function word proportion higher before disfluencies")

    # Statistical tests
    print("\n--- Statistical Tests ---")
    try:
        from scipy import stats

        filler  = df[df["label"] == "filler"]["elongation"].dropna()
        rep     = df[df["label"] == "reparandum"]["elongation"].dropna()
        fluent  = df[df["label"] == "fluent"]["elongation"].dropna()

        # Filler vs Fluent
        t_stat, p_val = stats.ttest_ind(filler, fluent)
        print(f"Filler vs Fluent elongation:     t={t_stat:.3f}, p={p_val:.4e}")

        # Reparandum vs Fluent
        t_stat, p_val = stats.ttest_ind(rep, fluent)
        print(f"Reparandum vs Fluent elongation: t={t_stat:.3f}, p={p_val:.4e}")

    except Exception as e:
        print(f"Statistical test failed: {e}")


if __name__ == "__main__":

    print("=" * 55)
    print("Step 7: Table 3 — Elongation Analysis")
    print("=" * 55)

    print("\nLoading words_features.csv ...")
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["elongation", "duration"])
    print(f"  {len(df):,} words loaded")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    table3 = compute_table3(df)
    print_table3(table3)

    table3.to_csv(OUTPUT_CSV, index=False)

    # Save readable text version
    with open(OUTPUT_TXT, "w") as f:
        f.write("Table 3: Elongation and Feature Statistics\n")
        f.write("=" * 65 + "\n\n")
        f.write(table3.to_string(index=False))
        f.write("\n\nPaper finding: Words before disfluencies are elongated\n")
        f.write("relative to their average duration.\n")

    print(f"\nSaved -> {OUTPUT_CSV}")
    print(f"Saved -> {OUTPUT_TXT}")
    print("\n" + "=" * 55)
    print("Step 7 COMPLETE")
    print("=" * 55)