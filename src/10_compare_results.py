"""
10_compare_results.py
Compares results between:
  - Method 1: PCFG syntactic surprisal (our approximation)
  - Method 2: ModelBlocks syntactic surprisal (paper's approach)

Produces a side by side comparison table and figure.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Step 10: Comparing PCFG vs ModelBlocks Results")
print("=" * 60)

# ── Load both accuracy files ──────────────────────────────────
def load_accuracy(filepath):
    results = {}
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if "Baseline" in line:
                    results["baseline"] = float(line.split()[-2])
                elif "Full model" in line:
                    results["full_model"] = float(line.split()[-2])
                elif "Paper baseline" in line:
                    results["paper_baseline"] = float(line.split()[-1].replace("%",""))
                elif "Paper full" in line:
                    results["paper_full"] = float(line.split()[-1].replace("%",""))
    except Exception as e:
        print(f"Could not load {filepath}: {e}")
    return results

# PCFG results (already exist)
pcfg_filler = load_accuracy(os.path.join(RESULTS_DIR, "accuracy_filler.txt"))
pcfg_rep    = load_accuracy(os.path.join(RESULTS_DIR, "accuracy_reparandum.txt"))

# ModelBlocks results (will exist after re-running GLMM)
mb_filler_path = os.path.join(RESULTS_DIR, "accuracy_filler_modelblocks.txt")
mb_rep_path    = os.path.join(RESULTS_DIR, "accuracy_reparandum_modelblocks.txt")

mb_filler = load_accuracy(mb_filler_path)
mb_rep    = load_accuracy(mb_rep_path)

# ── Load both coefficient tables ─────────────────────────────
def load_coefficients(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except:
        return None

pcfg_coef_rep = load_coefficients(
    os.path.join(RESULTS_DIR, "table1_reparandum.csv"))
mb_coef_rep   = load_coefficients(
    os.path.join(RESULTS_DIR, "table1_reparandum_modelblocks.csv"))

# ── Print comparison table ────────────────────────────────────
print("\n" + "=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)
print(f"\n{'Metric':<30} {'PCFG':>10} {'ModelBlocks':>12} {'Paper':>10}")
print("-" * 60)

rows = [
    ("Filler — Baseline",     pcfg_filler.get("baseline","N/A"),
                               mb_filler.get("baseline","N/A"),
                               pcfg_filler.get("paper_baseline","66.80")),
    ("Filler — Full Model",   pcfg_filler.get("full_model","N/A"),
                               mb_filler.get("full_model","N/A"),
                               pcfg_filler.get("paper_full","81.21")),
    ("Reparandum — Baseline", pcfg_rep.get("baseline","N/A"),
                               mb_rep.get("baseline","N/A"),
                               pcfg_rep.get("paper_baseline","66.29")),
    ("Reparandum — Full Model",pcfg_rep.get("full_model","N/A"),
                               mb_rep.get("full_model","N/A"),
                               pcfg_rep.get("paper_full","71.11")),
]

for label, pcfg, mb, paper in rows:
    pcfg_str  = f"{pcfg:.2f}%" if isinstance(pcfg, float) else str(pcfg)
    mb_str    = f"{mb:.2f}%"   if isinstance(mb, float)   else "Pending"
    paper_str = f"{paper}%"    if paper else "N/A"
    print(f"{label:<30} {pcfg_str:>10} {mb_str:>12} {paper_str:>10}")

# ── Coefficient comparison ────────────────────────────────────
if pcfg_coef_rep is not None and mb_coef_rep is not None:
    print("\n" + "=" * 60)
    print("COEFFICIENT COMPARISON — Reparandum Model")
    print("=" * 60)
    print(f"\n{'Feature':<20} {'PCFG Coef':>12} {'MB Coef':>12} {'Change':>10}")
    print("-" * 60)

    features = ["prev_lex_surp","foll_lex_surp","prev_syn_surp",
                "foll_syn_surp","prev_IC","foll_IC","prev_SC",
                "foll_SC","prev_dur","foll_dur"]

    for feat in features:
        pcfg_row = pcfg_coef_rep[pcfg_coef_rep["feature"]==feat]
        mb_row   = mb_coef_rep[mb_coef_rep["feature"]==feat]
        if not pcfg_row.empty and not mb_row.empty:
            pc = pcfg_row["Estimate"].values[0]
            mc = mb_row["Estimate"].values[0]
            ch = mc - pc
            arrow = "↑" if ch > 0 else "↓"
            print(f"{feat:<20} {pc:>12.4f} {mc:>12.4f} {arrow}{abs(ch):>8.4f}")
        elif not pcfg_row.empty:
            pc = pcfg_row["Estimate"].values[0]
            print(f"{feat:<20} {pc:>12.4f} {'Pending':>12} {'N/A':>10}")

# ── Generate comparison figure ────────────────────────────────
print("\nGenerating comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PCFG vs ModelBlocks — Syntactic Surprisal Comparison",
             fontsize=14, fontweight="bold")

tasks    = ["Filler", "Reparandum"]
pcfg_acc = [pcfg_filler.get("full_model", 0),
            pcfg_rep.get("full_model", 0)]
mb_acc   = [mb_filler.get("full_model", None),
            mb_rep.get("full_model", None)]
paper_acc= [81.21, 71.11]

x = np.arange(len(tasks))
w = 0.25

ax = axes[0]
bars1 = ax.bar(x - w, pcfg_acc,   w, label="Our PCFG",     color="#1E6091", alpha=0.85)
if any(v is not None for v in mb_acc):
    bars2 = ax.bar(x,     mb_acc,    w, label="ModelBlocks",  color="#2E7D52", alpha=0.85)
bars3 = ax.bar(x + w, paper_acc, w, label="Paper Target", color="#B8730A", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Full Model Accuracy", fontsize=12)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f"{h:.1f}%", ha="center", fontsize=9)
for bar in bars3:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+1,
            f"{h:.1f}%", ha="center", fontsize=9)

# Coefficient comparison for syntactic surprisal
ax2 = axes[1]
feat_labels = ["prev_syn_surp", "foll_syn_surp"]
if pcfg_coef_rep is not None:
    pcfg_vals = []
    mb_vals   = []
    for feat in feat_labels:
        row = pcfg_coef_rep[pcfg_coef_rep["feature"]==feat]
        pcfg_vals.append(row["Estimate"].values[0] if not row.empty else 0)
        if mb_coef_rep is not None:
            row2 = mb_coef_rep[mb_coef_rep["feature"]==feat]
            mb_vals.append(row2["Estimate"].values[0] if not row2.empty else None)
        else:
            mb_vals.append(None)

    x2 = np.arange(len(feat_labels))
    ax2.bar(x2 - 0.2, pcfg_vals, 0.35, label="Our PCFG",
            color="#1E6091", alpha=0.85)
    if any(v is not None for v in mb_vals):
        ax2.bar(x2 + 0.2, mb_vals, 0.35, label="ModelBlocks",
                color="#2E7D52", alpha=0.85)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["prev_syn_surp","foll_syn_surp"], fontsize=11)
    ax2.set_ylabel("Coefficient Estimate", fontsize=11)
    ax2.set_title("Syntactic Surprisal Coefficients\n(Reparandum Model)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
else:
    ax2.text(0.5, 0.5, "ModelBlocks results\nnot yet available",
             ha="center", va="center", fontsize=13, color="gray",
             transform=ax2.transAxes)
    ax2.set_title("Syntactic Surprisal Coefficients\n(Reparandum Model)", fontsize=12)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "figure_pcfg_vs_modelblocks.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("\nNote: Run this script again after re-running GLMM with")
print("ModelBlocks surprisal to see the full comparison.")