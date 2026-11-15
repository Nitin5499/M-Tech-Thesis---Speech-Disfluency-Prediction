"""
compare_pcfg_modelblocks.py
Compares PCFG vs ModelBlocks syntactic surprisal values
and runs GLMM with both to produce comparison table.
Run from project root: python src/compare_pcfg_modelblocks.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MB_DIR    = os.path.join(ROOT, 'data', 'modelblocks')
RESULTS   = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

print("=" * 60)
print("PCFG vs ModelBlocks Syntactic Surprisal Comparison")
print("=" * 60)

# ── Load both surprisal files ─────────────────────────────────
print("\nLoading PCFG syntactic surprisal...")
pcfg = pd.read_csv(os.path.join(PROCESSED, 'syntactic_surprisal.csv'))
print(f"  {len(pcfg):,} rows, {pcfg['prev_syn_surp'].notna().sum():,} valid")

print("\nLoading ModelBlocks syntactic surprisal...")
mb = pd.read_csv(os.path.join(MB_DIR, 'swbd_5000_surprisal.csv'))
print(f"  {len(mb):,} rows, {mb['prev_syn_surp'].notna().sum():,} valid")

# ── Merge on word_id ──────────────────────────────────────────
print("\nMerging on word_id...")
mb = mb.rename(columns={
    'prev_syn_surp': 'mb_prev_syn_surp',
    'foll_syn_surp': 'mb_foll_syn_surp'
})
merged = pcfg.merge(mb[['word_id','mb_prev_syn_surp','mb_foll_syn_surp']],
                    on='word_id', how='inner')
valid = merged.dropna(subset=['prev_syn_surp','mb_prev_syn_surp'])
print(f"  Words with both PCFG and MB values: {len(valid):,}")

# ── Statistical comparison ────────────────────────────────────
print("\n" + "=" * 60)
print("Statistical Comparison")
print("=" * 60)

corr, pval = stats.pearsonr(valid['prev_syn_surp'], valid['mb_prev_syn_surp'])
print(f"\nPearson correlation (prev_syn_surp): r = {corr:.4f}, p = {pval:.4e}")

print(f"\nPCFG prev_syn_surp:")
print(f"  Mean:   {valid['prev_syn_surp'].mean():.4f}")
print(f"  Std:    {valid['prev_syn_surp'].std():.4f}")
print(f"  Median: {valid['prev_syn_surp'].median():.4f}")

print(f"\nModelBlocks prev_syn_surp:")
print(f"  Mean:   {valid['mb_prev_syn_surp'].mean():.4f}")
print(f"  Std:    {valid['mb_prev_syn_surp'].std():.4f}")
print(f"  Median: {valid['mb_prev_syn_surp'].median():.4f}")

# ── Comparison table ──────────────────────────────────────────
comp_table = pd.DataFrame({
    'Metric': ['Words compared', 'Mean surprisal', 'Std surprisal',
               'Median surprisal', 'Pearson r with MB', 'Coverage'],
    'PCFG': [
        f"{len(valid):,}",
        f"{valid['prev_syn_surp'].mean():.4f}",
        f"{valid['prev_syn_surp'].std():.4f}",
        f"{valid['prev_syn_surp'].median():.4f}",
        f"{corr:.4f}",
        f"{len(pcfg[pcfg['prev_syn_surp'].notna()]):,} / {len(pcfg):,} words"
    ],
    'ModelBlocks': [
        f"{len(valid):,}",
        f"{valid['mb_prev_syn_surp'].mean():.4f}",
        f"{valid['mb_prev_syn_surp'].std():.4f}",
        f"{valid['mb_prev_syn_surp'].median():.4f}",
        "1.0000 (reference)",
        f"{len(mb[mb['mb_prev_syn_surp'].notna()]):,} / {len(mb):,} words"
    ]
})

print("\n" + "=" * 60)
print("Comparison Table")
print("=" * 60)
print(comp_table.to_string(index=False))
comp_table.to_csv(os.path.join(RESULTS, 'pcfg_vs_modelblocks_comparison.csv'), index=False)

# ── Scatter plot ──────────────────────────────────────────────
print("\nGenerating comparison figure...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter
ax = axes[0]
ax.scatter(valid['prev_syn_surp'], valid['mb_prev_syn_surp'],
           alpha=0.3, s=10, color='#1E6091')
ax.set_xlabel('PCFG Syntactic Surprisal', fontsize=12)
ax.set_ylabel('ModelBlocks Syntactic Surprisal', fontsize=12)
ax.set_title(f'PCFG vs ModelBlocks\nr = {corr:.4f}', fontsize=13)
# Add regression line
m, b = np.polyfit(valid['prev_syn_surp'], valid['mb_prev_syn_surp'], 1)
x_line = np.linspace(valid['prev_syn_surp'].min(), valid['prev_syn_surp'].max(), 100)
ax.plot(x_line, m*x_line+b, 'r-', linewidth=2, alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Distribution comparison
ax2 = axes[1]
ax2.hist(valid['prev_syn_surp'].clip(0,30), bins=50, alpha=0.6,
         color='#1E6091', label='PCFG', density=True)
ax2.hist(valid['mb_prev_syn_surp'].clip(0,30), bins=50, alpha=0.6,
         color='#E85B4B', label='ModelBlocks', density=True)
ax2.set_xlabel('Syntactic Surprisal (bits)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Distribution Comparison', fontsize=13)
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('PCFG vs ModelBlocks Syntactic Surprisal Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(RESULTS, 'figure_pcfg_vs_modelblocks.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure -> {out}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nFiles saved:")
print(f"  results/pcfg_vs_modelblocks_comparison.csv")
print(f"  results/figure_pcfg_vs_modelblocks.png")