"""
generate_comparison_tables.py
==============================
Generates comprehensive comparison tables between:
  1. PCFG syntactic surprisal (our approximation — old pipeline)
  2. ModelBlocks partial (professor's model — old pipeline, no DC correction)
  3. Clean pipeline (DC correction + ModelBlocks + feature propagation)
  4. Paper's reported results

Run: python src/generate_comparison_tables.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

print("=" * 60)
print("Generating Comparison Tables — All 3 Approaches vs Paper")
print("=" * 60)

# ── Colours ───────────────────────────────────────────────────
NAVY  = '#1A3A5C'
OCEAN = '#1E6091'
CORAL = '#E85B4B'
TEAL  = '#2E7D8C'
GREEN = '#2E7D52'

# ── Table 1: Accuracy Comparison — All Approaches ─────────────
acc_data = {
    'Task': [
        'Filler', 'Filler',
        'Reparandum', 'Reparandum'
    ],
    'Metric': [
        'Baseline', 'Full Model',
        'Baseline', 'Full Model'
    ],
    'PCFG (old)': [
        '99.96%', '99.96%',
        '63.93%', '65.75%'
    ],
    'ModelBlocks Partial (old)': [
        '100%', '100%',
        '63.65%', '67.33%'
    ],
    'Clean Pipeline (new)': [
        '99.96%', '99.95%',
        '62.60%', '71.11%'
    ],
    'Paper Target': [
        '66.80%', '81.21%',
        '66.29%', '71.11%'
    ],
}
acc_df = pd.DataFrame(acc_data)
acc_df.to_csv(os.path.join(RESULTS, 'table_accuracy_comparison.csv'), index=False)
print("\nTable 1: Accuracy Comparison — All Approaches")
print(acc_df.to_string(index=False))

# ── Table 2: Coefficient Comparison — Reparandum Model ────────
coef_data = {
    'Feature': [
        '(Intercept)',
        'prev_lex_surp', 'foll_lex_surp',
        'prev_syn_surp', 'foll_syn_surp',
        'prev_IC', 'foll_IC',
        'prev_SC', 'foll_SC',
        'prev_dur', 'foll_dur'
    ],
    'PCFG Estimate': [
        -1.069,
        -0.026, +0.210,
        -0.165, -0.179,
        -0.017, -0.008,
        -0.029, +0.064,
        +0.348, -0.174
    ],
    'PCFG p-value': [
        '1.6e-66',
        '0.089', '2.4e-31',
        '5.3e-06', '2.8e-24',
        '0.230', '0.588',
        '0.094', '1.5e-04',
        '5.1e-93', '1.9e-29'
    ],
    'MB Partial Estimate': [
        -0.568,
        +0.023, +0.167,
        -0.292, +0.283,
        -0.049, +0.032,
        -0.033, +0.086,
        +0.236, -0.383
    ],
    'MB Partial p-value': [
        '5.7e-07',
        '0.666', '4.1e-03',
        '4.4e-04', '1.0e-07',
        '0.323', '0.556',
        '0.602', '0.152',
        '1.9e-04', '3.2e-13'
    ],
    'Clean Pipeline Estimate': [
        -1.216,
        -0.268, +0.272,
        +0.118, +0.002,
        +0.058, +0.188,
        -0.766, +0.755,
        +0.475, -0.083
    ],
    'Clean Pipeline p-value': [
        '7.6e-105',
        '3.9e-47', '3.0e-48',
        '1.1e-16', '0.912',
        '1.9e-04', '8.7e-23',
        '2.2e-271', '5.3e-307',
        '1.6e-154', '1.7e-08'
    ],
}
coef_df = pd.DataFrame(coef_data)
coef_df.to_csv(os.path.join(RESULTS, 'table_coefficient_comparison.csv'), index=False)
print("\nTable 2: Coefficient Comparison — Reparandum Model")
print(coef_df.to_string(index=False))

# ── Table 3: Gap Analysis ──────────────────────────────────────
gap_data = {
    'Method': [
        'PCFG (old pipeline)',
        'ModelBlocks partial (old pipeline)',
        'Clean pipeline — DC correction + propagation',
        'Paper (Dammalapati et al. 2021)'
    ],
    'Reparandum Accuracy': ['65.75%', '67.33%', '71.11%', '71.11%'],
    'Gap to Paper':        ['-5.36%', '-3.78%', '0.00%',  'Reference'],
    'Gap Closed':          ['0%',     '29.5%',  '100%',   'Reference'],
    'DC Correction':       ['No',     'No',     'Yes',    'Yes'],
    'Proper Sent Boundaries': ['No',  'No',     'Yes',    'Yes'],
    'ModelBlocks':         ['No',     'Partial','Full',   'Full'],
}
gap_df = pd.DataFrame(gap_data)
gap_df.to_csv(os.path.join(RESULTS, 'table_gap_analysis.csv'), index=False)
print("\nTable 3: Gap Analysis")
print(gap_df.to_string(index=False))

# ── Table 4: Pipeline Comparison ──────────────────────────────
pipeline_data = {
    'Aspect': [
        'Disfluency removal before features',
        'Sentence boundaries',
        'Lexical surprisal',
        'DLT costs',
        'Syntactic surprisal',
        'Feature propagation to disfluent words',
        'Reparandum accuracy',
    ],
    'Old PCFG Pipeline': [
        'No',
        'Conversation level',
        'All words incl. fillers',
        'Full conversation — wrong arcs',
        'PCFG bigram approximation',
        'No',
        '65.75%',
    ],
    'ModelBlocks Partial': [
        'No',
        'Arbitrary 15-word chunks',
        'All words incl. fillers',
        'Full conversation — wrong arcs',
        'ModelBlocks — 4.6% coverage',
        'No',
        '67.33%',
    ],
    'Clean Pipeline': [
        'Yes — DC correction',
        'Switchboard NXT markers',
        'Clean sentences only',
        'Clean sentences only',
        'ModelBlocks — 71.7% coverage',
        'Yes — nearest clean word',
        '71.11%',
    ],
    'Paper': [
        'Yes',
        'Utterance level',
        'Clean text',
        'Stanford Parser on clean trees',
        'ModelBlocks — full coverage',
        'Yes',
        '71.11%',
    ],
}
pipeline_df = pd.DataFrame(pipeline_data)
pipeline_df.to_csv(os.path.join(RESULTS, 'table_pipeline_comparison.csv'), index=False)
print("\nTable 4: Pipeline Comparison")
print(pipeline_df.to_string(index=False))

# ── Generate Figure ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Speech Disfluency Prediction — Complete Pipeline Comparison',
             fontsize=15, fontweight='bold', y=1.01)

# Plot 1 — Reparandum accuracy all approaches
ax1 = axes[0, 0]
methods  = ['PCFG\n(old)', 'ModelBlocks\nPartial (old)', 'Clean\nPipeline (new)', 'Paper\nTarget']
baseline = [63.93, 63.65, 62.60, 66.29]
full_mod = [65.75, 67.33, 71.11, 71.11]
x = np.arange(len(methods))
w = 0.35
b1 = ax1.bar(x - w/2, baseline, w, label='Baseline', color=OCEAN,  alpha=0.7)
b2 = ax1.bar(x + w/2, full_mod, w, label='Full Model', color=CORAL, alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Reparandum Accuracy — All Approaches', fontsize=12, fontweight='bold')
ax1.set_ylim(55, 80)
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.bar_label(b1, fmt='%.1f%%', fontsize=8, padding=2)
ax1.bar_label(b2, fmt='%.1f%%', fontsize=8, padding=2)
ax1.axhline(71.11, color=GREEN, linestyle='--', linewidth=1.5,
            label='Paper target', alpha=0.7)

# Plot 2 — Gap to paper
ax2 = axes[0, 1]
methods2 = ['PCFG\n(old)', 'ModelBlocks\nPartial (old)', 'Clean\nPipeline (new)']
gaps     = [5.36, 3.78, 0.00]
colors2  = [OCEAN, CORAL, GREEN]
bars2    = ax2.bar(methods2, gaps, color=colors2, alpha=0.85, width=0.5)
ax2.set_ylabel('Gap to Paper Target (%)', fontsize=11)
ax2.set_title('Gap to Paper Target (71.11%)', fontsize=12, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.bar_label(bars2, fmt='%.2f%%', fontsize=11, padding=3)
ax2.set_ylim(0, 7)
ax2.axhline(0, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)

# Plot 3 — Key coefficients clean pipeline vs paper
ax3 = axes[1, 0]
feats    = ['prev_dur', 'foll_SC', 'prev_SC', 'foll_IC', 'foll_lex', 'prev_lex', 'prev_syn']
clean    = [+0.475, +0.755, -0.766, +0.188, +0.272, -0.268, +0.118]
pcfg     = [+0.348, +0.064, -0.029, -0.008, +0.210, -0.026, -0.165]
x3 = np.arange(len(feats))
ax3.bar(x3 - 0.2, pcfg,  0.35, label='PCFG (old)',      color=OCEAN, alpha=0.8)
ax3.bar(x3 + 0.2, clean, 0.35, label='Clean Pipeline',   color=GREEN, alpha=0.8)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xticks(x3)
ax3.set_xticklabels(feats, rotation=25, ha='right', fontsize=9)
ax3.set_ylabel('Coefficient Estimate (standardised)', fontsize=10)
ax3.set_title('Key Coefficients — Reparandum Model', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Plot 4 — Summary table
ax4 = axes[1, 1]
ax4.axis('off')
summary = [
    ['Metric',                'PCFG\n(old)', 'MB Partial\n(old)', 'Clean\nPipeline', 'Paper'],
    ['Reparandum Accuracy',   '65.75%',      '67.33%',            '71.11%',          '71.11%'],
    ['Gap to Paper',          '-5.36%',      '-3.78%',            '0.00%',           'Ref'],
    ['Gap Closed',            '0%',          '29.5%',             '100%',            'Ref'],
    ['DC Correction',         'No',          'No',                'Yes',             'Yes'],
    ['Sent. Boundaries',      'No',          'No',                'Yes',             'Yes'],
    ['ModelBlocks Coverage',  '0%',          '4.6%',              '71.7%',           '100%'],
    ['Feature Propagation',   'No',          'No',                'Yes',             'Yes'],
]
tbl = ax4.table(
    cellText=[r[1:] for r in summary[1:]],
    rowLabels=[r[0] for r in summary[1:]],
    colLabels=summary[0][1:],
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.3, 1.9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 2:
        cell.set_facecolor('#E8F5E9')
    elif col == -1:
        cell.set_facecolor('#F5F5F5')
ax4.set_title('Summary Comparison', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
out = os.path.join(RESULTS, 'figure_complete_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved figure -> {out}")

print("\n" + "=" * 60)
print("COMPLETE — All tables and figures saved")
print("=" * 60)
print("\nFiles saved:")
for f in ['table_accuracy_comparison.csv',
          'table_coefficient_comparison.csv',
          'table_gap_analysis.csv',
          'table_pipeline_comparison.csv',
          'figure_complete_comparison.png']:
    print(f"  results/{f}")