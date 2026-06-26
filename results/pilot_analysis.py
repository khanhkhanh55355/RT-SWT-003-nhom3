import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure plotting
sns.set_theme(style="whitegrid")

# Paths (adjust if your working dir differs)
GT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'pilot_ground_truth.csv'))
LLM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pilot_llm_output.csv'))

print('Looking for:')
print('  Ground truth:', GT_PATH)
print('  LLM output:  ', LLM_PATH)

missing = []
if not os.path.exists(GT_PATH):
    missing.append(GT_PATH)
if not os.path.exists(LLM_PATH):
    missing.append(LLM_PATH)

if missing:
    print('\nERROR: Missing files. Found no CSVs at the expected locations:')
    for p in missing:
        print(' -', p)
    print('\nPlease place `pilot_ground_truth.csv` in `data/` (one level up) and `pilot_llm_output.csv` in `results/`, then re-run this script.')
    raise SystemExit(1)

# Read files
df_gt = pd.read_csv(GT_PATH)
df_llm = pd.read_csv(LLM_PATH)

# Ensure 'id' column exists
if 'id' not in df_gt.columns or 'id' not in df_llm.columns:
    print('\nERROR: `id` column is required in both CSVs to merge records.')
    print('GT columns:', df_gt.columns.tolist())
    print('LLM columns:', df_llm.columns.tolist())
    raise SystemExit(1)

# Merge
df_analysis = pd.merge(df_gt[['id', 'ground_truth']], df_llm[['id', 'llm_output']], on='id', how='inner')
print(f"Total pilot samples merged: {len(df_analysis)}")
print(df_analysis.head().to_string(index=False))

# Optionally save merged file
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pilot_merged_analysis.csv'))
df_analysis.to_csv(OUT_PATH, index=False)
print('\nMerged file written to', OUT_PATH)

# If you want a quick plot of lengths
try:
    df_analysis['gt_len'] = df_analysis['ground_truth'].astype(str).map(len)
    df_analysis['llm_len'] = df_analysis['llm_output'].astype(str).map(len)
    plt.figure(figsize=(8,4))
    sns.kdeplot(df_analysis['gt_len'], label='ground_truth')
    sns.kdeplot(df_analysis['llm_len'], label='llm_output')
    plt.legend()
    plt.title('Length distribution: GT vs LLM (pilot)')
    plt.tight_layout()
    PLOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pilot_lengths.png'))
    plt.savefig(PLOT_PATH)
    print('Saved quick plot to', PLOT_PATH)
except Exception as e:
    print('Plotting skipped due to error:', e)

# End
print('\nDone.')
