#!/usr/bin/env python3
"""Calculate metric + statistical test (gate E4)."""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_pilot_data():
    gt_path = os.path.join(DATA_DIR, "pilot_ground_truth.csv")
    llm_path = os.path.join(RESULTS_DIR, "pilot_llm_output.csv")
    missing = [p for p in (gt_path, llm_path) if not os.path.exists(p)]
    if missing:
        print("Missing files:")
        for path in missing:
            print(" -", path)
        sys.exit(1)
    df_gt = pd.read_csv(gt_path)
    df_llm = pd.read_csv(llm_path)
    if "id" not in df_gt.columns or "id" not in df_llm.columns:
        print("Both CSVs must include an `id` column.")
        sys.exit(1)
    return pd.merge(
        df_gt[["id", "ground_truth"]],
        df_llm[["id", "llm_output"]],
        on="id",
        how="inner",
    )


def load_full_data():
    gt_path = os.path.join(DATA_DIR, "full_ground_truth.csv")
    llm_path = os.path.join(RESULTS_DIR, "full_llm_output.csv")
    missing = [p for p in (gt_path, llm_path) if not os.path.exists(p)]
    if missing:
        print("Missing files:")
        for path in missing:
            print(" -", path)
        sys.exit(1)
    df_gt = pd.read_csv(gt_path)
    df_llm = pd.read_csv(llm_path)
    if df_gt.empty or df_gt["ground_truth"].isna().all():
        print("full_ground_truth.csv chưa có nhãn — chạy pilot trước hoặc điền ground truth.")
        sys.exit(1)
    return pd.merge(
        df_gt[["id", "ground_truth"]],
        df_llm[["id", "llm_output"]],
        on="id",
        how="inner",
    )


def exact_match(df: pd.DataFrame) -> float:
    matches = df["ground_truth"].astype(str) == df["llm_output"].astype(str)
    return matches.mean()


def save_distribution_figure(df: pd.DataFrame, filename: str, title: str):
    plot_df = df.copy()
    plot_df["gt_len"] = plot_df["ground_truth"].astype(str).map(len)
    plot_df["llm_len"] = plot_df["llm_output"].astype(str).map(len)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.kdeplot(data=plot_df, x="gt_len", label="ground_truth", ax=ax)
    sns.kdeplot(data=plot_df, x="llm_len", label="llm_output", ax=ax)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", out_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute metrics and statistical tests")
    parser.add_argument(
        "--scope",
        choices=("pilot", "full"),
        default="pilot",
        help="Dataset scope to evaluate",
    )
    args = parser.parse_args()

    df = load_pilot_data() if args.scope == "pilot" else load_full_data()
    if df.empty or df["ground_truth"].isna().all():
        print("Chưa có ground truth — điền `data/pilot_ground_truth.csv` rồi chạy lại.")
        sys.exit(1)

    print(f"Samples merged: {len(df)}")
    print(df.head().to_string(index=False))

    metric = exact_match(df)
    print(f"Exact match rate: {metric:.4f}")

    gt_len = df["ground_truth"].astype(str).map(len)
    llm_len = df["llm_output"].astype(str).map(len)
    if len(df) >= 2:
        stat, p_value = stats.wilcoxon(gt_len, llm_len)
        print(f"Wilcoxon stat={stat:.4f}, p={p_value:.6f}")

    if args.scope == "pilot":
        save_distribution_figure(
            df,
            "fig1_distribution.png",
            "Phân phối độ dài: Ground truth vs LLM (pilot)",
        )
    else:
        save_distribution_figure(
            df,
            "fig2_comparison.png",
            "So sánh Ground truth vs LLM (full)",
        )


if __name__ == "__main__":
    main()
