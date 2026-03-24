from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import DataConfig


def run_eda(df: pd.DataFrame, out_dir: Path, cfg: DataConfig) -> None:
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution
    plt.figure(figsize=(7, 4))
    df[cfg.target_col].value_counts(dropna=False).reindex(cfg.class_order).plot(kind="bar")
    plt.title("Class distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Feature histograms
    for col in cfg.feature_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=40)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(eda_dir / f"hist_{col}.png", dpi=150)
        plt.close()

    # Boxplots by class
    for col in cfg.feature_cols:
        plt.figure(figsize=(8, 4))
        df.boxplot(column=col, by=cfg.target_col)
        plt.title(f"Boxplot of {col} by class")
        plt.suptitle("")
        plt.xlabel("Status")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(eda_dir / f"box_{col}_by_class.png", dpi=150)
        plt.close()

    # Correlation heatmap (matplotlib-only)
    corr = df[list(cfg.feature_cols)].corr(numeric_only=True)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(corr.values, interpolation="nearest")
    plt.title("Feature correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(eda_dir / "feature_correlation.png", dpi=150)
    plt.close()

