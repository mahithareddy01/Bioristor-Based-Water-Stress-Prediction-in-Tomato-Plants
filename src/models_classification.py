from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import DataConfig
from .evaluation import compute_metrics, save_confusion_matrix
from .preprocess import fit_transform_train, split_xy, transform


@dataclass(frozen=True)
class ClassificationResults:
    dt_metrics: dict
    rf_metrics: dict


def _train_test_split(
    df: pd.DataFrame,
    cfg: DataConfig,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # If plant_id exists, prefer group split to reduce leakage across time for the same plant.
    if cfg.plant_id_col in df.columns and df[cfg.plant_id_col].notna().any():
        groups = df[cfg.plant_id_col].astype(str).fillna("NA")
        n_groups = groups.nunique(dropna=False)
        if n_groups >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            (train_idx, test_idx) = next(splitter.split(df, groups=groups))
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[cfg.target_col],
    )
    return train_df.copy(), test_df.copy()


def train_and_evaluate_classifiers(df: pd.DataFrame, out_dir: Path, cfg: DataConfig) -> ClassificationResults:
    cls_dir = out_dir / "classification"
    cls_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _train_test_split(df, cfg)
    x_train, y_train = split_xy(train_df, cfg)
    x_test, y_test = split_xy(test_df, cfg)

    x_train_np, artifacts = fit_transform_train(x_train)
    x_test_np = transform(x_test, artifacts)

    labels = list(cfg.class_order)

    dt = DecisionTreeClassifier(random_state=42, criterion="entropy")  # info gain close to the paper
    dt.fit(x_train_np, y_train)
    dt_pred = dt.predict(x_test_np)
    dt_metrics = compute_metrics(y_test, dt_pred, labels=labels)
    save_confusion_matrix(
        y_test, dt_pred, labels=labels, out_path=cls_dir / "confusion_dt.png", title="Decision Tree (normalized)"
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x_train_np, y_train)
    rf_pred = rf.predict(x_test_np)
    rf_metrics = compute_metrics(y_test, rf_pred, labels=labels)
    save_confusion_matrix(
        y_test, rf_pred, labels=labels, out_path=cls_dir / "confusion_rf.png", title="Random Forest (normalized)"
    )

    (cls_dir / "metrics_dt.txt").write_text(
        "\n".join(
            [
                f"accuracy: {dt_metrics['accuracy']:.4f}",
                f"precision_weighted: {dt_metrics['precision_weighted']:.4f}",
                f"recall_weighted: {dt_metrics['recall_weighted']:.4f}",
                f"f1_weighted: {dt_metrics['f1_weighted']:.4f}",
                "",
                dt_metrics["report"],
            ]
        ),
        encoding="utf-8",
    )
    (cls_dir / "metrics_rf.txt").write_text(
        "\n".join(
            [
                f"accuracy: {rf_metrics['accuracy']:.4f}",
                f"precision_weighted: {rf_metrics['precision_weighted']:.4f}",
                f"recall_weighted: {rf_metrics['recall_weighted']:.4f}",
                f"f1_weighted: {rf_metrics['f1_weighted']:.4f}",
                "",
                rf_metrics["report"],
            ]
        ),
        encoding="utf-8",
    )

    return ClassificationResults(dt_metrics=dt_metrics, rf_metrics=rf_metrics)

