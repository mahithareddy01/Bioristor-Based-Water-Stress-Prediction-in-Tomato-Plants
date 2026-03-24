from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import DataConfig


@dataclass
class PreprocessArtifacts:
    imputer: SimpleImputer
    scaler: StandardScaler


def split_xy(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    x = df.loc[:, list(cfg.feature_cols)].copy()
    y = df.loc[:, cfg.target_col].copy()
    return x, y


def fit_transform_train(
    x_train: pd.DataFrame,
) -> tuple[np.ndarray, PreprocessArtifacts]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)  # z-score

    x_imp = imputer.fit_transform(x_train)
    x_scaled = scaler.fit_transform(x_imp)
    return x_scaled, PreprocessArtifacts(imputer=imputer, scaler=scaler)


def transform(
    x: pd.DataFrame, artifacts: PreprocessArtifacts
) -> np.ndarray:
    x_imp = artifacts.imputer.transform(x)
    x_scaled = artifacts.scaler.transform(x_imp)
    return x_scaled

