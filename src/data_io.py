from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


_LABEL_CANON = {
    "healthy": "Healthy",
    "uncertain": "Uncertain",
    "stress": "Stress",
    "stressed": "Stress",
    "recovery": "Recovery",
}


def load_dataset(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # Basic column checks
    required = set(cfg.feature_cols) | {cfg.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset missing required columns: "
            + ", ".join(sorted(missing))
            + f". Found columns: {list(df.columns)}"
        )

    # Parse timestamp if present
    if cfg.timestamp_col in df.columns:
        df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")

    # Normalize labels to project convention
    df[cfg.target_col] = (
        df[cfg.target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: _LABEL_CANON.get(x, x))
    )

    return df


def ensure_out_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out

