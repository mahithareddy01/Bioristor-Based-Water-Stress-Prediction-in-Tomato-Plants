from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    timestamp_col: str = "timestamp"
    plant_id_col: str = "plant_id"
    target_col: str = "status"
    feature_cols: tuple[str, ...] = ("Rds", "Delta_Igs", "tds", "tgs")
    class_order: tuple[str, ...] = ("Healthy", "Uncertain", "Stress", "Recovery")


@dataclass(frozen=True)
class LSTMConfig:
    past_hours: int = 6
    horizon_hours: int = 24
    resample_rule: str = "1h"
    rolling_mean_hours: int = 1  # smooth 15-min data into 1-hour mean
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 10
    lstm_units: int = 30
    learning_rate: float = 1e-3

