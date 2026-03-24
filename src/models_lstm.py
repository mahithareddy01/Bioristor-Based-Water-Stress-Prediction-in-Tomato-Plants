from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from .config import DataConfig, LSTMConfig
from .evaluation import compute_metrics, save_confusion_matrix


@dataclass(frozen=True)
class LSTMResults:
    metrics: dict
    history: dict


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"LSTM pipeline requires columns {missing}. Found: {list(df.columns)}")


def _make_hourly(df: pd.DataFrame, cfg: DataConfig, lstm_cfg: LSTMConfig) -> pd.DataFrame:
    _require_cols(df, [cfg.timestamp_col, cfg.plant_id_col])
    d = df.copy()
    d = d.dropna(subset=[cfg.timestamp_col])
    d = d.sort_values([cfg.plant_id_col, cfg.timestamp_col])

    # Use 1-hour rolling mean from 15-min readings if present; else a 1h rolling on whatever is present.
    win = max(int(lstm_cfg.rolling_mean_hours * 60 / 15), 1)

    hourly_parts: list[pd.DataFrame] = []
    for pid, g in d.groupby(cfg.plant_id_col):
        g = g.set_index(cfg.timestamp_col)
        g_feat = g[list(cfg.feature_cols)].rolling(window=win, min_periods=1).mean()
        g_hour = g_feat.resample(lstm_cfg.resample_rule).mean()
        g_hour[cfg.target_col] = g[cfg.target_col].resample(lstm_cfg.resample_rule).ffill()
        g_hour[cfg.plant_id_col] = pid
        g_hour = g_hour.dropna(subset=[cfg.target_col])
        hourly_parts.append(g_hour.reset_index())

    hourly = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else pd.DataFrame()
    return hourly


def _build_sequences(
    hourly: pd.DataFrame,
    cfg: DataConfig,
    lstm_cfg: LSTMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # Create sequences of length past_hours to predict t + horizon_hours
    x_list: list[np.ndarray] = []
    y_list: list[str] = []

    hourly = hourly.sort_values([cfg.plant_id_col, cfg.timestamp_col])
    for _, g in hourly.groupby(cfg.plant_id_col):
        g = g.reset_index(drop=True)
        feats = g[list(cfg.feature_cols)].to_numpy(dtype=float)
        labels = g[cfg.target_col].astype(str).to_numpy()

        for t in range(lstm_cfg.past_hours - 1, len(g) - lstm_cfg.horizon_hours):
            x_seq = feats[t - (lstm_cfg.past_hours - 1) : t + 1]
            y = labels[t + lstm_cfg.horizon_hours]
            if np.any(~np.isfinite(x_seq)):
                continue
            x_list.append(x_seq)
            y_list.append(y)

    if not x_list:
        raise ValueError(
            "No sequences could be built. Check timestamp frequency, plant_id, and that there is enough history "
            f"(needs >= past_hours + horizon_hours = {lstm_cfg.past_hours + lstm_cfg.horizon_hours} hours per plant)."
        )

    x = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=str)
    return x, y


def _encode_labels(y: np.ndarray, class_order: tuple[str, ...]) -> tuple[np.ndarray, dict[str, int]]:
    idx = {c: i for i, c in enumerate(class_order)}
    y_idx = np.array([idx.get(v, -1) for v in y], dtype=int)
    if (y_idx < 0).any():
        bad = sorted(set(y[y_idx < 0].tolist()))
        raise ValueError(f"Found unknown labels in data: {bad}. Expected: {list(class_order)}")
    y_oh = keras.utils.to_categorical(y_idx, num_classes=len(class_order))
    return y_oh, idx


def _build_model(input_shape: tuple[int, int], n_classes: int, lstm_cfg: LSTMConfig) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.LSTM(
        lstm_cfg.lstm_units,
        activation="relu",
        recurrent_activation="sigmoid",
    )(inputs)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=lstm_cfg.learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_lstm_forecaster(
    df: pd.DataFrame,
    out_dir: Path,
    cfg: DataConfig,
    lstm_cfg: LSTMConfig,
) -> LSTMResults:
    fore_dir = out_dir / "forecasting"
    fore_dir.mkdir(parents=True, exist_ok=True)

    hourly = _make_hourly(df, cfg, lstm_cfg)
    x, y = _build_sequences(hourly, cfg, lstm_cfg)

    # z-score normalization: fit on train only, per feature (flatten time dimension)
    n_samples, t_steps, n_feats = x.shape
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_samples)
    split = int(0.8 * n_samples)
    tr_idx, va_idx = idx[:split], idx[split:]

    x_train, x_val = x[tr_idx], x[va_idx]
    y_train_raw, y_val_raw = y[tr_idx], y[va_idx]

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, n_feats)
    x_val_2d = x_val.reshape(-1, n_feats)
    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(-1, t_steps, n_feats)
    x_val_scaled = scaler.transform(x_val_2d).reshape(-1, t_steps, n_feats)

    y_train_oh, label_to_idx = _encode_labels(y_train_raw, cfg.class_order)
    y_val_oh, _ = _encode_labels(y_val_raw, cfg.class_order)

    # Class weights to handle imbalance
    y_train_idx = np.argmax(y_train_oh, axis=1)
    classes = np.arange(len(cfg.class_order))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_idx)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = _build_model(input_shape=(t_steps, n_feats), n_classes=len(cfg.class_order), lstm_cfg=lstm_cfg)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=lstm_cfg.patience, restore_best_weights=True),
    ]

    hist = model.fit(
        x_train_scaled,
        y_train_oh,
        validation_data=(x_val_scaled, y_val_oh),
        epochs=lstm_cfg.max_epochs,
        batch_size=lstm_cfg.batch_size,
        class_weight=class_weight,
        verbose=0,
    )

    # Plots
    plt.figure(figsize=(7, 4))
    plt.plot(hist.history.get("loss", []), label="train")
    plt.plot(hist.history.get("val_loss", []), label="val")
    plt.title("LSTM loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fore_dir / "lstm_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(hist.history.get("accuracy", []), label="train")
    plt.plot(hist.history.get("val_accuracy", []), label="val")
    plt.title("LSTM accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fore_dir / "lstm_accuracy.png", dpi=150)
    plt.close()

    # Evaluation on validation set
    y_val_prob = model.predict(x_val_scaled, verbose=0)
    y_val_pred_idx = np.argmax(y_val_prob, axis=1)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    y_val_pred = np.array([idx_to_label[i] for i in y_val_pred_idx], dtype=object)

    labels = list(cfg.class_order)
    metrics = compute_metrics(y_val_raw, y_val_pred, labels=labels)
    save_confusion_matrix(
        y_val_raw,
        y_val_pred,
        labels=labels,
        out_path=fore_dir / "confusion_lstm_24h.png",
        title="LSTM 24h-ahead (normalized)",
    )

    (fore_dir / "metrics_lstm.txt").write_text(
        "\n".join(
            [
                f"accuracy: {metrics['accuracy']:.4f}",
                f"precision_weighted: {metrics['precision_weighted']:.4f}",
                f"recall_weighted: {metrics['recall_weighted']:.4f}",
                f"f1_weighted: {metrics['f1_weighted']:.4f}",
                "",
                metrics["report"],
            ]
        ),
        encoding="utf-8",
    )

    return LSTMResults(metrics=metrics, history=hist.history)

