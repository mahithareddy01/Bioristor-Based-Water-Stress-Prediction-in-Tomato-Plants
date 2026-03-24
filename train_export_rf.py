from __future__ import annotations

"""
Train + export RandomForest model to rf_model.pkl.

This script trains a scikit-learn Pipeline:
Imputer (median) -> StandardScaler (z-score) -> RandomForestClassifier

The exported artifact can be loaded directly in Streamlit with:
    model = joblib.load("rf_model.pkl")
    model.predict(X)
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw" / "dataset.csv"
OUT = ROOT / "rf_model.pkl"

FEATURES = ["Rds", "Delta_Igs", "tds", "tgs"]
TARGET = "status"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    x = df[FEATURES]
    y = df[TARGET].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # z-score normalization
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Holdout accuracy: {acc:.4f}")

    joblib.dump(model, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

