from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IrrigationDecision:
    predicted_status: str
    irrigate: bool


def decide_irrigation(predicted_status: str) -> IrrigationDecision:
    status = str(predicted_status).strip().lower()
    irrigate = status == "stress"
    return IrrigationDecision(predicted_status=predicted_status, irrigate=irrigate)


def save_decisions(decisions: list[IrrigationDecision], out_dir: Path, filename: str = "decisions.csv") -> Path:
    out_dir = out_dir / "irrigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"predicted_status": d.predicted_status, "irrigate": bool(d.irrigate)} for d in decisions]
    )
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path

