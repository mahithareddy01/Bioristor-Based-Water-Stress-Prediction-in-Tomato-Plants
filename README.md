# Tomato water-stress classification + forecasting (sensor data)

This project implements:
- **EDA**
- **Preprocessing** (missing values + **z-score normalization**)
- **Classification**: Decision Tree, Random Forest
- **Evaluation**: accuracy, precision, recall, F1, confusion matrix (+ plots)
- **Forecasting**: LSTM using **past 6 hours** to predict **plant status 24 hours ahead**
- **Smart irrigation logic**: if stress predicted → trigger irrigation

## Expected dataset format

Put your dataset CSV at:
- `data/raw/dataset.csv`

Required columns:
- **features**: `Rds`, `Delta_Igs`, `tds`, `tgs`
- **target**: `status` with labels in `{Healthy, Uncertain, Stress, Recovery}`

Strongly recommended columns (for proper time-series building):
- `timestamp` (parseable datetime)
- `plant_id` (plant identifier)

If your label strings are lowercase (e.g. `healthy`) the code will map them automatically.

## Setup

Create a venv, then install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### 1) End-to-end pipeline

```bash
python main.py --data data/raw/dataset.csv --out outputs
```

### 2) Only classification (DT + RF)

```bash
python main.py --data data/raw/dataset.csv --out outputs --skip_lstm
```

### 3) Only LSTM forecasting

```bash
python main.py --data data/raw/dataset.csv --out outputs --skip_classification
```

## Outputs

Saved into `outputs/`:
- `eda/` plots (distributions, boxplots, correlations, class counts)
- `classification/` metrics + confusion matrices
- `forecasting/` training curves + confusion matrix (24h ahead)
- `irrigation/` simple “irrigate / no-irrigate” decisions

