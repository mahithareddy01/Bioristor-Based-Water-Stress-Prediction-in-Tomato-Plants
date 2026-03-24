"""
Tomato Water Stress Detection — Streamlit Dashboard (v2 — fixed)
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
APP_TITLE    = "Tomato Water Stress Detection"
MODEL_PATH   = Path(__file__).parent / "rf_model.pkl"
DATASET_PATH = Path(__file__).parent / "data" / "raw" / "dataset.csv"
FEATURE_COLS = ["Rds", "Delta_Igs", "tds", "tgs"]
CLASS_ORDER  = ["Healthy", "Uncertain", "Stress", "Recovery"]
NAV_ITEMS    = ["Overview", "Predict", "Batch", "Explore"]


# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:       #0e1a12;
  --bg2:      #162019;
  --bg3:      #1e2e22;
  --bg4:      #243329;
  --border:   rgba(255,255,255,0.07);
  --green:    #3db96a;
  --green-d:  #2e9a57;
  --text:     #ddeee3;
  --muted:    rgba(221,238,227,0.45);
  --red:      #ef4444;
  --amber:    #f59e0b;
  --blue:     #60a5fa;
  --r:        10px;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* hide chrome */
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 99px; }

/* ── TOP HEADER ── */
.app-header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 40px;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 0;
}
.app-header-name {
  font-size: 17px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.3px;
}
.app-header-name span { color: var(--green); }

/* ── cards ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 26px;
  margin-bottom: 20px;
}
.label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--green);
  margin-bottom: 5px;
}
.card-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}
.card-sub {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.55;
}

/* ── stat grid ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 24px;
}
.stat-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
}
.stat-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.stat-value { font-size: 22px; font-weight: 800; color: var(--text); letter-spacing: -0.5px; }

/* ── param table ── */
.param-table { width: 100%; border-collapse: collapse; }
.param-table th {
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  padding: 0 10px 10px;
  border-bottom: 1px solid var(--border);
}
.param-table td {
  padding: 9px 10px;
  font-size: 13px;
  color: var(--text);
  border-bottom: 1px solid var(--border);
}
.param-table tr:last-child td { border-bottom: none; }
.param-table td:first-child { font-weight: 700; color: var(--green); }
.param-table td:nth-child(2) { color: var(--muted); font-size: 12px; }

/* ── irrigation rule rows ── */
.irr-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
}
.irr-row:last-child { border-bottom: none; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.irr-label { font-weight: 600; font-size: 13px; min-width: 80px; }
.irr-note { font-size: 12px; color: var(--muted); }

/* ── result card ── */
.result-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 22px;
}
.result-status-label {
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.result-status-value {
  font-size: 44px;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -1.5px;
}
.badge {
  padding: 10px 20px;
  border-radius: 99px;
  font-weight: 700;
  font-size: 13px;
  white-space: nowrap;
}

/* ── all buttons reset ── */
.stButton > button {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  height: 44px !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: var(--bg4) !important;
  border-color: rgba(255,255,255,0.14) !important;
}

/* primary button */
.btn-primary .stButton > button {
  background: var(--green) !important;
  border-color: transparent !important;
  color: #08130d !important;
  font-weight: 700 !important;
}
.btn-primary .stButton > button:hover { background: var(--green-d) !important; }

/* nav buttons */
.nav-host .stButton > button {
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--muted) !important;
  border-radius: 99px !important;
  height: 36px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}
.nav-host .stButton > button:hover {
  background: var(--bg3) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
.nav-active .stButton > button {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
  border-radius: 99px !important;
  height: 36px !important;
  font-size: 13px !important;
}

/* sliders */
div[data-testid="stSlider"] label {
  font-size: 11px !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}
div[data-testid="stSlider"] > div > div > div > div {
  background: var(--green) !important;
}

/* file uploader */
div[data-testid="stFileUploader"] section {
  background: var(--bg2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
}

/* download btn */
div[data-testid="stDownloadButton"] button {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 9px !important;
}

/* select/input */
div[data-baseweb="select"] div,
div[data-baseweb="input"] input {
  background: var(--bg3) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}

/* altair */
.vega-embed, .vega-embed canvas { background: transparent !important; }

/* hide metric widget */
div[data-testid="metric-container"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
</style>
"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_dataset_if_available(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    for c in FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[FEATURE_COLS].isna().any(axis=None):
        bad = out[out[FEATURE_COLS].isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(f"Non-numeric values at rows: {bad}")
    return out


def predict_one(model, rds, delta_igs, tds, tgs):
    x = np.array([[rds, delta_igs, tds, tgs]], dtype=float)
    pred = str(model.predict(x)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)[0]
        except Exception:
            pass
    return pred, proba


def predict_many(model, df: pd.DataFrame) -> pd.DataFrame:
    df2 = _coerce(df)
    x   = df2[FEATURE_COLS].to_numpy(dtype=float)
    out = df2.copy()
    out["prediction"] = model.predict(x).astype(str)
    out["irrigation"] = np.where(
        out["prediction"].str.lower() == "stress",
        "🚨 Irrigation Needed", "✅ Plant is Stable",
    )
    return out


def irr_badge(pred: str):
    if str(pred).strip().lower() == "stress":
        return "🚨 Irrigation Needed", "#ef4444"
    return "✅ Plant is Stable", "#3db96a"


def sc(s: str) -> str:
    return {"healthy": "#3db96a", "uncertain": "#f59e0b",
            "stress": "#ef4444", "recovery": "#60a5fa"}.get(str(s).strip().lower(), "#888")


def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Rds": 0.82, "Delta_Igs": 0.79, "tds": 12.6, "tgs": 15.5},
        {"Rds": 0.55, "Delta_Igs": 0.40, "tds": 18.2, "tgs": 20.1},
        {"Rds": 0.25, "Delta_Igs": 0.15, "tds": 22.0, "tgs": 24.5},
        {"Rds": 0.70, "Delta_Igs": 0.60, "tds": 10.0, "tgs": 12.0},
    ])


def _dark_theme():
    return {"config": {
        "background": "transparent",
        "axis": {"gridColor": "rgba(255,255,255,0.05)", "labelColor": "#7a9e85",
                 "titleColor": "#7a9e85", "domainColor": "rgba(255,255,255,0.07)",
                 "tickColor": "rgba(255,255,255,0.07)"},
        "legend": {"labelColor": "#7a9e85", "titleColor": "#7a9e85"},
        "view":   {"stroke": "transparent"},
        "title":  {"color": "#ddeee3"},
    }}

alt.themes.register("dg", _dark_theme)
alt.themes.enable("dg")


def chart_prob(proba, model):
    if proba is None:
        st.info("No probability output from this model.")
        return
    classes = [str(c) for c in getattr(model, "classes_", np.array(CLASS_ORDER))]
    df = pd.DataFrame({"status": classes, "prob": proba})
    df["col"] = df["status"].map(sc)
    st.altair_chart(
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("status:N", sort=classes, title=None),
            y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            color=alt.Color("col:N", scale=None, legend=None),
            tooltip=["status:N", alt.Tooltip("prob:Q", format=".3f")],
        ).properties(height=210),
        use_container_width=True,
    )


def chart_imp(model):
    est = model
    if hasattr(model, "named_steps") and "rf" in getattr(model, "named_steps", {}):
        est = model.named_steps["rf"]
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        st.info("Feature importance not available.")
        return
    df = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp}).sort_values("importance", ascending=False)
    st.altair_chart(
        alt.Chart(df).mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".4f")],
            color=alt.value("#3db96a"),
        ).properties(height=210),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon="🍅", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "tab" not in st.session_state:
    st.session_state.tab = 0


# ─── HEADER ───────────────────────────────────
st.markdown(
    """<div class="app-header">
         <span style="font-size:24px">🍅</span>
         <div class="app-header-name"><span>Tomato</span> WSD</div>
       </div>""",
    unsafe_allow_html=True,
)

# ─── NAV (centered pill row) ──────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

gap, n0, n1, n2, n3, gap2 = st.columns([2.5, 1, 1, 1, 1, 2.5])
for col, idx, label in [
    (n0, 0, "🏠  Overview"),
    (n1, 1, "🎯  Predict"),
    (n2, 2, "📄  Batch"),
    (n3, 3, "📊  Explore"),
]:
    with col:
        div_class = "nav-active" if st.session_state.tab == idx else "nav-host"
        st.markdown(f'<div class="{div_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{idx}", use_container_width=True):
            st.session_state.tab = idx
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ─── PAGE BODY ────────────────────────────────
_, body, _ = st.columns([1, 10, 1])
with body:

    if not MODEL_PATH.exists():
        st.markdown(
            "<div class='card'><div class='label'>setup needed</div>"
            "<div class='card-title'>Model file not found</div>"
            "<div class='card-sub'>Place <code>rf_model.pkl</code> next to <code>app.py</code>.</div></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    model   = load_model(MODEL_PATH)
    dataset = load_dataset_if_available(DATASET_PATH)
    tab     = st.session_state.tab

    # ══════════════════════════════════════
    # OVERVIEW
    # ══════════════════════════════════════
    if tab == 0:
        st.markdown(
            """<div class='card'>
              <div class='label'>about this dashboard</div>
              <div class='card-title'>Sensor-based water stress classification</div>
              <div class='card-sub'>Predicts tomato plant water status from bioristor-derived features:
                <strong>Rds</strong>, <strong>Delta_Igs</strong>, <strong>tds</strong>, <strong>tgs</strong>.
                Use the tabs above to make predictions, run batch scoring, or explore the dataset.</div>
            </div>""",
            unsafe_allow_html=True,
        )

        ds = "Yes" if dataset is not None else "No"
        st.markdown(
            f"""<div class='stat-grid'>
              <div class='stat-card'><div class='stat-label'>Model</div>
                <div class='stat-value' style='font-size:17px'>Random Forest</div></div>
              <div class='stat-card'><div class='stat-label'>Classes</div>
                <div class='stat-value'>4</div></div>
              <div class='stat-card'><div class='stat-label'>Smart irrigation</div>
                <div class='stat-value' style='color:var(--green)'>Enabled</div></div>
              <div class='stat-card'><div class='stat-label'>Dataset loaded</div>
                <div class='stat-value'>{ds}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                "<div class='label'>model explainability</div>"
                "<div style='font-size:15px;font-weight:700;margin-bottom:12px;color:var(--text)'>Feature importances</div>",
                unsafe_allow_html=True,
            )
            chart_imp(model)

        with right:
            st.markdown(
                "<div class='label'>input ranges</div>"
                "<div style='font-size:15px;font-weight:700;margin-bottom:12px;color:var(--text)'>Parameter guide</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """<table class='param-table'>
                  <thead><tr><th>Feature</th><th>Range</th><th>Description</th></tr></thead>
                  <tbody>
                    <tr><td>Rds</td><td>0 – 1</td><td>Drain-source resistance ratio</td></tr>
                    <tr><td>Delta_Igs</td><td>0 – 1</td><td>Gate-source current delta</td></tr>
                    <tr><td>tds</td><td>0 – 30</td><td>Drain stress time (s)</td></tr>
                    <tr><td>tgs</td><td>0 – 30</td><td>Gate stress time (s)</td></tr>
                  </tbody>
                </table>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='label'>irrigation rule</div>", unsafe_allow_html=True)
            rules = [
                ("Healthy",   "#3db96a", "✅ Stable"),
                ("Uncertain", "#f59e0b", "✅ Stable"),
                ("Stress",    "#ef4444", "🚨 Irrigate"),
                ("Recovery",  "#60a5fa", "✅ Stable"),
            ]
            rows = "".join(
                f"<div class='irr-row'><div class='dot' style='background:{c}'></div>"
                f"<span class='irr-label' style='color:{c}'>{l}</span>"
                f"<span class='irr-note'>{n}</span></div>"
                for l, c, n in rules
            )
            st.markdown(
                f"<div class='card' style='padding:14px 18px;margin-top:6px'>{rows}</div>",
                unsafe_allow_html=True,
            )


    # ══════════════════════════════════════
    # PREDICT
    # ══════════════════════════════════════
    elif tab == 1:
        st.markdown(
            """<div class='card'>
              <div class='label'>single prediction</div>
              <div class='card-title'>Adjust sensor values</div>
              <div class='card-sub'>Move the sliders then hit Predict to classify the current water stress status.</div>
            </div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1: rds       = st.slider("Rds",       0.0, 1.0,  0.80, 0.01)
        with c2: delta_igs = st.slider("Delta_Igs", 0.0, 1.0,  0.75, 0.01)
        with c3: tds_v     = st.slider("tds",       0.0, 30.0, 12.0, 0.1)
        with c4: tgs_v     = st.slider("tgs",       0.0, 30.0, 15.0, 0.1)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        b1, b2, _ = st.columns([1.2, 1.2, 4])
        with b1:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            do_predict = st.button("✨  Predict", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            if st.button("🧪  Fill sample", use_container_width=True):
                s = sample_df().iloc[0].to_dict()
                st.session_state["_sf"] = s
                st.rerun()

        if "_sf" in st.session_state:
            s = st.session_state.pop("_sf")
            rds, delta_igs, tds_v, tgs_v = float(s["Rds"]), float(s["Delta_Igs"]), float(s["tds"]), float(s["tgs"])

        if do_predict:
            pred, proba = predict_one(model, rds, delta_igs, tds_v, tgs_v)
            bt, bc = irr_badge(pred)
            color = sc(pred)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class='result-card'>
                  <div>
                    <div class='result-status-label'>predicted status</div>
                    <div class='result-status-value' style='color:{color}'>{pred}</div>
                  </div>
                  <div class='badge' style='background:{bc}18;border:1px solid {bc}44;color:{bc}'>{bt}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("<div class='label'>prediction probabilities</div>", unsafe_allow_html=True)
                chart_prob(proba, model)
            with p2:
                st.markdown("<div class='label'>input summary</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""<table class='param-table'>
                      <thead><tr><th>Feature</th><th>Value</th></tr></thead>
                      <tbody>
                        <tr><td>Rds</td><td>{rds:.2f}</td></tr>
                        <tr><td>Delta_Igs</td><td>{delta_igs:.2f}</td></tr>
                        <tr><td>tds</td><td>{tds_v:.1f}</td></tr>
                        <tr><td>tgs</td><td>{tgs_v:.1f}</td></tr>
                      </tbody>
                    </table>""",
                    unsafe_allow_html=True,
                )


    # ══════════════════════════════════════
    # BATCH
    # ══════════════════════════════════════
    elif tab == 2:
        st.markdown(
            """<div class='card'>
              <div class='label'>batch scoring</div>
              <div class='card-title'>Upload a CSV file</div>
              <div class='card-sub'>Required columns: <code>Rds</code>, <code>Delta_Igs</code>,
                <code>tds</code>, <code>tgs</code>. Extra columns are preserved in the output.</div>
            </div>""",
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2, gap="medium")
        with b1:
            if st.button("📌  Show sample table", use_container_width=True):
                st.dataframe(sample_df(), use_container_width=True, hide_index=True)
        with b2:
            st.download_button(
                "⬇️  Download sample CSV",
                data=sample_df().to_csv(index=False).encode(),
                file_name="sample_tomato_inputs.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.markdown("<div class='label'>preview</div>", unsafe_allow_html=True)
                st.dataframe(df_up.head(30), use_container_width=True)

                st.markdown('<div class="btn-primary" style="margin-top:10px">', unsafe_allow_html=True)
                run = st.button("🚀  Run batch prediction", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if run:
                    res = predict_many(model, df_up)
                    st.success(f"✅  Scored {len(res):,} rows successfully.")

                    sc1, sc2 = st.columns(2, gap="large")
                    counts = res["prediction"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
                    with sc1:
                        st.markdown("<div class='label'>prediction counts</div>", unsafe_allow_html=True)
                        st.dataframe(
                            counts.rename_axis("status").reset_index(name="count"),
                            use_container_width=True, hide_index=True,
                        )
                    with sc2:
                        st.markdown("<div class='label'>irrigation decisions</div>", unsafe_allow_html=True)
                        irr = res["irrigation"].value_counts()
                        st.dataframe(
                            irr.rename_axis("decision").reset_index(name="count"),
                            use_container_width=True, hide_index=True,
                        )

                    st.markdown(
                        "<div class='label' style='margin-top:20px'>full results</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(res, use_container_width=True)
                    st.download_button(
                        "⬇️  Download results CSV",
                        data=res.to_csv(index=False).encode(),
                        file_name="tomato_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not process file: {e}")


    # ══════════════════════════════════════
    # EXPLORE
    # ══════════════════════════════════════
    elif tab == 3:
        st.markdown(
            """<div class='card'>
              <div class='label'>exploratory data analysis</div>
              <div class='card-title'>Dataset insights</div>
              <div class='card-sub'>Interactive charts from your training dataset.</div>
            </div>""",
            unsafe_allow_html=True,
        )

        if dataset is None:
            st.info(f"Dataset not found at `{DATASET_PATH}`. Add it to unlock EDA charts.")
        else:
            cols_miss = [c for c in FEATURE_COLS + ["status"] if c not in dataset.columns]
            if cols_miss:
                st.error(f"Missing columns: {cols_miss}")
            else:
                d = dataset.copy()
                d["status"] = d["status"].astype(str)
                has_pid = "plant_id"  in d.columns
                has_ts  = "timestamp" in d.columns

                st.markdown(
                    f"""<div class='stat-grid'>
                      <div class='stat-card'><div class='stat-label'>Rows</div>
                        <div class='stat-value'>{len(d):,}</div></div>
                      <div class='stat-card'><div class='stat-label'>Plants</div>
                        <div class='stat-value'>{d['plant_id'].nunique() if has_pid else '—'}</div></div>
                      <div class='stat-card'><div class='stat-label'>Start</div>
                        <div class='stat-value' style='font-size:14px'>{str(d['timestamp'].iloc[0]) if has_ts else '—'}</div></div>
                      <div class='stat-card'><div class='stat-label'>End</div>
                        <div class='stat-value' style='font-size:14px'>{str(d['timestamp'].iloc[-1]) if has_ts else '—'}</div></div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("<div class='label'>class distribution</div>", unsafe_allow_html=True)
                dist = d["status"].value_counts().reindex(CLASS_ORDER).fillna(0).reset_index()
                dist.columns = ["status", "count"]
                dist["col"] = dist["status"].map(sc)
                st.altair_chart(
                    alt.Chart(dist)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X("status:N", sort=CLASS_ORDER, title=None),
                        y=alt.Y("count:Q", title="Count"),
                        color=alt.Color("col:N", scale=None, legend=None),
                        tooltip=["status:N", "count:Q"],
                    ).properties(height=240),
                    use_container_width=True,
                )

                st.markdown("<div class='label'>feature distributions</div>", unsafe_allow_html=True)
                fcol = st.selectbox("Select feature", FEATURE_COLS)
                dd = d[[fcol, "status"]].copy()
                st.altair_chart(
                    alt.Chart(dd)
                    .transform_density(fcol, groupby=["status"], as_=[fcol, "density"])
                    .mark_area(opacity=0.28)
                    .encode(
                        x=alt.X(f"{fcol}:Q", title=fcol),
                        y=alt.Y("density:Q", title="Density"),
                        color=alt.Color("status:N", sort=CLASS_ORDER),
                    ).properties(height=240),
                    use_container_width=True,
                )

                st.markdown("<div class='label'>feature correlation (pearson)</div>", unsafe_allow_html=True)
                corr_m = (
                    d[FEATURE_COLS].corr(numeric_only=True)
                    .reset_index()
                    .melt(id_vars="index", var_name="fy", value_name="corr")
                    .rename(columns={"index": "fx"})
                )
                st.altair_chart(
                    alt.Chart(corr_m).mark_rect().encode(
                        x=alt.X("fy:N", title=None, sort=FEATURE_COLS),
                        y=alt.Y("fx:N", title=None, sort=FEATURE_COLS),
                        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                        tooltip=["fx:N", "fy:N", alt.Tooltip("corr:Q", format=".2f")],
                    ).properties(height=280),
                    use_container_width=True,
                )

                if has_ts and has_pid:
                    st.markdown("<div class='label'>time-series view</div>", unsafe_allow_html=True)
                    ts = d.copy()
                    ts["timestamp"] = pd.to_datetime(ts["timestamp"], errors="coerce")
                    ts  = ts.dropna(subset=["timestamp"])
                    pid = st.selectbox("Plant ID", sorted(ts["plant_id"].unique().tolist()))
                    tdf = ts[ts["plant_id"] == pid].sort_values("timestamp")
                    v   = st.multiselect("Features", FEATURE_COLS, default=["Rds", "Delta_Igs"])
                    if v:
                        tlong = tdf.melt(id_vars=["timestamp"], value_vars=v,
                                         var_name="feature", value_name="value")
                        st.altair_chart(
                            alt.Chart(tlong).mark_line().encode(
                                x=alt.X("timestamp:T"),
                                y=alt.Y("value:Q"),
                                color=alt.Color("feature:N"),
                                tooltip=["timestamp:T", "feature:N",
                                         alt.Tooltip("value:Q", format=".4f")],
                            ).properties(height=280),
                            use_container_width=True,
                        )
                    st.markdown("<div class='label'>raw rows</div>", unsafe_allow_html=True)
                    st.dataframe(tdf.head(50), use_container_width=True)