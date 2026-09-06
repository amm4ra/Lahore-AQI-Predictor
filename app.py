"""
Lahore AQI Predictor — Streamlit dashboard.

Loads the three registered models (+24h/+48h/+72h) and the latest features from the
Hopsworks Feature Store, computes a 3-day AQI forecast, and shows EDA, a SHAP-based
explanation, and a hazard alert.

Run:  streamlit run app.py
Config (Streamlit secrets or env vars): HOPSWORKS_KEY (required), HOPSWORKS_PROJECT, HOPSWORKS_HOST
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

HORIZONS = [24, 48, 72]

st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌫️", layout="wide")


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def cfg(name, default=None, required=False):
    """Read config from Streamlit secrets first, then environment."""
    val = None
    try:
        val = st.secrets[name]
    except Exception:
        val = os.environ.get(name, default)
    if required and not val:
        st.error(f"Missing config: {name}. Set it in .streamlit/secrets.toml or as an env var.")
        st.stop()
    return val


def aqi_category(aqi):
    """US EPA AQI band -> (label, color)."""
    if aqi <= 50:  return "Good", "#00e400"
    if aqi <= 100: return "Moderate", "#cccc00"
    if aqi <= 150: return "Unhealthy (Sensitive)", "#ff7e00"
    if aqi <= 200: return "Unhealthy", "#ff0000"
    if aqi <= 300: return "Very Unhealthy", "#8f3f97"
    return "Hazardous", "#7e0023"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Connecting to Hopsworks…")
def connect():
    import hopsworks
    return hopsworks.login(
        project=cfg("HOPSWORKS_PROJECT", "project123456789"),
        host=cfg("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        port=443,
        api_key_value=cfg("HOPSWORKS_KEY", required=True),
    )


@st.cache_resource(ttl=3600, show_spinner="Loading models from the registry…")
def load_models(_project):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    mr = _project.get_model_registry()
    models = {}
    for h in HORIZONS:
        # get_model() with no version silently defaults to version=1, not the
        # latest — fetch all versions and take the highest instead.
        versions = mr.get_models(f"lahore_aqi_model_{h}h")
        m = max(versions, key=lambda v: v.version)
        d = m.download()
        scaler = joblib.load(os.path.join(d, "scaler.pkl"))
        feature_cols = joblib.load(os.path.join(d, "feature_cols.pkl"))

        pkl = os.path.join(d, "model.pkl")
        if os.path.exists(pkl):                          # sklearn / XGBoost (Ridge, RF, or XGBoost)
            model = joblib.load(pkl)
            if isinstance(model, RandomForestRegressor):
                kind = "rf"
            elif isinstance(model, XGBRegressor):
                kind = "xgb"
            else:
                kind = "ridge"
            art = {"model": model, "scaler": scaler, "feature_cols": feature_cols, "kind": kind}
        else:                                            # PyTorch MLP
            import torch
            import torch.nn as nn
            meta = joblib.load(os.path.join(d, "mlp_meta.pkl"))

            class MLPRegressor(nn.Module):
                def __init__(self, in_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
                    )
                def forward(self, x):
                    return self.net(x)

            model = MLPRegressor(meta["input_dim"])
            model.load_state_dict(torch.load(os.path.join(d, "model.pt")))
            model.eval()
            art = {"model": model, "scaler": scaler, "feature_cols": feature_cols,
                   "kind": "mlp", "y_mean": meta["y_mean"], "y_std": meta["y_std"]}

        art["metrics"] = getattr(m, "training_metrics", {}) or {}
        models[h] = art
    return models


@st.cache_data(ttl=3600, show_spinner="Reading features…")
def load_features(_project):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group("lahore_aqi_features", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict(art, X_row):
    model, scaler, kind = art["model"], art["scaler"], art["kind"]
    if kind in ("rf", "xgb"):                             # trained on raw features
        return float(model.predict(X_row)[0])
    Xs = scaler.transform(X_row)                         # ridge / mlp use scaled input
    if kind == "ridge":
        return float(model.predict(Xs)[0])
    import torch
    with torch.no_grad():
        p = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()[0]
    return float(p * art["y_std"] + art["y_mean"])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("🌫️ Lahore AQI Predictor")
st.caption("3-day US AQI forecast from Open-Meteo features, served via the Hopsworks Feature Store & Model Registry.")

project = connect()
models = load_models(project)
df = load_features(project)

# sidebar: model summary + refresh
st.sidebar.header("Models in use")
for h in HORIZONS:
    a = models[h]
    r2 = a["metrics"].get("r2", float("nan"))
    st.sidebar.write(f"**+{h}h** — {a['kind'].upper()}  ·  R² {r2:.2f}")
st.sidebar.caption("Best of Ridge (statistical), RandomForest / XGBoost (ensemble) and MLP (deep) per horizon.")
if st.sidebar.button("🔄 Refresh data & models"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# latest fully-populated feature row
feature_cols = models[HORIZONS[0]]["feature_cols"]
complete = df.dropna(subset=feature_cols)
if complete.empty:
    st.error("No feature rows with complete features in the store yet. Run the feature pipeline first.")
    st.stop()

latest = complete.iloc[-1]
X_row = complete.iloc[[-1]][feature_cols]
current_aqi = float(latest["us_aqi"])
latest_time = latest["time"]

preds = {h: predict(models[h], X_row) for h in HORIZONS}
peak = max([current_aqi] + list(preds.values()))

# hazard alert
peak_cat, _ = aqi_category(peak)
if peak > 200:
    st.error(f"🚨 **{peak_cat}** air expected in the next 3 days (peak AQI ≈ {peak:.0f}). Avoid outdoor exertion; keep windows closed.")
elif peak > 150:
    st.error(f"⚠️ **{peak_cat}** air expected (peak AQI ≈ {peak:.0f}). Sensitive groups should stay indoors.")
elif peak > 100:
    st.warning(f"**{peak_cat}** air expected (peak AQI ≈ {peak:.0f}). Sensitive groups take care.")
else:
    st.success(f"Air quality looks **{peak_cat}** over the next 3 days (peak AQI ≈ {peak:.0f}).")

tab_forecast, tab_eda, tab_explain = st.tabs(["📈 Forecast", "🔍 EDA", "🧠 Explanation"])

# ---- Forecast tab ----
with tab_forecast:
    cat, color = aqi_category(current_aqi)
    cols = st.columns(4)
    cols[0].metric("📍 Now (actual)", f"{current_aqi:.0f}", cat)
    for i, h in enumerate(HORIZONS):
        c, _ = aqi_category(preds[h])
        delta = preds[h] - current_aqi
        target_date = latest_time + pd.Timedelta(hours=h)
        cols[i + 1].metric(f"🔮 {target_date:%a, %b %d}", f"{preds[h]:.0f}", f"{delta:+.0f} vs now")
        cols[i + 1].caption(f"+{h}h · predicted")

    st.caption(f"Latest features from: {latest_time:%Y-%m-%d %H:%M}")

    st.subheader("Current conditions")
    pollutants = [
        ("PM2.5", "pm2_5", "µg/m³"),
        ("PM10", "pm10", "µg/m³"),
        ("CO", "carbon_monoxide", "µg/m³"),
        ("NO2", "nitrogen_dioxide", "µg/m³"),
        ("SO2", "sulphur_dioxide", "µg/m³"),
        ("O3", "ozone", "µg/m³"),
    ]
    for i in range(0, len(pollutants), 3):
        row = pollutants[i:i + 3]
        pcols = st.columns(3)
        for pcol, (label, key, unit) in zip(pcols, row):
            val = latest.get(key)
            pcol.metric(label, f"{val:.1f} {unit}" if pd.notna(val) else "—")
    st.caption(
        f"Weather: {latest['temperature_2m']:.1f}°C · {latest['relative_humidity_2m']:.0f}% humidity · "
        f"wind {latest['wind_speed_10m']:.1f} km/h"
    )

    st.subheader("AQI: last 24 hours (actual) → next 3 days (predicted)")
    hist = df[df["time"] <= latest_time].tail(24)[["time", "us_aqi"]].rename(columns={"us_aqi": "Actual"})
    future = pd.DataFrame({
        "time": [latest_time] + [latest_time + pd.Timedelta(hours=h) for h in HORIZONS],
        "Predicted": [current_aqi] + [preds[h] for h in HORIZONS],
    })
    fc = pd.merge(hist, future, on="time", how="outer").sort_values("time").set_index("time")
    st.line_chart(fc)

# ---- EDA tab ----
with tab_eda:
    st.subheader("AQI over time")
    st.line_chart(df.set_index("time")["us_aqi"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average AQI by hour of day")
        by_hour = df.groupby("hour")["us_aqi"].mean()
        st.bar_chart(by_hour)
    with c2:
        st.subheader("AQI trend — last 24 hours")
        last_24h = df.tail(24).set_index("time")["us_aqi"]
        st.line_chart(last_24h)

    st.subheader("PM2.5 vs AQI")
    st.scatter_chart(df, x="pm2_5", y="us_aqi")
    st.caption(f"History: {len(df):,} hourly rows from {df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}.")

# ---- Explanation tab ----
with tab_explain:
    h_sel = st.selectbox("Explain which horizon's prediction?", HORIZONS, format_func=lambda h: f"+{h}h")
    art = models[h_sel]
    st.write(f"Model: **{art['kind'].upper()}**  ·  predicted AQI **{preds[h_sel]:.0f}**")

    def native_importance():
        if art["kind"] in ("rf", "xgb"):
            imp = pd.Series(art["model"].feature_importances_, index=feature_cols)
        elif art["kind"] == "ridge":
            imp = pd.Series(np.abs(art["model"].coef_), index=feature_cols)
        else:
            return None
        return imp.sort_values(ascending=False).head(12)

    shown = False
    try:
        import shap
        bg = df.dropna(subset=feature_cols)[feature_cols].sample(
            min(100, len(complete)), random_state=0)
        if art["kind"] in ("rf", "xgb"):
            explainer = shap.TreeExplainer(art["model"])
            vals = explainer.shap_values(X_row)[0]
        elif art["kind"] == "ridge":
            bg_s = art["scaler"].transform(bg)
            explainer = shap.LinearExplainer(art["model"], bg_s)
            vals = explainer.shap_values(art["scaler"].transform(X_row))[0]
        else:
            vals = None
        if vals is not None:
            contrib = pd.Series(vals, index=feature_cols)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)
            st.subheader("SHAP contributions to this prediction")
            st.caption("Positive pushes the predicted AQI up; negative pulls it down.")
            st.bar_chart(top)
            shown = True
    except Exception as e:
        st.info(f"SHAP unavailable ({type(e).__name__}); showing the model's built-in importance instead.")

    if not shown:
        imp = native_importance()
        if imp is not None:
            st.subheader("Feature importance")
            st.bar_chart(imp)
