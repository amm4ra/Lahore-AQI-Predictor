# Lahore AQI Predictor — Dashboard

A Streamlit web app that loads the trained models and features from the Hopsworks
Feature Store / Model Registry, forecasts US AQI for the next 3 days, and shows
EDA, a SHAP-based explanation, and a hazard alert.

## What it does

- **Loads** the three registered models (`lahore_aqi_model_24h`, `_48h`, `_72h`) and the
  latest feature row from the feature store.
- **Predicts** AQI at +24h / +48h / +72h (one point per day) — RandomForest at 24h,
  Ridge at 48h and 72h (best per horizon).
- **Alerts** when the predicted peak crosses unhealthy/hazardous thresholds.
- **EDA tab** — AQI over time, average AQI by hour and by month, PM2.5 vs AQI.
- **Explanation tab** — SHAP contributions for the selected horizon (falls back to the
  model's built-in importance if SHAP isn't available).

## Setup

1. **Create an environment and install dependencies**

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Provide your Hopsworks config.** Create `.streamlit/secrets.toml` next to `app.py`:

   ```toml
   HOPSWORKS_KEY = "your_api_key_here"
   HOPSWORKS_PROJECT = "your_project_name"
   HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
   ```

   (Or set the same names as environment variables — the app checks secrets first, then env.)
   Get the API key from Hopsworks → Account Settings → API Keys. **Don't commit this file** —
   add `.streamlit/secrets.toml` to `.gitignore`.

3. **Run**

   ```bash
   streamlit run app.py
   ```

   It opens at http://localhost:8501.

## Notes

- **Predictions reflect the latest row in the feature store.** If it looks stale, run your
  incremental feature pipeline to add the current hour, then hit **Refresh data** in the sidebar.
- **Feature data is cached for 1 hour** (and models/connection for the session). The Refresh
  button clears the data cache.
- **`torch` is optional** — only needed if an MLP ends up registered as the best model for some
  horizon. Your current best-per-horizon set (RF/Ridge/Ridge) doesn't need it. If you retrain and
  an MLP wins, `pip install torch` and uncomment it in `requirements.txt`.
- **SHAP** uses `TreeExplainer` for RandomForest and `LinearExplainer` for Ridge. If SHAP fails
  to install or errors, the app automatically shows the model's native feature importance instead,
  so the dashboard still runs.

## Files

- `app.py` — the dashboard
- `requirements.txt` — dependencies
- `.streamlit/secrets.toml` — your credentials (create this yourself; do not commit)
