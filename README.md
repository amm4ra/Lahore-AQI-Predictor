# Lahore AQI Predictor

Predicts Lahore's US AQI 24/48/72 hours ahead. Weather + pollutant data is pulled from
Open-Meteo, engineered into features and stored in a Hopsworks Feature Store, used to train
and register per-horizon models (Ridge, RandomForest, XGBoost, MLP — best one wins per
horizon) in the Hopsworks Model Registry, and served through a Streamlit dashboard.

## Project structure

| File | Purpose |
|---|---|
| `backfill.ipynb` | One-time historical pull (2 years) from Open-Meteo, feature engineering, and initial write to the Hopsworks Feature Store. Also includes a "gap-fill" step that closes the ~5-day ERA5 archive lag by pulling recent days from Open-Meteo's live forecast endpoint instead, so the feature store's latest row reflects the actual current time when you re-run it. |
| `training-pipeline-multihorizon.ipynb` | Reads features from the Hopsworks Feature Store, trains Ridge / RandomForest / XGBoost / MLP for each horizon (+24h/+48h/+72h), compares them against a persistence baseline, and registers the best model per horizon in the Hopsworks Model Registry. |
| `app.py` | Streamlit dashboard: 3-day forecast (with dates), current conditions (pollutants + weather), EDA (AQI history, hourly pattern, last-24h trend), SHAP-based explanations, and hazard alerts. |
| `requirements.txt` | Pinned dependencies for both the notebooks and the app. |

## What the dashboard does

- **Forecast tab** — current AQI plus predicted AQI for +24h/+48h/+72h, each labeled with the
  actual calendar date and clearly marked as predicted; a line graph of the last 24 actual
  hours flowing into the 3-day forecast; a "Current conditions" panel (PM2.5, PM10, CO, NO2,
  SO2, O3, plus temperature/humidity/wind).
- **Hazard alert** — banner at the top when the predicted peak AQI crosses unhealthy/hazardous
  thresholds.
- **EDA tab** — AQI over the full history, average AQI by hour of day, AQI trend over the last
  24 hours, PM2.5 vs AQI scatter.
- **Explanation tab** — SHAP contributions for the selected horizon's prediction (falls back to
  the model's built-in feature importance if SHAP errors out).

## Setup

### 0. Prerequisites

- Python 3.11
- A free [Hopsworks](https://www.hopsworks.ai/) account and project, for the Feature Store and
  Model Registry.
- (Only if you want to re-run `backfill.ipynb` from scratch) no API key is needed for
  Open-Meteo itself — it's a free, keyless API.

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd "Internship - 10Pearls"

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure your Hopsworks credentials

Create `.streamlit/secrets.toml` next to `app.py` (this file is git-ignored — never commit it):

```toml
HOPSWORKS_KEY = "your_api_key_here"
HOPSWORKS_PROJECT = "your_project_name"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
```

Get the API key from Hopsworks → **Account Settings → API Keys**. The notebooks read the same
key from the `HOPSWORKS_KEY` environment variable instead (`os.environ["HOPSWORKS_KEY"]`), so
for those, `setx HOPSWORKS_KEY "your_key"` (Windows) or `export HOPSWORKS_KEY=your_key`
(macOS/Linux) before launching Jupyter.

### 3. Populate the Feature Store (first time only)

Run `backfill.ipynb` top to bottom. It creates the `lahore_aqi_features` feature group and
loads ~2 years of history. Its last cell (the gap-fill step) also closes the ERA5 lag so the
latest row reflects the current day — re-run just that notebook whenever you want fresher data,
until an automated pipeline exists to do it hourly.

### 4. Train and register models

Run `training-pipeline-multihorizon.ipynb` top to bottom. It trains all four model types per
horizon, prints a comparison table, and registers the best model per horizon
(`lahore_aqi_model_24h` / `_48h` / `_72h`) in the Hopsworks Model Registry.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` and `.streamlit/secrets.toml` are already git-ignored).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo,
   branch, and `app.py` as the main file.
3. In the app's **Settings → Secrets**, paste the same three keys as in
   `.streamlit/secrets.toml` above (with your real values).
4. Deploy.

## Notes

- **Predictions reflect the latest row in the feature store**, not necessarily right now — see
  step 3 above. If it looks stale, re-run `backfill.ipynb`'s gap-fill step, then hit
  **🔄 Refresh data** in the sidebar (feature data is cached for 1 hour).
- **Which model wins can differ per horizon** — check the sidebar for what's currently deployed
  and its R² per horizon.
- **SHAP** uses `TreeExplainer` for RandomForest/XGBoost and `LinearExplainer` for Ridge; MLP
  falls back to no SHAP view. If SHAP itself fails to load, the app shows the model's native
  feature importance instead so the dashboard still runs.
