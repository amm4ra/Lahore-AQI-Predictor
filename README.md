# Lahore AQI Predictor

Predicts Lahore's US AQI 24/48/72 hours ahead. Weather + pollutant data is pulled from
Open-Meteo, engineered into features and stored in a Hopsworks Feature Store, used to train
and register per-horizon models (Ridge, RandomForest, XGBoost, MLP — best one wins per
horizon) in the Hopsworks Model Registry, and served through a Streamlit dashboard.

## Project structure

| File | Purpose |
|---|---|
| `backfill.ipynb` | **One-time** historical pull (2 years) from Open-Meteo and initial write to the Hopsworks Feature Store. Run this once when setting the project up from scratch — never scheduled, since re-running it would re-fetch and re-upload 2 years of data every time. |
| `feature_pipeline.ipynb` | **Hourly** job: pulls a recent window from Open-Meteo's live endpoints (no ERA5 lag) and inserts only the new rows into the feature store. Scheduled by `.github/workflows/feature-pipeline.yml`, which runs it headlessly via [papermill](https://papermill.readthedocs.io/). |
| `training-pipeline-multihorizon.ipynb` | **Daily** job: trains Ridge / RandomForest / XGBoost / MLP for each horizon (+24h/+48h/+72h) on the current feature set, prints a comparison table, and registers the best model per horizon. Scheduled by `.github/workflows/training-pipeline.yml`, also via papermill. |
| `app.py` | Streamlit dashboard: 3-day forecast (with dates), current conditions (pollutants + weather), EDA (AQI history, hourly pattern, last-24h trend), SHAP-based explanations, and hazard alerts. |
| `requirements.txt` | Pinned dependencies for the app and the training notebook. |
| `requirements-feature.txt` | Lighter dependency set for the hourly feature notebook (no torch/xgboost/shap needed for a data pull). |
| `.github/workflows/` | GitHub Actions definitions for the two scheduled jobs above. |

Both scheduled notebooks are run directly (via papermill), not converted to `.py` scripts — the
only exception would be `backfill.ipynb`, which mixes a one-time 2-year fetch with logic you'd
want to repeat, so its recurring part was split out into `feature_pipeline.ipynb` instead of
scheduling the whole thing.

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
loads ~2 years of history, stopping ~5 days back (the ERA5 archive's publish lag). Keeping it
current from there on is `feature_pipeline.ipynb`'s job (see CI/CD below) — you can also run it
once by hand (Run All) right after the backfill.

### 4. Train and register models

Run `training-pipeline-multihorizon.ipynb` top to bottom. It trains all four model types per
horizon, prints a comparison table, and registers the best model per horizon
(`lahore_aqi_model_24h` / `_48h` / `_72h`) in the Hopsworks Model Registry.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## CI/CD (GitHub Actions)

Two scheduled workflows keep the project current without manual intervention — each installs
its notebook's dependencies, registers a Jupyter kernel, then runs the notebook headlessly with
`papermill notebook.ipynb /tmp/output.ipynb` (the executed output notebook is discarded, not
committed back, so the repo stays clean):

- **`.github/workflows/feature-pipeline.yml`** — runs `feature_pipeline.ipynb` every hour,
  pulling the latest weather/AQI data into the feature store.
- **`.github/workflows/training-pipeline.yml`** — runs `training-pipeline-multihorizon.ipynb`
  once a day, retraining and re-registering the best model per horizon on the freshest data.

Both need the same three Hopsworks values as repo secrets (**Settings → Secrets and variables →
Actions → New repository secret**): `HOPSWORKS_KEY`, `HOPSWORKS_PROJECT`, `HOPSWORKS_HOST`.
Once those are set, the schedules run on their own — you can also trigger either one manually
from the **Actions** tab (`workflow_dispatch`) to test it without waiting for the cron.

`app.py` picks up newly-registered models automatically: it always loads the highest version
number per model name (not just whatever it saw first), and refreshes its model cache hourly
(or immediately via the sidebar's **🔄 Refresh data & models** button).

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` and `.streamlit/secrets.toml` are already git-ignored).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo,
   branch, and `app.py` as the main file.
3. In the app's **Settings → Secrets**, paste the same three keys as in
   `.streamlit/secrets.toml` above (with your real values).
4. Deploy.

## Notes

- **Predictions reflect the latest row in the feature store.** Once the CI/CD workflows above
  are running, this stays current on its own; without them, run `feature_pipeline.ipynb` by
  hand, then hit **🔄 Refresh data & models** in the sidebar (feature data is cached for 1 hour).
- **Which model wins can differ per horizon** — check the sidebar for what's currently deployed
  and its R² per horizon.
- **SHAP** uses `TreeExplainer` for RandomForest/XGBoost and `LinearExplainer` for Ridge; MLP
  falls back to no SHAP view. If SHAP itself fails to load, the app shows the model's native
  feature importance instead so the dashboard still runs.
