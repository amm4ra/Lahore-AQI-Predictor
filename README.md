# Lahore AQI Predictor

**Live app:** https://amm4ra-lahore-aqi-predictor-app-8hod2z.streamlit.app/

Predicts Lahore's US AQI 24/48/72 hours ahead. Weather and pollutant data comes from
Open-Meteo, gets engineered into features stored in a Hopsworks Feature Store, trains
per-horizon models (Ridge, RandomForest, XGBoost, MLP, best one wins per horizon) in the
Hopsworks Model Registry, and is served through a Streamlit dashboard.

## Project structure

| File | Purpose |
|---|---|
| `backfill.ipynb` | **One-time** historical pull (2 years) from Open-Meteo, initial write to the Feature Store. Never scheduled: re-running it would re-fetch and re-upload 2 years of data every time. |
| `feature_pipeline.ipynb` | **Hourly** job. Pulls a recent window from Open-Meteo's live endpoints (no ERA5 lag) and inserts only the new rows. Scheduled by `.github/workflows/feature-pipeline.yml` via [papermill](https://papermill.readthedocs.io/). |
| `training-pipeline-multihorizon.ipynb` | **Daily** job. Trains Ridge / RandomForest / XGBoost / MLP per horizon, prints a comparison table, registers the best model per horizon. Scheduled by `.github/workflows/training-pipeline.yml`, also via papermill. |
| `app.py` | Streamlit dashboard: 3-day forecast with dates, current conditions, EDA, SHAP explanations, hazard alerts. |
| `requirements.txt` | Pinned dependencies for the app and the training notebook. |
| `requirements-feature.txt` | Lighter dependency set for the hourly feature notebook (no torch/xgboost/shap). |
| `.github/workflows/` | GitHub Actions definitions for the two scheduled jobs above. |

Both scheduled notebooks run directly via papermill, not converted to `.py` scripts. The one
exception is `backfill.ipynb`, which mixes a one-time 2-year fetch with logic worth repeating,
so the recurring part was split into `feature_pipeline.ipynb` instead of scheduling all of it.

## What the dashboard does

- **Forecast tab**: current AQI plus predicted AQI for +24h/+48h/+72h, each labeled with its
  actual calendar date and marked as predicted. A line graph of the last 24 actual hours
  flowing into the 3-day forecast, plus a "Current conditions" panel (PM2.5, PM10, CO, NO2,
  SO2, O3, temperature, humidity, wind).
- **Hazard alert**: banner when the predicted peak AQI crosses unhealthy/hazardous thresholds.
- **EDA tab**: AQI over the full history, average AQI by hour of day, AQI trend over the last
  24 hours, PM2.5 vs AQI scatter.
- **Explanation tab**: SHAP contributions for the selected horizon's prediction (falls back to
  the model's built-in feature importance if SHAP errors out).

## Setup

### 0. Prerequisites

- Python 3.11
- A free [Hopsworks](https://www.hopsworks.ai/) account and project, for the Feature Store and
  Model Registry.
- No API key needed for Open-Meteo itself, it's free and keyless.

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

Create `.streamlit/secrets.toml` next to `app.py` (git-ignored, never commit it):

```toml
HOPSWORKS_KEY = "your_api_key_here"
HOPSWORKS_PROJECT = "your_project_name"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
```

Get the API key from Hopsworks under Account Settings > API Keys. The notebooks read the same
key from the `HOPSWORKS_KEY` environment variable instead, so for those: `setx HOPSWORKS_KEY
"your_key"` (Windows) or `export HOPSWORKS_KEY=your_key` (macOS/Linux) before launching Jupyter.

### 3. Populate the Feature Store (first time only)

Run `backfill.ipynb` top to bottom. It creates the `lahore_aqi_features` feature group and
loads about 2 years of history, stopping about 5 days back (the ERA5 archive's publish lag).
Keeping it current after that is `feature_pipeline.ipynb`'s job (see CI/CD below); you can also
run it once by hand (Run All) right after the backfill.

### 4. Train and register models

Run `training-pipeline-multihorizon.ipynb` top to bottom. It trains all four model types per
horizon, prints a comparison table, and registers the best model per horizon
(`lahore_aqi_model_24h` / `_48h` / `_72h`) in the Model Registry.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## CI/CD (GitHub Actions)

Two scheduled workflows keep the project current without manual work. Each installs its
notebook's dependencies, registers a Jupyter kernel, then runs the notebook headlessly with
`papermill notebook.ipynb /tmp/output.ipynb`. The executed output notebook is discarded, not
committed, so the repo stays clean.

- `.github/workflows/feature-pipeline.yml`: runs `feature_pipeline.ipynb` every hour, pulling
  the latest weather/AQI data into the feature store.
- `.github/workflows/training-pipeline.yml`: runs `training-pipeline-multihorizon.ipynb` once
  a day, retraining and re-registering the best model per horizon on the freshest data.

Both need the same three Hopsworks values as repo secrets (Settings > Secrets and variables >
Actions > New repository secret): `HOPSWORKS_KEY`, `HOPSWORKS_PROJECT`, `HOPSWORKS_HOST`. Once
those are set, the schedules run on their own. You can also trigger either one manually from
the Actions tab (`workflow_dispatch`) to test it without waiting for the cron.

`app.py` picks up newly-registered models automatically: it always loads the highest version
number per model name, not whatever it saw first, and refreshes its model cache hourly (or
immediately via the sidebar's "Refresh data & models" button).

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` and `.streamlit/secrets.toml` are already git-ignored).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo,
   branch, and `app.py` as the main file.
3. Click **Advanced settings** and set the Python version to **3.11**, matching what's pinned
   in `requirements.txt`. Community Cloud otherwise defaults to a newer Python that may lack
   prebuilt wheels for pinned packages like `pandas`, forcing a slow source build that can
   stall the deploy.
4. In the app's Settings > Secrets, paste the same three keys as in `.streamlit/secrets.toml`
   above, with your real values, no `[section]` header, just flat `KEY = "value"` lines. Save,
   then reboot the app so it picks them up.
5. Deploy.

## Notes

- **Predictions reflect the latest row in the feature store.** Once the CI/CD workflows above
  are running, this stays current on its own. Without them, run `feature_pipeline.ipynb` by
  hand, then hit "Refresh data & models" in the sidebar (feature data is cached for 1 hour).
- **Which model wins can differ per horizon.** Check the sidebar for what's currently deployed
  and its R2 per horizon.
- **SHAP** uses `TreeExplainer` for RandomForest/XGBoost and `LinearExplainer` for Ridge. MLP
  falls back to no SHAP view. If SHAP itself fails to load, the app shows the model's native
  feature importance instead so the dashboard still runs.
