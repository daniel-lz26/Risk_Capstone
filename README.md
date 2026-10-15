# SentinX — Supply-Chain Disruption Early Warning (prototype)

CPSC 491 capstone (CSUF). Forecasts country-level supply-chain disruption risk one year ahead for
**Asia-Pacific, 2010–2025** from ACLED civil-unrest data and World Bank macro indicators.
Outputs a **fragility score (0–100)**, a **disruption alert**, and **SHAP drivers**, served by FastAPI
and shown on a dashboard.

## Setup

```bash
pip install -r requirements.txt
```

Raw inputs (not modified by the pipeline):
- `data/raw/acled/Asia-Pacific_aggregated_data_up_to_week_of-2026-10-03.xlsx` (gitignored, ~10 MB)
- `data/raw/worldbank/wdi_data.csv` (DataBank WDI export; unemployment + 2010 are gap-filled from `wbgapi`)

## Run, step by step (from the repo root)

| Step | Command | Output |
|---|---|---|
| 1. Data pipeline | `python pipeline/run_pipeline.py` | `data/processed/{acled_country_year,wdi_country_year,model_table}.csv` |
| 2. Train + explain | `python model/train.py` | `model/artifacts/metrics.json`, `shap_summary.png`, `shap_values.csv`, models; `data/processed/predictions.csv` |
| 3. API | `uvicorn api.main:app --reload` | http://localhost:8000/docs |
| 4. Dashboard | open http://localhost:8000/dashboard/ (served by the API) | heat map, trends, drivers, metrics |

Individual pipeline steps: `python pipeline/acled_country_year.py <acled.xlsx>`, `python pipeline/worldbank.py`,
`python pipeline/merge.py`.

## Method (prototype)

- **Features (year t):** unrest / fatalities / violent events per million, YoY changes, riot share; GDP growth,
  inflation, trade, export growth, manufacturing, FDI, log GDP per capita, unemployment, their lag-1, and 3-year
  volatility of growth and inflation.
- **Target:** `disruption_next = 1` if next year's GDP growth falls > 3 pp below the trailing 3-year mean, or next
  year's inflation > 10%. `fragility_next` = that growth shortfall (≥ 0); the API score maps 10 pp → 100.
- **Split (temporal, no shuffling):** train ≤ 2019, validate 2020–21, test 2022–23; forecast = latest year.
- **Models:** Logistic / Ridge baselines, Random Forest, XGBoost; SHAP `TreeExplainer` on XGBoost.

## API

`GET /health`, `/countries`, `/risk/latest`, `/risk/{iso3}/history`, `/risk/{iso3}/drivers`, `/metrics`

## Known limitations / next steps

- ~200 labelled rows → metrics are noisy and preliminary. Scale to all regions next.
- Taiwan is dropped (not a UN member in the ISO mapping).
- Dashboard is a static Plotly page (`dashboard/index.html`); the Next.js version comes later.
- Next datasets: FAO food security, WGI governance, sovereign credit. Then Supabase storage and a what-if simulator.
- Search the code for `TODO(prototype)` for the simplifications made today.
