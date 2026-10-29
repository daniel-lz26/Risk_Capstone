"""
SentinX - FastAPI backend serving predictions, history and SHAP drivers from in-memory CSVs.

Run (from repo root):
    uvicorn api.main:app --reload
Docs: http://localhost:8000/docs   Dashboard: http://localhost:8000/dashboard/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ART = ROOT / "model" / "artifacts"
HISTORY_FEATURES = ["unrest", "unrest_per_m", "fatalities", "gdp_growth", "inflation", "trade_pct_gdp"]

# ---- load everything once at startup (no DB today; Supabase later) ----
PRED = pd.read_csv(PROCESSED / "predictions.csv")
TABLE = pd.read_csv(PROCESSED / "model_table.csv")
SHAP = pd.read_csv(ART / "shap_values.csv")
METRICS = json.loads((ART / "metrics.json").read_text())
FEATURES = METRICS["features"]
HIST = PRED.merge(TABLE[["iso3", "year", *HISTORY_FEATURES]], on=["iso3", "year"], how="left")

app = FastAPI(title="SentinX API", version="0.1.0",
              description="Country-level supply-chain disruption risk (Asia-Pacific prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET"], allow_headers=["*"])


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> null)."""
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def check_iso3(iso3: str) -> str:
    iso3 = iso3.upper()
    if iso3 not in set(PRED["iso3"]):
        raise HTTPException(status_code=404, detail=f"Unknown country code {iso3}")
    return iso3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/countries")
def countries():
    return records(PRED[["iso3", "country"]].drop_duplicates().sort_values("country"))


@app.get("/risk/latest")
def risk_latest():
    latest = PRED[PRED["is_forecast_row"]]
    return records(latest[["iso3", "country", "year", "fragility_score", "alert_prob", "alert"]]
                   .sort_values("fragility_score", ascending=False))

# "/risk/{iso3}/history" -> /countries/{iso3}/history based off project proposal doc
@app.get("/risk/{iso3}/history")
def risk_history(iso3: str):
    iso3 = check_iso3(iso3)
    h = HIST[HIST["iso3"] == iso3].sort_values("year")
    return {"iso3": iso3, "country": h["country"].iloc[0],
            "history": records(h[["year", "fragility_score", "alert_prob", "alert", *HISTORY_FEATURES]])}


@app.get("/risk/{iso3}/drivers")
def risk_drivers(iso3: str, top: int = 5):
    iso3 = check_iso3(iso3)
    year = int(PRED.loc[(PRED["iso3"] == iso3) & PRED["is_forecast_row"], "year"].iloc[0])
    sv = SHAP[(SHAP["iso3"] == iso3) & (SHAP["year"] == year)][FEATURES].iloc[0]
    vals = TABLE[(TABLE["iso3"] == iso3) & (TABLE["year"] == year)][FEATURES].iloc[0]
    order = sv.abs().sort_values(ascending=False).index[:top]
    drivers = [{"feature": f, "value": None if pd.isna(vals[f]) else round(float(vals[f]), 3),
                "shap": round(float(sv[f]), 3)} for f in order]
    return {"iso3": iso3, "year": year, "units": "percentage-point growth shortfall", "drivers": drivers}


@app.get("/metrics")
def metrics():
    return METRICS


# static dashboard served from the same origin, so no CORS setup is needed for the demo
DASH = ROOT / "dashboard"
if (DASH / "index.html").exists():
    app.mount("/dashboard", StaticFiles(directory=DASH, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/dashboard/")
