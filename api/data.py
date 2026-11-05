"""
SentinX's data access layer. Loads the precomputed tables once and answers lookups.

Everything that knows where the data lives (CSV today, SQLite/Parquet later) stays in this file,
so the routers never touch pandas file paths directly.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ART = ROOT / "model" / "artifacts"
HISTORY_FEATURES = ["unrest", "unrest_per_m", "fatalities", "gdp_growth", "inflation", "trade_pct_gdp"]
SCORE_UNITS = "percentage-point growth shortfall"

# Records helper function
def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> None)."""
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))

# Store class: Loading the data
class Store:
    def __init__(self) -> None:
        self.pred = pd.read_csv(PROCESSED / "predictions.csv")
        # TODO: Our proposal says API should support 2-year forecasts and a "range"
        # around each score. Our data doesn't have those yet, so these lines
        # add 2 empty-ish columns to the data so the API has the right shape.
        # This is temporary.
        # Fill the columns the API contract expects until the model output includes them.
        if "horizon" not in self.pred:
            self.pred["horizon"] = 1
        for col in ["score_lo", "score_hi"]:
            if col not in self.pred:
                self.pred[col] = np.nan
        self.table = pd.read_csv(PROCESSED / "model_table.csv")
        self.shap = pd.read_csv(ART / "shap_values.csv")
        self.metrics = json.loads((ART / "metrics.json").read_text())
        self.features: list[str] = self.metrics["features"]
        self.history_table = self.pred[self.pred["horizon"] == 1].merge(
            self.table[["iso3", "year", *HISTORY_FEATURES]], on=["iso3", "year"], how="left")

    def has_country(self, iso3: str) -> bool:
        return iso3 in set(self.pred["iso3"])

    def horizons(self) -> list[int]:
        return sorted(int(h) for h in self.pred["horizon"].unique())

    def latest(self, horizon: int = 1) -> pd.DataFrame:
        """One forecast row per country (the latest feature year), highest score first."""
        rows = self.pred[self.pred["is_forecast_row"] & (self.pred["horizon"] == horizon)]
        return rows.sort_values("fragility_score", ascending=False)

    def history(self, iso3: str) -> pd.DataFrame:
        return self.history_table[self.history_table["iso3"] == iso3].sort_values("year")
    
    def feature_row(self, iso3: str) -> tuple[int, pd.DataFrame]:
        """The model inputs for the country's forecast row (one-row DataFrame), plus its year."""
        # finds which year is this country's latest
        year = int(self.latest(1).set_index("iso3").loc[iso3, "year"])
        # picks out that one country-year row 
        # self.features keeps only 25 columns the model knows about (table has ~50)
        row = self.table[(self.table["iso3"] == iso3) & (self.table["year"] == year)][self.features]
        return year, row.reset_index(drop=True)


    def forecast(self, iso3: str, horizon: int) -> dict | None:
        rows = self.latest(horizon)
        rows = rows[rows["iso3"] == iso3]
        return records(rows)[0] if len(rows) else None

    def attribution(self, iso3: str, top: int | None = None) -> dict:
        """SHAP vector for the country's forecast row, sorted by absolute contribution."""
        year = int(self.latest(1).set_index("iso3").loc[iso3, "year"])
        sv = self.shap[(self.shap["iso3"] == iso3) & (self.shap["year"] == year)][self.features].iloc[0]
        vals = self.table[(self.table["iso3"] == iso3) & (self.table["year"] == year)][self.features].iloc[0]
        order = sv.abs().sort_values(ascending=False).index[:top]
        drivers = [{"feature": f, "value": None if pd.isna(vals[f]) else round(float(vals[f]), 3),
                    "shap": round(float(sv[f]), 3)} for f in order]
        return {"year": year, "units": SCORE_UNITS, "drivers": drivers}


@lru_cache
def get_store() -> Store:
    """Built on first use and reused for every request (FastAPI dependency)."""
    return Store()