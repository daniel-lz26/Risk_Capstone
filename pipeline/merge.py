"""
SentinX - merge ACLED unrest + WDI macro into the model table, build features and targets.

Input : data/processed/acled_country_year.csv, data/processed/wdi_country_year.csv
Output: data/processed/model_table.csv

Usage (from repo root):
    python pipeline/merge.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ACLED_PATH = Path("data/processed/acled_country_year.csv")
WDI_PATH = Path("data/processed/wdi_country_year.csv")
OUT_PATH = Path("data/processed/model_table.csv")

MACRO = ["gdp_growth", "inflation", "trade_pct_gdp", "export_growth", "manuf_pct_gdp",
         "fdi_pct_gdp", "log_gdp_pc", "unemployment"]

SHORTFALL_PP = 3.0     # next-year growth this far below the trailing 3y mean -> disruption
INFLATION_ALERT = 10.0  # or next-year inflation above this


def macro_features(wdi: pd.DataFrame) -> pd.DataFrame:
    """Lags, volatility and targets on the full WDI panel, so history before ACLED coverage still counts."""
    wdi = wdi.sort_values(["iso3", "year"]).reset_index(drop=True)
    wdi["log_gdp_pc"] = np.log(wdi["gdp_pc"])
    g = wdi.groupby("iso3")

    # volatility over years t-2..t (only info available at t)
    wdi["gdp_growth_vol3"] = g["gdp_growth"].transform(lambda s: s.rolling(3, min_periods=2).std())
    wdi["inflation_vol3"] = g["inflation"].transform(lambda s: s.rolling(3, min_periods=2).std())
    for c in MACRO:
        wdi[f"{c}_lag1"] = g[c].shift(1)

    # --- targets (prototype definition) ---
    # trailing 3y mean as of year t (t-2..t) is the baseline next year's growth is judged against
    growth_3y_mean = g["gdp_growth"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    next_growth = g["gdp_growth"].shift(-1)
    next_inflation = g["inflation"].shift(-1)
    # a "next year" only counts if it is literally year t+1
    has_next = g["year"].shift(-1) == wdi["year"] + 1
    next_growth = next_growth.where(has_next)
    next_inflation = next_inflation.where(has_next)

    shortfall = growth_3y_mean - next_growth
    wdi["fragility_next"] = shortfall.clip(lower=0)
    # TODO(prototype): if only inflation is known for next year we still label it; revisit missing-target rules
    disrupt = (shortfall > SHORTFALL_PP) | (next_inflation > INFLATION_ALERT)
    known = next_growth.notna() | next_inflation.notna()
    wdi["disruption_next"] = disrupt.astype(float).where(known)
    return wdi


def main() -> None:
    acled = pd.read_csv(ACLED_PATH)
    wdi = macro_features(pd.read_csv(WDI_PATH))

    df = acled.merge(wdi, on=["iso3", "year"], how="inner")
    df["unrest_per_m"] = df["unrest"] / df["population"] * 1e6
    df["fatalities_per_m"] = df["fatalities"] / df["population"] * 1e6
    df["violent_per_m"] = df["violent_events"] / df["population"] * 1e6

    # last year per country has no next year -> prediction row, not a training row
    df["is_forecast_row"] = df["year"] == df.groupby("iso3")["year"].transform("max")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    train_like = df[~df["is_forecast_row"]]
    print(f"Wrote {OUT_PATH}: {len(df)} rows, {df['iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()}")
    print(f"  ACLED rows: {len(acled)}  -> after inner join with WDI: {len(df)}")
    print(f"  forecast rows: {df['is_forecast_row'].sum()}  |  labelled rows: {train_like['disruption_next'].notna().sum()}")
    print("\nClass balance of disruption_next (labelled rows):")
    print(train_like["disruption_next"].value_counts(dropna=False).rename({0.0: "0 (no)", 1.0: "1 (disruption)"}).to_string())
    print("\nMissingness per column (share NaN, only columns with any):")
    miss = df.isna().mean().round(3)
    print(miss[miss > 0].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
