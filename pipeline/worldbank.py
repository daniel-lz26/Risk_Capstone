"""
SentinX - World Bank WDI macro indicators -> country-year panel.

Input : data/raw/worldbank/*.csv (DataBank wide export), or wbgapi if no CSV is present
Output: data/processed/wdi_country_year.csv  -> one row per (iso3, year), one column per indicator

Usage (from repo root):
    python pipeline/worldbank.py
"""
import logging
from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd

logging.getLogger("country_converter").setLevel(logging.ERROR)

RAW_DIR = Path("data/raw/worldbank")
OUT_PATH = Path("data/processed/wdi_country_year.csv")
YEARS = range(2010, 2026)

SERIES = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "NE.EXP.GNFS.KD.ZG": "export_growth",
    "NV.IND.MANF.ZS": "manuf_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "NY.GDP.PCAP.CD": "gdp_pc",
    "SP.POP.TOTL": "population",
    "SL.UEM.TOTL.ZS": "unemployment",
}


def load_databank_csv(path: Path) -> pd.DataFrame:
    """DataBank wide export -> long (iso3, year, indicator columns)."""
    df = pd.read_csv(path, na_values="..")
    df = df[df["Series Code"].isin(SERIES)]  # also drops the footer rows
    year_cols = [c for c in df.columns if c[:4].isdigit()]
    long = df.melt(id_vars=["Country Code", "Series Code"], value_vars=year_cols,
                   var_name="year", value_name="value")
    long["year"] = long["year"].str[:4].astype(int)
    out = (long.pivot_table(index=["Country Code", "year"], columns="Series Code", values="value")
           .rename(columns=SERIES).reset_index().rename(columns={"Country Code": "iso3"}))
    out.columns.name = None
    return out


def fetch_wbgapi(codes: list[str]) -> pd.DataFrame:
    """Same series via the World Bank API (fallback / gap filler)."""
    import wbgapi as wb
    df = wb.data.DataFrame(codes, time=YEARS, labels=False, columns="series").reset_index()
    df = df.rename(columns={"economy": "iso3", "time": "year", **SERIES})
    df["year"] = df["year"].astype(str).str.replace("YR", "").astype(int)
    return df


def keep_real_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Drop WDI aggregates (World, East Asia & Pacific, income groups...) - they aren't ISO-3 countries."""
    codes = set(df["iso3"].unique())
    valid = codes & set(coco.CountryConverter().data["ISO3"])
    print(f"Dropped {len(codes - valid)} aggregate / non-country codes, kept {len(valid)}")
    return df[df["iso3"].isin(valid)]


def main() -> None:
    csvs = sorted(p for p in RAW_DIR.glob("*.csv") if "metadata" not in p.name.lower())
    if csvs:
        print(f"Loading DataBank CSV: {csvs[0]}")
        df = load_databank_csv(csvs[0])
    else:
        print("No DataBank CSV found -> fetching via wbgapi")
        df = fetch_wbgapi(list(SERIES))

    if csvs:
        # TODO(prototype): the DataBank export has no unemployment series and starts at 2011;
        # gap-fill from the API (CSV values win). Re-export DataBank with SL.UEM.TOTL.ZS + 2010 later.
        try:
            api = fetch_wbgapi(list(SERIES))
            n_before = df.notna().sum().sum()
            df = df.set_index(["iso3", "year"]).combine_first(api.set_index(["iso3", "year"])).reset_index()
            print(f"Gap-filled {df.notna().sum().sum() - n_before:,} cells from wbgapi")
        except Exception as e:  # offline -> keep going with whatever the CSV has
            print(f"  wbgapi gap-fill failed ({e}); continuing with CSV only")
    for col in SERIES.values():
        if col not in df.columns:
            df[col] = np.nan

    df = keep_real_countries(df)
    df = df[df["year"].isin(YEARS)]
    df = df[["iso3", "year", *SERIES.values()]].sort_values(["iso3", "year"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(df)} rows, {df['iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()}")
    print("Non-missing share by year (2025 lags ~1 year at the World Bank):")
    print(df.groupby("year")[list(SERIES.values())].apply(lambda g: g.notna().mean()).round(2).to_string())


if __name__ == "__main__":
    main()
