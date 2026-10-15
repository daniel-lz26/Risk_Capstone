"""
SentinX - ACLED cleaning + country-year aggregation.

Input : ACLED regional aggregated file (weekly x admin1 x event type), .xlsx or .csv
Output: data/processed/acled_country_year.csv  -> one row per (iso3, year) with unrest features

Usage (from repo root):
    python pipeline/acled_country_year.py data/raw/acled/Asia-Pacific_aggregated_data_up_to_week_of-2026-10-03.xlsx
"""
import logging
import sys
from pathlib import Path

import country_converter as coco
import pandas as pd

logging.getLogger("country_converter").setLevel(logging.ERROR)

FIRST_YEAR = 2010
LAST_FULL_YEAR = 2025   # 2026 is a partial year -> excluded so counts aren't biased low
MIN_EVENTS_TOTAL = 50   # drop micro-territories with almost no recorded activity
OUT_PATH = Path("data/processed/acled_country_year.csv")
DEFAULT_RAW = "data/raw/acled/Asia-Pacific_aggregated_data_up_to_week_of-2026-10-03.xlsx"

EVENT_COLS = {
    "Protests": "protests",
    "Riots": "riots",
    "Battles": "battles",
    "Violence against civilians": "viol_civilians",
    "Explosions/Remote violence": "explosions",
    "Strategic developments": "strategic_dev",
}


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    df["week"] = pd.to_datetime(df["week"])
    df["year"] = df["week"].dt.year
    return df


def harmonize_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """Map ACLED country names to ISO-3 and keep only UN-member sovereign states."""
    names = df["country"].unique()
    iso3 = coco.convert(names=list(names), to="ISO3", not_found=None)
    # coco.convert(to="UNmember") crashes on NA in country_converter 1.3.x, so read the table directly
    cc_data = coco.CountryConverter().data
    un_iso3 = set(cc_data.loc[cc_data["UNmember"].notna(), "ISO3"])
    un = [code if code in un_iso3 else None for code in iso3]
    lookup = pd.DataFrame({"country": names, "iso3": iso3, "un_member": un})
    # coco echoes the input name back when it can't match (e.g. "Indian Ocean"), so require a real 3-letter code
    valid_iso = lookup["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}") & (lookup["iso3"] != lookup["country"])
    lookup["keep"] = valid_iso & lookup["un_member"].notna() & (lookup["un_member"] != lookup["country"])
    dropped = sorted(lookup.loc[~lookup["keep"], "country"])
    print(f"Dropped {len(dropped)} non-sovereign / non-UN entities: {dropped}")
    df = df.merge(lookup.loc[lookup["keep"], ["country", "iso3"]], on="country", how="inner")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # coverage start per country (rows before this = no data, NOT zero events)
    cov_start = df.groupby("iso3")["week"].min().dt.year.rename("coverage_start")

    counts = (
        df.pivot_table(index=["iso3", "country", "year"], columns="event_type",
                       values="events", aggfunc="sum", fill_value=0)
        .rename(columns=EVENT_COLS)
    )
    for c in EVENT_COLS.values():
        if c not in counts:
            counts[c] = 0
    counts = counts[list(EVENT_COLS.values())]

    fat = df.groupby(["iso3", "country", "year"])["fatalities"].sum().rename("fatalities")
    out = counts.join(fat).reset_index()

    # complete the panel: every country x every covered year, zeros where nothing happened
    out = out.merge(cov_start, on="iso3")
    names = out.drop_duplicates("iso3").set_index("iso3")["country"]
    full = pd.MultiIndex.from_product([out["iso3"].unique(), range(FIRST_YEAR, LAST_FULL_YEAR + 1)],
                                      names=["iso3", "year"]).to_frame(index=False)
    full = full.merge(cov_start, on="iso3")
    full = full[full["year"] >= full["coverage_start"]]
    # a country whose coverage begins mid-year has a partial first year -> drop it
    first_week = df.groupby("iso3")["week"].min()
    partial = first_week[first_week.dt.dayofyear > 14].dt.year
    full = full[~full.set_index(["iso3", "year"]).index.isin(list(partial.items()))]

    out = full.merge(out.drop(columns=["country", "coverage_start"]), on=["iso3", "year"], how="left")
    out["country"] = out["iso3"].map(names)
    num = list(EVENT_COLS.values()) + ["fatalities"]
    out[num] = out[num].fillna(0).astype(int)

    # drop countries with negligible activity overall
    totals = out.groupby("iso3")[num].sum().sum(axis=1)
    small = totals[totals < MIN_EVENTS_TOTAL].index
    if len(small):
        print(f"Dropped {len(small)} low-activity countries: {sorted(out.loc[out.iso3.isin(small), 'country'].unique())}")
    out = out[~out["iso3"].isin(small)]
    return out


def engineer(out: pd.DataFrame) -> pd.DataFrame:
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    out["unrest"] = out["protests"] + out["riots"]
    out["violent_events"] = out["battles"] + out["viol_civilians"] + out["explosions"]
    out["total_events"] = out[list(EVENT_COLS.values())].sum(axis=1)
    out["riot_share"] = (out["riots"] / out["unrest"]).where(out["unrest"] > 0, 0).round(4)

    g = out.groupby("iso3")
    for c in ["unrest", "violent_events", "fatalities"]:
        out[f"{c}_lag1"] = g[c].shift(1)
        # year-over-year % change (log-ish, robust to zeros)
        out[f"{c}_yoy"] = ((out[c] - out[f"{c}_lag1"]) / (out[f"{c}_lag1"] + 1)).round(4)
        out[f"{c}_3y_mean"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean()).round(2)

    cols = ["iso3", "country", "year", "coverage_start",
            *EVENT_COLS.values(), "fatalities", "unrest", "violent_events", "total_events", "riot_share",
            *[c for c in out.columns if c.endswith(("_lag1", "_yoy", "_3y_mean"))]]
    return out[cols]


def main(path: str) -> None:
    df = load(path)
    print(f"Loaded {len(df):,} rows, {df['country'].nunique()} entities, {df['week'].min().date()} -> {df['week'].max().date()}")
    df = harmonize_iso3(df)
    out = engineer(aggregate(df))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(out)} country-year rows, {out['iso3'].nunique()} countries, "
          f"{out['year'].min()}-{out['year'].max()}")
    print(out.groupby("iso3")["year"].agg(["min", "max", "count"]).sort_values("min").to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW)
