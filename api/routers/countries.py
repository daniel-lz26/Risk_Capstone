"""SentinX: per-country endpoints; list, history, forecast and SHAP attribution."""
from fastapi import APIRouter, Depends, HTTPException, Query

from api.data import Store, get_store, records
from api.schemas import Attribution, CountryHistory, CountrySummary, Forecast

router = APIRouter(prefix="/countries", tags=["countries"])


def valid_iso3(iso3: str, store: Store = Depends(get_store)) -> str:
    """Path dependency: normalise the code and 404 on unknown countries."""
    iso3 = iso3.upper()
    if not store.has_country(iso3):
        raise HTTPException(status_code=404, detail=f"Unknown country code {iso3}")
    return iso3


@router.get("", response_model=list[CountrySummary])
def list_countries(store: Store = Depends(get_store)):
    latest = store.latest(1).sort_values("country").rename(columns={"year": "latest_year"})
    return records(latest[["iso3", "country", "latest_year", "fragility_score", "alert_prob", "alert"]])


@router.get("/{iso3}/history", response_model=CountryHistory)
def country_history(iso3: str = Depends(valid_iso3), store: Store = Depends(get_store)):
    h = store.history(iso3)
    return {"iso3": iso3, "country": h["country"].iloc[0], "history": records(h)}


@router.get("/{iso3}/forecast", response_model=Forecast)
def country_forecast(iso3: str = Depends(valid_iso3), horizon: int = Query(1, ge=1, le=2),
                     store: Store = Depends(get_store)):
    row = store.forecast(iso3, horizon)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"No {horizon}-year forecast yet; available horizons: {store.horizons()}")
    return {**row, "base_year": row["year"], "target_year": row["year"] + horizon}


@router.get("/{iso3}/attribution", response_model=Attribution)
def country_attribution(iso3: str = Depends(valid_iso3),
                        top: int | None = Query(None, ge=1, description="Only the N largest drivers"),
                        store: Store = Depends(get_store)):
    country = store.latest(1).set_index("iso3").loc[iso3, "country"]
    return {"iso3": iso3, "country": country, **store.attribution(iso3, top)}
