"""SentinX: whole-map and service endpoints: heatmap, metrics, health."""
from fastapi import APIRouter, Depends, HTTPException, Query

from api.data import Store, get_store, records
from api.schemas import Health, HeatmapEntry, Metrics

router = APIRouter()


@router.get("/health", response_model=Health, tags=["meta"])
def health():
    return {"status": "ok"}


@router.get("/heatmap", response_model=list[HeatmapEntry], tags=["overview"])
def heatmap(horizon: int = Query(1, ge=1, le=2), store: Store = Depends(get_store)):
    """Latest forecast for every country in one call, for the choropleth."""
    if horizon not in store.horizons():
        raise HTTPException(status_code=404,
                            detail=f"No {horizon}-year forecast yet; available horizons: {store.horizons()}")
    latest = store.latest(horizon)
    return records(latest[["iso3", "country", "year", "horizon", "fragility_score", "alert_prob", "alert"]])


@router.get("/metrics", response_model=Metrics, tags=["meta"])
def metrics(store: Store = Depends(get_store)):
    return store.metrics
