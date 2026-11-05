"""SentinX: what-if simulator: re-score a country with user-changed indicator values."""
from fastapi import APIRouter, Depends, HTTPException

from api.data import Store, get_store
from api.model import Scorer, get_scorer
from api.schemas import SimulateRequest, SimulateResult

router = APIRouter(tags=["simulate"])


@router.post("/simulate", response_model=SimulateResult)
def simulate(req: SimulateRequest, store: Store = Depends(get_store), scorer: Scorer = Depends(get_scorer)):
    """Apply the overrides to the country's latest inputs and return baseline vs simulated risk."""
    iso3 = req.iso3.upper()
    if not store.has_country(iso3):
        raise HTTPException(status_code=404, detail=f"Unknown country code {iso3}")
    unknown = sorted(set(req.overrides) - set(store.features))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown features {unknown}; valid: {store.features}")

    # TODO(prototype): derived features (lags, 3y volatility) are not recomputed from overridden values
    year, base = store.feature_row(iso3)
    scenario = base.copy()
    for feature, value in req.overrides.items():
        scenario[feature] = value

    baseline, simulated = scorer.score(base), scorer.score(scenario)
    return {"iso3": iso3, "country": store.latest(1).set_index("iso3").loc[iso3, "country"], "year": year,
            "overrides": req.overrides, "baseline": baseline, "simulated": simulated,
            "drivers": simulated["drivers"]}
