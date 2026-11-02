"""
SentinX: Pydantic response models. These are the API contract: FastAPI validates every response
against them and publishes them in the OpenAPI schema at /docs.
"""
from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str


class CountrySummary(BaseModel):
    iso3: str
    country: str
    latest_year: int = Field(description="Latest year of observed data; the forecast is for the year after")
    fragility_score: float = Field(ge=0, le=100)
    alert_prob: float = Field(ge=0, le=1)
    alert: bool


class HeatmapEntry(BaseModel):
    iso3: str
    country: str
    year: int = Field(description="Feature year; the score forecasts year + horizon")
    horizon: int
    fragility_score: float = Field(ge=0, le=100)
    alert_prob: float = Field(ge=0, le=1)
    alert: bool


class HistoryPoint(BaseModel):
    year: int
    fragility_score: float | None
    alert_prob: float | None
    alert: bool | None
    unrest: float | None
    unrest_per_m: float | None
    fatalities: float | None
    gdp_growth: float | None
    inflation: float | None
    trade_pct_gdp: float | None


class CountryHistory(BaseModel):
    iso3: str
    country: str
    history: list[HistoryPoint]


class Forecast(BaseModel):
    iso3: str
    country: str
    horizon: int
    base_year: int = Field(description="Year of the features the forecast is made from")
    target_year: int = Field(description="Year being forecast (base_year + horizon)")
    fragility_score: float = Field(ge=0, le=100)
    score_lo: float | None = Field(description="Lower bound of the confidence interval (null until the model provides one)")
    score_hi: float | None = Field(description="Upper bound of the confidence interval (null until the model provides one)")
    alert_prob: float = Field(ge=0, le=1)
    alert: bool


class Driver(BaseModel):
    feature: str
    value: float | None = Field(description="The country's value for this feature")
    shap: float = Field(description="Contribution to the score; positive raises risk")


class Attribution(BaseModel):
    iso3: str
    country: str
    year: int
    units: str
    drivers: list[Driver]


Metrics = dict[str, Any]  # evaluation report from model/train.py; passed through as-is
