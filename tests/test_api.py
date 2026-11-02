"""Endpoint tests for the SentinX API (run from repo root: pytest)."""
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def iso3() -> str:
    """A country that is actually in the data (the highest-risk one)."""
    return client.get("/heatmap").json()[0]["iso3"]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_countries_have_latest_scores():
    body = client.get("/countries").json()
    assert len(body) > 0
    assert {"iso3", "country", "latest_year", "fragility_score", "alert_prob", "alert"} <= body[0].keys()


def test_heatmap_sorted_by_score():
    scores = [r["fragility_score"] for r in client.get("/heatmap").json()]
    assert scores == sorted(scores, reverse=True)


def test_history(iso3):
    body = client.get(f"/countries/{iso3}/history").json()
    years = [p["year"] for p in body["history"]]
    assert body["iso3"] == iso3 and years == sorted(years)


def test_forecast_horizon_1(iso3):
    body = client.get(f"/countries/{iso3}/forecast?horizon=1").json()
    assert body["target_year"] == body["base_year"] + 1


def test_forecast_horizon_2_not_available_yet(iso3):
    assert client.get(f"/countries/{iso3}/forecast?horizon=2").status_code == 404


def test_forecast_rejects_bad_horizon(iso3):
    assert client.get(f"/countries/{iso3}/forecast?horizon=5").status_code == 422


def test_attribution_sorted_by_magnitude(iso3):
    drivers = client.get(f"/countries/{iso3}/attribution").json()["drivers"]
    mags = [abs(d["shap"]) for d in drivers]
    assert mags == sorted(mags, reverse=True)
    assert len(client.get(f"/countries/{iso3}/attribution?top=5").json()["drivers"]) == 5


def test_lowercase_iso3_accepted(iso3):
    assert client.get(f"/countries/{iso3.lower()}/history").status_code == 200


def test_unknown_country_404():
    assert client.get("/countries/XXX/history").status_code == 404


@pytest.mark.parametrize("path", ["/countries", "/heatmap", "/countries/{iso3}/attribution"])
def test_latency_under_500ms(path, iso3):
    start = time.perf_counter()
    assert client.get(path.format(iso3=iso3)).status_code == 200
    assert time.perf_counter() - start < 0.5