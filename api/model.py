"""
SentinX: service layer. Loads the trained XGBoost models (from model/artifacts) once and re-scores feature rows on demand,
for the what-if simulator. Every other endpoint reads precomputed results from api/data.py.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from api.data import ART

SCORE_CAP_PP = 10.0  # must match model/train.py: a 10pp growth shortfall maps to score 100

# small converter
def to_score(shortfall_pp: float) -> float:
    return round(float(np.clip(shortfall_pp / SCORE_CAP_PP, 0, 1) * 100), 1)

# loading the models
class Scorer:
    def __init__(self) -> None:
        import shap  # slow import (~4s) so only paid the first time /simulate is used

        self.score_model = joblib.load(ART / "score_xgboost.joblib")
        self.alert_model = joblib.load(ART / "alert_xgboost.joblib")
        self.explainer = shap.TreeExplainer(self.score_model)

    # using models (score method)
    def score(self, row: pd.DataFrame) -> dict:
        """Score a single feature row: fragility score, alert, and SHAP drivers sorted by size."""
        alert_prob = round(float(self.alert_model.predict_proba(row)[0, 1]), 4)
        sv = pd.Series(self.explainer.shap_values(row)[0], index=row.columns)
        order = sv.abs().sort_values(ascending=False).index
        values = row.iloc[0]
        drivers = [{"feature": f, "value": None if pd.isna(values[f]) else round(float(values[f]), 3),
                    "shap": round(float(sv[f]), 3)} for f in order]
        return {"fragility_score": to_score(self.score_model.predict(row)[0]),
                "alert_prob": alert_prob, "alert": alert_prob >= 0.5, "drivers": drivers}

# build scorer first time then keep hanging back same one
# models load once, not on every request
@lru_cache
def get_scorer() -> Scorer:
    """Built on first use and reused for every request (FastAPI dependency)."""
    return Scorer()



