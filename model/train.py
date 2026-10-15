"""
SentinX - train baseline + XGBoost models for the disruption alert and fragility score, explain with SHAP.

Input : data/processed/model_table.csv
Output: model/artifacts/{*.joblib, metrics.json, shap_summary.png, shap_values.csv}
        data/processed/predictions.csv

Usage (from repo root):
    python model/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLE_PATH = Path("data/processed/model_table.csv")
PRED_PATH = Path("data/processed/predictions.csv")
ART = Path("model/artifacts")
SEED = 42

FEATURES = [
    # unrest (ACLED)
    "unrest_per_m", "fatalities_per_m", "violent_per_m", "riot_share",
    "unrest_yoy", "violent_events_yoy", "fatalities_yoy",
    # macro (WDI)
    "gdp_growth", "inflation", "trade_pct_gdp", "export_growth", "manuf_pct_gdp",
    "fdi_pct_gdp", "log_gdp_pc", "unemployment",
    "gdp_growth_vol3", "inflation_vol3",
    "gdp_growth_lag1", "inflation_lag1", "trade_pct_gdp_lag1", "export_growth_lag1",
    "manuf_pct_gdp_lag1", "fdi_pct_gdp_lag1", "log_gdp_pc_lag1", "unemployment_lag1",
]
CLS_TARGET, REG_TARGET = "disruption_next", "fragility_next"
# TODO(prototype): fixed scale - a 10pp growth shortfall maps to score 100. Calibrate later.
SCORE_CAP_PP = 10.0


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Temporal split on feature-year. Never shuffled."""
    lab = df[~df["is_forecast_row"] & df[CLS_TARGET].notna()]
    return {
        "train": lab[lab["year"] <= 2019],
        "val": lab[lab["year"].between(2020, 2021)],
        "test": lab[lab["year"].between(2022, 2023)],
        "forecast": df[df["is_forecast_row"]],
    }


def make_models(pos_weight: float) -> tuple[dict, dict]:
    impute = lambda: SimpleImputer(strategy="median")  # fit on train only (inside the pipeline)
    classifiers = {
        "logistic": make_pipeline(impute(), StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "random_forest": make_pipeline(impute(), RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=SEED)),
        "xgboost": XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=pos_weight,
                                 eval_metric="logloss", random_state=SEED),
    }
    regressors = {
        "ridge": make_pipeline(impute(), StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": make_pipeline(impute(), RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=SEED)),
        "xgboost": XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED),
    }
    return classifiers, regressors


def cls_metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    one_class = len(np.unique(y)) < 2  # AUC undefined if a split has only one class
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": None if one_class else round(roc_auc_score(y, p), 3),
        "pr_auc": None if one_class else round(average_precision_score(y, p), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
    }


def reg_metrics(y, p) -> dict:
    return {"n": int(len(y)), "mae": round(mean_absolute_error(y, p), 3), "r2": round(r2_score(y, p), 3)}


def to_score(shortfall_pp: np.ndarray) -> np.ndarray:
    return np.clip(shortfall_pp / SCORE_CAP_PP, 0, 1) * 100


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE_PATH)
    s = split(df)
    for k, v in s.items():
        pos = f", positives={int(v[CLS_TARGET].sum())}" if k != "forecast" else ""
        print(f"{k:>8}: {len(v):3d} rows, years {v['year'].min()}-{v['year'].max()}{pos}")

    tr = s["train"]
    pos_weight = float((tr[CLS_TARGET] == 0).sum() / max((tr[CLS_TARGET] == 1).sum(), 1))
    classifiers, regressors = make_models(pos_weight)

    metrics = {"classification": {}, "regression": {}, "split": {k: len(v) for k, v in s.items()},
               "features": FEATURES}
    for name, m in classifiers.items():
        m.fit(tr[FEATURES], tr[CLS_TARGET].astype(int))
        metrics["classification"][name] = {
            k: cls_metrics(s[k][CLS_TARGET].astype(int).values, m.predict_proba(s[k][FEATURES])[:, 1])
            for k in ["val", "test"]}
    reg_tr = tr[tr[REG_TARGET].notna()]
    for name, m in regressors.items():
        m.fit(reg_tr[FEATURES], reg_tr[REG_TARGET])
        metrics["regression"][name] = {}
        for k in ["val", "test"]:
            part = s[k][s[k][REG_TARGET].notna()]
            metrics["regression"][name][k] = reg_metrics(part[REG_TARGET].values, m.predict(part[FEATURES]))

    # ---- comparison table ----
    rows = []
    for name, d in metrics["classification"].items():
        for k, v in d.items():
            rows.append({"task": "alert", "model": name, "split": k, **{x: v[x] for x in
                         ["n", "positives", "roc_auc", "pr_auc", "precision", "recall", "f1"]}})
    for name, d in metrics["regression"].items():
        for k, v in d.items():
            rows.append({"task": "score", "model": name, "split": k, **v})
    table = pd.DataFrame(rows)
    print("\n=== Model comparison (temporal holdout; train <=2019, val 2020-21, test 2022-23) ===")
    print(table.astype({"positives": "Int64"}).to_string(index=False, na_rep="-"))

    # ---- final XGBoost models: refit on all labelled rows for the forecast ----
    # TODO(prototype): metrics above come from the <=2019 fit; the served model is refit on 2010-2024 labels.
    lab = pd.concat([s["train"], s["val"], s["test"],
                     df[~df["is_forecast_row"] & df[CLS_TARGET].notna() & (df["year"] > 2023)]])
    all_pw = float((lab[CLS_TARGET] == 0).sum() / max((lab[CLS_TARGET] == 1).sum(), 1))
    clf = XGBClassifier(**{**classifiers["xgboost"].get_params(), "scale_pos_weight": all_pw})
    clf.fit(lab[FEATURES], lab[CLS_TARGET].astype(int))
    lab_r = lab[lab[REG_TARGET].notna()]
    reg = XGBRegressor(**regressors["xgboost"].get_params())
    reg.fit(lab_r[FEATURES], lab_r[REG_TARGET])

    for name, m in classifiers.items():
        joblib.dump(m, ART / f"alert_{name}_eval.joblib")
    for name, m in regressors.items():
        joblib.dump(m, ART / f"score_{name}_eval.joblib")
    joblib.dump(clf, ART / "alert_xgboost.joblib")
    joblib.dump(reg, ART / "score_xgboost.joblib")
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- predictions for every row (incl. forecast rows) ----
    pred = df[["iso3", "country", "year", "is_forecast_row"]].copy()
    pred["fragility_raw_pp"] = reg.predict(df[FEATURES]).astype(float).round(3)
    pred["fragility_score"] = to_score(pred["fragility_raw_pp"].values).round(1)
    pred["alert_prob"] = clf.predict_proba(df[FEATURES])[:, 1].astype(float).round(4)
    pred["alert"] = (pred["alert_prob"] >= 0.5).astype(int)
    pred.to_csv(PRED_PATH, index=False)

    # ---- SHAP on the XGBoost fragility-score model (drivers of the score) ----
    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(df[FEATURES])
    shap_df = pd.DataFrame(sv, columns=FEATURES)
    shap_df.insert(0, "year", df["year"].values)
    shap_df.insert(0, "iso3", df["iso3"].values)
    shap_df.to_csv(ART / "shap_values.csv", index=False)
    # alert-model SHAP too (log-odds units)
    sv_cls = shap.TreeExplainer(clf).shap_values(df[FEATURES])
    pd.concat([df[["iso3", "year"]].reset_index(drop=True), pd.DataFrame(sv_cls, columns=FEATURES)], axis=1) \
        .to_csv(ART / "shap_values_alert.csv", index=False)

    shap.summary_plot(sv, df[FEATURES], plot_type="bar", show=False, max_display=15)
    plt.title("SentinX - global drivers of fragility score (XGBoost, mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(ART / "shap_summary.png", dpi=150)
    plt.close()

    imp = shap_df[FEATURES].abs().mean().sort_values(ascending=False)
    print("\nTop 8 global SHAP drivers (fragility score):")
    print(imp.head(8).round(3).to_string())
    fc = pred[pred["is_forecast_row"]].sort_values("fragility_score", ascending=False)
    print(f"\nWrote {PRED_PATH} ({len(pred)} rows). Latest-year forecast, top 8 by score:")
    print(fc[["iso3", "country", "year", "fragility_score", "alert_prob", "alert"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
