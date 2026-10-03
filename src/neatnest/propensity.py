"""Layer 3 - Acquisition: launch-campaign purchase propensity.

Two views of the same binary logistic regression:
  * statsmodels Logit on the full data for coefficients, p-values, odds ratios
  * scikit-learn Pipeline (one-hot + scaling + logistic) for honest out-of-sample
    evaluation: stratified 5-fold cross-validation plus a 25% holdout, decile lift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["age", "monthly_income_k", "digital_engagement_score", "prev_home_org_purchases",
       "site_visits", "fit_quiz_completed"]
CAT = ["source"]
TARGET = "purchased"


def statsmodels_logit(df: pd.DataFrame):
    X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=True, dtype=float)
    X = sm.add_constant(X)
    res = sm.Logit(df[TARGET], X).fit(disp=0)
    ci = res.conf_int()
    table = pd.DataFrame({"coef": res.params, "std_err": res.bse, "p_value": res.pvalues,
                          "odds_ratio": np.exp(res.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1])})
    table["significant"] = table["p_value"] < 0.05
    return res, table.drop(index="const")


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def evaluate(df: pd.DataFrame, seed: int = 42) -> dict:
    X, y = df[NUM + CAT], df[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="roc_auc")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe = build_pipeline().fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, prob)
    dec = pd.DataFrame({"prob": prob, "y": yte.values})
    dec["decile"] = pd.qcut(dec["prob"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    lift = dec.groupby("decile", observed=True).agg(prospects=("y", "size"), buyers=("y", "sum"),
                                                     conv_rate=("y", "mean"))
    lift["lift"] = lift["conv_rate"] / dec["y"].mean()
    lift["cum_capture"] = lift["buyers"].cumsum() / lift["buyers"].sum()
    final = build_pipeline().fit(X, y)
    return {"cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
            "holdout": {"auc": float(roc_auc_score(yte, prob)), "accuracy": float(accuracy_score(yte, pred)),
                        "precision": float(precision_score(yte, pred)), "recall": float(recall_score(yte, pred)),
                        "f1": float(f1_score(yte, pred)), "n_test": int(len(yte))},
            "confusion": confusion_matrix(yte, pred).tolist(), "roc": (fpr, tpr), "lift": lift,
            "base_rate": float(y.mean()), "model": final}


def score_prospect(model: Pipeline, **features) -> float:
    row = pd.DataFrame([{k: features[k] for k in NUM + CAT}])
    return float(model.predict_proba(row)[:, 1][0])
