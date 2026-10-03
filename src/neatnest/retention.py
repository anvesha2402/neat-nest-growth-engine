"""Layer 4 - Retention and customer value.

For a durable product, "churn" is redefined as LAPSE: a registered FlexFold
owner who, in months 7-12 after purchase, buys no accessory, makes no referral
and does not engage with the care hub. Features are a month-6 snapshot, the
outcome is measured afterwards, so there is no look-ahead leakage.

Models
  * Lapse classifier: logistic baseline vs Random Forest (sklearn Pipelines)
  * 12-month customer value: OLS (interpretable) vs Ridge/Lasso (CV penalty)
  * Behavioural segments: K-Means on standardised engagement/value features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
                             mean_squared_error)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["age", "care_hub_logins_m1_6", "whatsapp_open_rate", "accessory_orders_m1_6",
       "accessory_spend_m1_6", "delivery_delay_days", "product_issue_reported", "support_tickets_m1_6",
       "review_written", "referrals_m1_6", "nps_m3", "discount_pct_first_order"]
CAT = ["first_sku", "first_channel", "city"]
TARGET = "lapsed_m7_12"
VALUE = "value_12m"
SEG_FEATURES = ["care_hub_logins_m1_6", "whatsapp_open_rate", "accessory_spend_m1_6",
                "referrals_m1_6", "support_tickets_m1_6", "nps_m3"]


def _pre(scale: bool = True) -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


def lapse_models(df: pd.DataFrame, seed: int = 42) -> dict:
    X, y = df[NUM + CAT], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    models = {
        "Logistic (balanced)": Pipeline([("pre", _pre()), ("clf", LogisticRegression(max_iter=3000,
                                                                                      class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", _pre(False)), ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20, class_weight="balanced",
            random_state=seed, n_jobs=-1))]),
    }
    out, probs = {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name] = p
        out[name] = {"auc": roc_auc_score(yte, p), "accuracy": accuracy_score(yte, pred),
                     "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
                     "f1": f1_score(yte, pred),
                     "cv_auc": float(cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean())}
    best = max(out, key=lambda k: out[k]["auc"])
    rf = models["Random Forest"]
    imp = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=NUM + CAT).sort_values(ascending=False)
    # odds ratios from the logistic model (per 1 SD)
    lr = models["Logistic (balanced)"]
    names = lr.named_steps["pre"].get_feature_names_out()
    odds = pd.Series(np.exp(lr.named_steps["clf"].coef_[0]), index=[n.split("__")[1] for n in names])
    # decile capture for the best model
    p = probs[best]
    dec = pd.DataFrame({"p": p, "y": yte.values}).sort_values("p", ascending=False)
    dec["decile"] = np.repeat(np.arange(1, 11), int(np.ceil(len(dec) / 10)))[:len(dec)]
    lift = dec.groupby("decile").agg(customers=("y", "size"), lapsers=("y", "sum"), rate=("y", "mean"))
    lift["cum_capture"] = lift["lapsers"].cumsum() / lift["lapsers"].sum()
    final = models[best].fit(X, y)
    return {"metrics": pd.DataFrame(out).T, "best": best, "importance": importance, "odds_ratios": odds,
            "lift": lift, "roc": {k: roc_curve(yte, v)[:2] for k, v in probs.items()},
            "model": final, "base_rate": float(y.mean())}


def value_models(df: pd.DataFrame, seed: int = 42) -> dict:
    # accessory_orders duplicates accessory_spend (VIF > 30), so it is left out here
    feats = [c for c in NUM if c != "accessory_orders_m1_6"]
    X = pd.get_dummies(df[feats + CAT], columns=CAT, drop_first=True, dtype=float)
    y = df[VALUE]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    res = {}
    for name, m in {"OLS": LinearRegression(), "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 30)),
                    "Lasso": LassoCV(cv=5, random_state=seed, max_iter=20000)}.items():
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        res[name] = {"r2": r2_score(yte, pred), "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                     "mae": mean_absolute_error(yte, pred),
                     "n_zero_coef": int((np.abs(getattr(m, "coef_", np.zeros(1))) < 1e-8).sum())}
    ols = sm.OLS(ytr.values, sm.add_constant(pd.DataFrame(Xtr_s, columns=X.columns))).fit()
    coefs = pd.DataFrame({"coef_per_sd": ols.params.drop("const"), "p_value": ols.pvalues.drop("const")})
    coefs = coefs.reindex(coefs["coef_per_sd"].abs().sort_values(ascending=False).index)
    # VIF on training predictors
    Xv = sm.add_constant(pd.DataFrame(Xtr_s, columns=X.columns))
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])], index=X.columns)
    return {"metrics": pd.DataFrame(res).T, "coefficients": coefs, "vif_max": float(vif.max()),
            "adj_r2_train": float(ols.rsquared_adj)}


def segments(df: pd.DataFrame, k: int = 4, seed: int = 42) -> dict:
    Z = StandardScaler().fit_transform(df[SEG_FEATURES])
    sil = {kk: float(silhouette_score(Z, KMeans(kk, n_init=10, random_state=seed).fit_predict(Z),
                                      sample_size=4000, random_state=seed)) for kk in range(2, 8)}
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_
    prof = df.assign(segment=lab).groupby("segment").agg(
        customers=("customer_id", "size"), lapse_rate=(TARGET, "mean"), value_12m=(VALUE, "mean"),
        **{f: (f, "mean") for f in SEG_FEATURES})
    prof["share"] = prof["customers"] / len(df)
    names = name_segments(prof)
    prof.index = [names[i] for i in prof.index]
    return {"profile": prof.round(3), "silhouette": sil, "labels": pd.Series([names[i] for i in lab], index=df.index)}


def name_segments(prof: pd.DataFrame) -> dict:
    """Deterministic naming rules so labels survive re-seeding."""
    names = {}
    order = prof.sort_values("value_12m", ascending=False).index.tolist()
    worst = prof["support_tickets_m1_6"].idxmax()
    for s in prof.index:
        if s == worst:
            names[s] = "Service-scarred"
        elif s == order[0]:
            names[s] = "Nest Advocates"
        elif prof.loc[s, "lapse_rate"] == prof.drop(index=[worst, order[0]])["lapse_rate"].max():
            names[s] = "Quiet One-timers"
        else:
            names[s] = "Steady Replenishers"
    return names


def retention_scenario(df: pd.DataFrame, probs: np.ndarray, contact_share: float = 0.3,
                       save_rate: float = 0.2, accessory_margin_per_save: float = 900.0) -> dict:
    """What a targeted save programme protects (save rate is an ASSUMPTION to test)."""
    d = df.assign(p=probs).sort_values("p", ascending=False)
    top = d.head(int(len(d) * contact_share))
    captured = top[TARGET].sum()
    saved = captured * save_rate
    return {"contacted": int(len(top)), "lapsers_reached": int(captured),
            "share_of_lapsers_reached": float(captured / df[TARGET].sum()),
            "customers_saved": float(saved), "margin_protected": float(saved * accessory_margin_per_save)}
