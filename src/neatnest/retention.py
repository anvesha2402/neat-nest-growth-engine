"""Layer 4 - Retention and customer value.

For a durable product, "churn" is redefined as LAPSE: a registered FlexFold
owner who, in months 7-12 after purchase, buys no accessory, makes no referral
and does not engage with the care hub. Features are a month-6 snapshot and the
outcome is measured afterwards, so the FEATURES do not look ahead.

Three disclosures belong at the top of this file, because each one changes how
its outputs should be read.

1. TWO VALUE TARGETS, AND ONE OF THEM IS LEAKY.
   `value_12m` is total realised 12-month value, and it is built from three
   things that are also model features: the first-order price (set by
   `first_sku`), `accessory_spend_m1_6` and `referrals_m1_6`. Regressing it on
   those features recovers an accounting identity. Its R-squared near 0.93 is
   target leakage, not predictive skill, and it is kept only to demonstrate
   exactly that. `value_fwd_7_12` holds only accessory spend from months 7-12,
   so nothing used to predict it sits inside it. Its much lower R-squared is
   the honest answer to "what can a month-6 snapshot tell us about future
   value", and `value_model_comparison()` publishes the two side by side.

2. SCORING IS OUT OF FOLD.
   `lapse_models()` fits on a train split for evaluation and then refits on all
   rows to return a deployable model. Scoring every customer with that refitted
   model - which is what the earlier version of the pipeline did - produces
   in-sample probabilities, and the save-programme scenario built on them was
   therefore optimistic. `oof_lapse_scores()` gives every customer a score from
   a fold in which they were held out, and that is what the pipeline now uses.

3. K = 4 IS A PRESENTATION CHOICE, NOT A FIT RESULT.
   The silhouette score prefers k = 2. Four segments are retained because they
   map onto distinguishable commercial actions, and the full silhouette curve is
   published alongside so the reader can see the price of that choice.
   `segment_choice()` states it in one line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["age", "care_hub_logins_m1_6", "whatsapp_open_rate", "accessory_orders_m1_6",
       "accessory_spend_m1_6", "delivery_delay_days", "product_issue_reported", "support_tickets_m1_6",
       "review_written", "referrals_m1_6", "nps_m3", "discount_pct_first_order"]
CAT = ["first_sku", "first_channel", "city"]
TARGET = "lapsed_m7_12"
VALUE = "value_12m"              # leaky by construction - see disclosure 1
VALUE_FWD = "value_fwd_7_12"     # forward-only, the honest target
SEG_FEATURES = ["care_hub_logins_m1_6", "whatsapp_open_rate", "accessory_spend_m1_6",
                "referrals_m1_6", "support_tickets_m1_6", "nps_m3"]

#: Features that appear inside `value_12m` and therefore leak into it.
VALUE_LEAKS = ["accessory_spend_m1_6", "accessory_orders_m1_6", "referrals_m1_6", "first_sku"]


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
                     "cv_auc": float(cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()),
                     # Calibration, measured rather than assumed. Both models are
                     # class-weighted for ranking, which deliberately distorts the
                     # probability scale - so the Brier score is reported, and an
                     # isotonic recalibration is fitted to show what it would cost
                     # to make these numbers usable as probabilities.
                     "brier": float(brier_score_loss(yte, p)),
                     "brier_calibrated": _brier_after_isotonic(m, Xtr, ytr, Xte, yte, seed)}
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
    # Reliability curve for the best model, on the held-out split.
    frac_pos, mean_pred = calibration_curve(yte, probs[best], n_bins=10, strategy="quantile")
    final = models[best].fit(X, y)
    return {"metrics": pd.DataFrame(out).T, "best": best, "importance": importance, "odds_ratios": odds,
            "lift": lift, "roc": {k: roc_curve(yte, v)[:2] for k, v in probs.items()},
            "reliability": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
            "model": final, "base_rate": float(y.mean()),
            "note": ("`model` is refitted on all rows and is for deployment. Do NOT score the "
                     "training rows with it - use oof_lapse_scores() for anything that reports "
                     "how well targeting would have worked.")}


def _brier_after_isotonic(model, Xtr, ytr, Xte, yte, seed: int) -> float:
    """Brier score after isotonic recalibration, fitted on the training split only."""
    cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)
    return float(brier_score_loss(yte, cal.predict_proba(Xte)[:, 1]))


def oof_lapse_scores(df: pd.DataFrame, model=None, seed: int = 42, folds: int = 5) -> np.ndarray:
    """Out-of-fold lapse probability for every customer.

    Each row is scored by a model fitted without it, so the risk table and the
    save-programme scenario are honest about what targeting would actually have
    achieved. Scoring with a model fitted on all rows - the earlier behaviour -
    inflates the apparent capture rate.
    """
    X, y = df[NUM + CAT], df[TARGET]
    if model is None:
        model = Pipeline([("pre", _pre()), ("clf", LogisticRegression(max_iter=3000,
                                                                     class_weight="balanced"))])
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def value_model_comparison(df: pd.DataFrame, seed: int = 42) -> dict:
    """Fit the same specification to a leaky target and an honest one.

    This is the clearest teaching artefact in the project. Identical features,
    identical models, two targets:

      value_12m       contains the first-order price, accessory spend in months
                      1-6 and referrals in months 1-6 - all of which are model
                      features. R-squared comes back near 0.93.
      value_fwd_7_12  accessory spend in months 7-12 only. No feature appears
                      inside it. R-squared collapses.

    The gap between the two is the size of the leak. A reader who sees only the
    first number would conclude that customer value is nearly fully predictable
    from a month-6 snapshot. It is not.
    """
    leaky = value_models(df, seed=seed, target=VALUE)
    honest = value_models(df, seed=seed, target=VALUE_FWD)
    comp = pd.DataFrame({
        "target": [VALUE, VALUE_FWD],
        "describes": ["total realised 12-month value", "accessory value in months 7-12 only"],
        "contains_model_features": [True, False],
        "leaking_features": [", ".join(VALUE_LEAKS), "none"],
        "r2_ols": [leaky["metrics"].loc["OLS", "r2"], honest["metrics"].loc["OLS", "r2"]],
        "r2_ridge": [leaky["metrics"].loc["Ridge", "r2"], honest["metrics"].loc["Ridge", "r2"]],
        "rmse_ols": [leaky["metrics"].loc["OLS", "rmse"], honest["metrics"].loc["OLS", "rmse"]],
        "mean_target": [df[VALUE].mean(), df[VALUE_FWD].mean()],
        "zero_share": [float((df[VALUE] == 0).mean()), float((df[VALUE_FWD] == 0).mean())],
        "interpretation": ["target leakage - an accounting identity, not a prediction",
                           "honest predictive ceiling from a month-6 snapshot"],
    })
    r2_drop = float(leaky["metrics"].loc["OLS", "r2"] - honest["metrics"].loc["OLS", "r2"])
    return {
        "leaky": leaky, "forward": honest, "comparison": comp.round(4),
        "r2_drop": round(r2_drop, 4),
        "headline": (f"Removing the leakage drops OLS R-squared from "
                     f"{leaky['metrics'].loc['OLS', 'r2']:.3f} to "
                     f"{honest['metrics'].loc['OLS', 'r2']:.3f}. The first number was measuring an "
                     f"accounting identity; the second is what a month-6 snapshot actually supports."),
        "caveat": (f"{float((df[VALUE_FWD] == 0).mean()):.0%} of customers have zero forward value, so a "
                   "linear model on this target is mis-specified at the floor. A two-part model "
                   "(lapse classifier, then value among the active) is the right specification and is "
                   "not fitted here - the point being made is about the leak, not about squeezing the "
                   "last of the signal out of this target."),
    }


def value_models(df: pd.DataFrame, seed: int = 42, target: str = VALUE) -> dict:
    # accessory_orders duplicates accessory_spend (VIF > 30), so it is left out here
    feats = [c for c in NUM if c != "accessory_orders_m1_6"]
    X = pd.get_dummies(df[feats + CAT], columns=CAT, drop_first=True, dtype=float)
    y = df[target]
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
            "adj_r2_train": float(ols.rsquared_adj), "target": target,
            "leaky": target == VALUE}


def segments(df: pd.DataFrame, k: int = 4, seed: int = 42, k_max: int = 9) -> dict:
    """K-Means segments, published with the full silhouette curve.

    k defaults to 4 for interpretability, which is NOT the value the fit
    prefers. The curve and `choice` make that explicit rather than leaving the
    reader to assume four segments emerged from the data.
    """
    Z = StandardScaler().fit_transform(df[SEG_FEATURES])
    sil = {kk: float(silhouette_score(Z, KMeans(kk, n_init=10, random_state=seed).fit_predict(Z),
                                      sample_size=4000, random_state=seed))
           for kk in range(2, k_max + 1)}
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_
    prof = df.assign(segment=lab).groupby("segment").agg(
        customers=("customer_id", "size"), lapse_rate=(TARGET, "mean"), value_12m=(VALUE, "mean"),
        value_fwd_7_12=(VALUE_FWD, "mean"),
        **{f: (f, "mean") for f in SEG_FEATURES})
    prof["share"] = prof["customers"] / len(df)
    names = name_segments(prof)
    prof.index = [names[i] for i in prof.index]
    return {"profile": prof.round(3), "silhouette": sil, "k": k,
            "choice": segment_choice(sil, k),
            "labels": pd.Series([names[i] for i in lab], index=df.index)}


def segment_choice(sil: dict, k: int) -> dict:
    """State the k trade-off in numbers instead of leaving it implicit."""
    best_k = max(sil, key=sil.get)
    return {
        "k_used": k,
        "k_preferred_by_silhouette": int(best_k),
        "silhouette_at_k_used": round(sil[k], 4),
        "silhouette_at_best_k": round(sil[best_k], 4),
        "silhouette_cost_of_choice": round(sil[best_k] - sil[k], 4),
        "rationale": ("k = 2 separates the data best, but splits owners only into engaged and "
                      "disengaged, which supports one action. k = 4 scores worse and distinguishes "
                      "four groups that take different interventions - a service-recovery group, a "
                      "high-value advocacy group, a dormant group and a steady replenishment group. "
                      "The segments are a planning device, and the silhouette cost above is what "
                      "that device costs in separation terms."),
        "honest_summary": (f"Silhouette prefers k={best_k} ({sil[best_k]:.3f}); this project uses "
                           f"k={k} ({sil[k]:.3f}), giving up {sil[best_k] - sil[k]:.3f} of separation "
                           f"for commercial interpretability."),
    }


def name_segments(prof: pd.DataFrame) -> dict:
    """Deterministic, collision-free naming that works for any k.

    The earlier version compared floats with `==` to pick the high-lapse
    segment, and could assign the same name twice when one segment was both the
    highest-value and the worst-serviced. This version claims each name by rank
    from a shrinking pool, so every segment gets exactly one label and the
    labels survive re-seeding.
    """
    pool = list(prof.index)
    names: dict = {}

    def claim(column: str, label: str, biggest: bool = True) -> None:
        if not pool:
            return
        s = prof.loc[pool, column]
        pick = s.idxmax() if biggest else s.idxmin()
        names[pick] = label
        pool.remove(pick)

    claim("support_tickets_m1_6", "Service-scarred")
    claim("value_12m", "Nest Advocates")
    claim("lapse_rate", "Quiet One-timers")
    # Anything left is a replenishment group, ordered by value for stable suffixes.
    rest = prof.loc[pool, "value_12m"].sort_values(ascending=False).index.tolist()
    for i, s in enumerate(rest):
        names[s] = "Steady Replenishers" if i == 0 else f"Steady Replenishers {i + 1}"
    return names


def retention_scenario(df: pd.DataFrame, probs: np.ndarray, contact_share: float = 0.3,
                       save_rate: float = 0.2, accessory_margin_per_save: float = 900.0,
                       scores_are_oof: bool = True) -> dict:
    """What a targeted save programme protects.

    Pass OUT-OF-FOLD probabilities. Scoring customers with a model that was
    fitted on them inflates the share of lapsers the top decile appears to
    reach, and therefore the margin this scenario claims to protect.

    Two of the three inputs here are assumptions with no empirical basis:
    `save_rate` (that one contacted lapser in five is retained) and
    `accessory_margin_per_save`. The reachable share is a model output; the
    rupee figure is an assumption stack on top of it, and should be read as an
    order of magnitude, not a forecast.
    """
    d = df.assign(p=probs).sort_values("p", ascending=False)
    top = d.head(int(len(d) * contact_share))
    captured = top[TARGET].sum()
    saved = captured * save_rate
    return {"contacted": int(len(top)), "lapsers_reached": int(captured),
            "share_of_lapsers_reached": float(captured / df[TARGET].sum()),
            "customers_saved": float(saved), "margin_protected": float(saved * accessory_margin_per_save),
            "scores_are_oof": bool(scores_are_oof),
            "assumptions": {"save_rate": save_rate,
                            "accessory_margin_per_save": accessory_margin_per_save,
                            "basis": "both ASSUMED - no save programme has been run"},
            "lift_vs_random": float((captured / df[TARGET].sum()) / contact_share)}
