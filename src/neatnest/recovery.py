"""Parameter recovery: what the models get back out of a known data-generating process.

This is the module that makes the rest of the project honest.

Every effect size in this repository was planted by `generate_data.py` and declared
in its `TRUE_PARAMS` dictionary. So no coefficient reported anywhere in this project
is a discovery about laundry baskets, Indian households or willingness to pay. Each
one is an ATTEMPT TO RECOVER a number that was written down first, and the only real
finding available here is how well that attempt succeeds.

That turns out to be a more interesting result than a fabricated insight would have
been, because the attempts fail in four distinct and recognisable ways:

  recovered             the confidence interval covers the planted value
  attenuated / inflated significant, but the interval misses the planted value -
                        usually because the model conditions on a mediator
  missed                the planted effect is real and the model cannot see it at
                        this sample size: a power failure, not a modelling error
  not recoverable       the planted driver is latent, or was planted through a
                        different functional form than the model assumes - so the
                        surrounding coefficients are confounded rather than wrong

Two planted nulls (`age` in the campaign model, `source[Meta paid social]`) are
included as controls: a method that cannot keep a true zero at zero is not
measuring anything.

Public entry point: `recovery_report(survey, campaign, customers)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import propensity, research
from .generate_data import PERSONAS, TRUE_PARAMS, _tp

# Verdict vocabulary, ordered worst-to-best for sorting and display.
VERDICTS = ["FALSE POSITIVE", "not recoverable", "missed", "inflated", "attenuated",
            "recovered", "correctly null"]


# ---------------------------------------------------------------------------
# verdict logic
# ---------------------------------------------------------------------------
def _verdict(true: float, est: float, lo: float, hi: float, p: float,
             observable: bool, alpha: float = 0.05) -> str:
    if not observable:
        return "not recoverable"
    significant = p < alpha
    if true == 0:
        return "correctly null" if not significant else "FALSE POSITIVE"
    if not significant:
        return "missed"
    if lo <= true <= hi:
        return "recovered"
    return "attenuated" if abs(est) < abs(true) else "inflated"


def _row(block: str, term: str, label: str, true: float, est: float, se: float,
         lo: float, hi: float, p: float, observable: bool, note: str = "") -> dict:
    v = _verdict(true, est, lo, hi, p, observable)
    return {"block": block, "term": label, "planted": true, "estimated": est,
            "std_err": se, "ci_low": lo, "ci_high": hi, "p_value": p,
            "pct_of_planted": (est / true * 100) if true else np.nan,
            "verdict": v, "note": note}


# ---------------------------------------------------------------------------
# Layer 3 - campaign purchase propensity
# ---------------------------------------------------------------------------
# statsmodels_logit() fits on raw, unscaled features, so its coefficients are
# directly comparable to the planted log-odds. Centring income and age in the
# generator shifts only the intercept, not the slopes.
CAMPAIGN_MAP = {
    "prev_home_org_purchases": "prev_home_org_purchases",
    "digital_engagement_score": "digital_engagement_score",
    "fit_quiz_completed": "fit_quiz_completed",
    "site_visits": "site_visits",
    "monthly_income_k": "monthly_income_k",
    "age": "age",
    "source_Google search": "source[Google search]",
    "source_Housing-society activation": "source[Housing-society activation]",
    "source_Meta paid social": "source[Meta paid social]",
}


def campaign_recovery(campaign: pd.DataFrame) -> pd.DataFrame:
    _, table = propensity.statsmodels_logit(campaign)
    truth = _tp("campaign_purchase")
    rows = []
    for est_name, true_name in CAMPAIGN_MAP.items():
        if est_name not in table.index:
            continue
        t, r = truth[true_name], table.loc[est_name]
        # table carries odds-ratio CIs; convert back to the log-odds scale
        rows.append(_row("Campaign purchase (logit)", true_name, true_name,
                         float(t["value"]), float(r["coef"]), float(r["std_err"]),
                         float(np.log(r["ci_low"])), float(np.log(r["ci_high"])),
                         float(r["p_value"]), bool(t.get("observable", True)),
                         str(t.get("note", ""))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 4 - customer lapse
# ---------------------------------------------------------------------------
# The PRODUCTION lapse model is a class-weighted logistic regression on
# standardised features, tuned for ranking. Its coefficients are therefore NOT
# comparable to the generator's raw log-odds. For the recovery check only, we fit
# a plain unweighted logit on raw features. This is a measurement instrument, not
# a second production model, and it is never used for scoring.
LAPSE_RECOVERY_NUM = ["care_hub_logins_m1_6", "whatsapp_open_rate", "accessory_orders_m1_6",
                      "delivery_delay_days", "product_issue_reported", "support_tickets_m1_6",
                      "referrals_m1_6", "nps_m3", "discount_pct_first_order"]
LAPSE_MAP = {
    "support_tickets_m1_6": "support_tickets_m1_6",
    "delivery_delay_days": "delivery_delay_days",
    "product_issue_reported": "product_issue_reported",
    "discount_pct_first_order": "discount_pct_first_order",
    "nps_m3": "nps_m3",
    "first_sku_HERI": "first_sku[HERI]",
    "first_sku_LITE": "first_sku[LITE]",
    "accessory_orders_m1_6": "accessory_orders_m1_6_capped3",
    "referrals_m1_6": "referrals_m1_6_capped3",
}


def lapse_recovery(customers: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(customers[LAPSE_RECOVERY_NUM + ["first_sku"]],
                       columns=["first_sku"], drop_first=True, dtype=float)
    res = sm.Logit(customers["lapsed_m7_12"], sm.add_constant(X)).fit(disp=0)
    ci = res.conf_int()
    truth = _tp("customer_lapse")
    rows = []
    for est_name, true_name in LAPSE_MAP.items():
        if est_name not in res.params.index:
            continue
        t = truth[true_name]
        rows.append(_row("Customer lapse (logit)", true_name, true_name,
                         float(t["value"]), float(res.params[est_name]),
                         float(res.bse[est_name]), float(ci.loc[est_name, 0]),
                         float(ci.loc[est_name, 1]), float(res.pvalues[est_name]),
                         bool(t.get("observable", True)), str(t.get("note", ""))))
    # The dominant planted driver is latent and has no estimate at all.
    lat = truth["latent_engagement"]
    rows.append({"block": "Customer lapse (logit)", "term": "latent_engagement",
                 "planted": float(lat["value"]), "estimated": np.nan, "std_err": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                 "pct_of_planted": np.nan, "verdict": "not recoverable",
                 "note": str(lat.get("note", ""))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 2 - factor structure
# ---------------------------------------------------------------------------
def _latent_sd(trait: str) -> float:
    """SD of a planted latent, including the persona shifts.

    latent = N(0,1) + a_k for the drawn persona k, so
        Var = 1 + E[a^2] - (E[a])^2
    with the expectation taken over the persona distribution.
    """
    LAT = TRUE_PARAMS["survey_latents"]
    probs = dict(zip(PERSONAS, LAT["persona_probs"]))
    shifts = LAT["shifts"][trait]
    ex = sum(probs[k] * shifts.get(k, 0.0) for k in PERSONAS)
    ex2 = sum(probs[k] * shifts.get(k, 0.0) ** 2 for k in PERSONAS)
    return float(np.sqrt(1.0 + ex2 - ex ** 2))


def implied_loading(nominal: float, trait: str) -> float:
    """The standardised loading an EFA can actually recover.

    item = L*f + sqrt(1 - L^2)*e with Var(f) = s^2, Var(e) = 1, so
        corr(item, f) = L*s / sqrt(L^2 s^2 + 1 - L^2)
    This exceeds the nominal L whenever s > 1, which is the case for all three
    traits here. Scoring the EFA against the nominal loading would make a correct
    extraction look like an overestimate.
    """
    s = _latent_sd(trait)
    return float(nominal * s / np.sqrt(nominal ** 2 * s ** 2 + 1 - nominal ** 2))


FACTOR_TRAIT_TO_NAME = {"digital": "Digital Engagement", "spending": "Spending Power",
                        "loyalty": "Loyalty & Promo Response"}


def factor_recovery(survey: pd.DataFrame, fa: dict | None = None) -> pd.DataFrame:
    fa = fa or research.factor_analysis(survey)
    L = fa["loadings"]
    rows = []
    for trait, items in _tp("survey_factors").items():
        col = FACTOR_TRAIT_TO_NAME[trait]
        for item, nominal in items.items():
            target = implied_loading(nominal, trait)
            est = float(L.loc[item, col])
            on_own = bool(L.loc[item].abs().idxmax() == col)
            rows.append({"block": "Factor loadings (EFA)", "item": item, "factor": col,
                         "nominal_loading": nominal, "recoverable_target": round(target, 3),
                         "estimated": round(est, 3),
                         "abs_error": round(abs(est - target), 3),
                         "loads_on_intended_factor": on_own,
                         "max_cross_loading": round(float(L.loc[item].drop(col).abs().max()), 3)})
    # 'age' is in the EFA item list but carries no planted loading at all.
    rows.append({"block": "Factor loadings (EFA)", "item": "age", "factor": "(none planted)",
                 "nominal_loading": 0.0, "recoverable_target": 0.0,
                 "estimated": round(float(L.loc["age"].abs().max()), 3),
                 "abs_error": round(float(L.loc["age"].abs().max()), 3),
                 "loads_on_intended_factor": False,
                 "max_cross_loading": round(float(L.loc["age"].abs().max()), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 1 - the constructed price band
# ---------------------------------------------------------------------------
def price_recovery(survey: pd.DataFrame, vw: dict | None = None) -> pd.DataFrame:
    """Van Westendorp recovers the planted anchor. It does not discover a price.

    The four price points are drawn as fixed multiplicative windows around
    `anchor_inr`, so the crossings are a deterministic consequence of those
    windows. Reporting the resulting band as market evidence would be circular;
    reporting how close it lands to the anchor is a legitimate check.
    """
    vw = vw or research.van_westendorp(survey)
    anchor = float(_tp("van_westendorp")["anchor_inr"]["value"])
    rows = []
    for key, label in [("PMC", "Point of marginal cheapness"), ("OPP", "Optimal price point"),
                       ("IPP", "Indifference price point"), ("PME", "Point of marginal expensiveness")]:
        est = float(vw[key])
        rows.append({"block": "Van Westendorp (constructed)", "statistic": f"{key} - {label}",
                     "planted_anchor": anchor, "estimated": est,
                     "ratio_to_anchor": round(est / anchor, 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def recovery_report(survey: pd.DataFrame, campaign: pd.DataFrame,
                    customers: pd.DataFrame, fa: dict | None = None,
                    vw: dict | None = None) -> dict:
    coef = pd.concat([campaign_recovery(campaign), lapse_recovery(customers)], ignore_index=True)
    counts = coef["verdict"].value_counts()
    summary = {v: int(counts.get(v, 0)) for v in VERDICTS if counts.get(v, 0)}
    n_scored = int((coef["verdict"] != "not recoverable").sum())
    n_ok = int(coef["verdict"].isin(["recovered", "correctly null"]).sum())
    return {
        "coefficients": coef.round(4),
        "factors": factor_recovery(survey, fa),
        "prices": price_recovery(survey, vw),
        "summary": summary,
        "n_terms": int(len(coef)),
        "n_scored": n_scored,
        "n_clean": n_ok,
        "clean_share": round(n_ok / n_scored, 3) if n_scored else np.nan,
        "false_positives": int(counts.get("FALSE POSITIVE", 0)),
    }


def headline(report: dict) -> str:
    """One sentence for the deck and the app."""
    s = report["summary"]
    biased = s.get("attenuated", 0) + s.get("inflated", 0)
    fp = report["false_positives"]
    parts = [
        f"{s.get('recovered', 0)} were recovered inside their confidence interval",
        f"{s.get('correctly null', 0)} planted nulls stayed null",
        f"{biased} came back biased" + (" (one term)" if biased == 1 else ""),
        f"{s.get('missed', 0)} real effects were missed at this sample size",
        f"{s.get('not recoverable', 0)} were structurally unrecoverable",
        f"there {'was' if fp == 1 else 'were'} {fp} false positive{'' if fp == 1 else 's'}",
    ]
    return f"Of {report['n_terms']} planted coefficients, " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


if __name__ == "__main__":
    import pandas as pd

    from .config import DATA_DIR
    rep = recovery_report(pd.read_csv(DATA_DIR / "survey.csv"),
                          pd.read_csv(DATA_DIR / "campaign.csv"),
                          pd.read_csv(DATA_DIR / "customers.csv"))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(rep["coefficients"].to_string(index=False))
    print()
    print(rep["factors"].to_string(index=False))
    print()
    print(rep["prices"].to_string(index=False))
    print()
    print(headline(rep))
