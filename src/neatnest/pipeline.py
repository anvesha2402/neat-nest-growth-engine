"""End-to-end build: simulate data -> run all four layers -> save artifacts.

    python -m neatnest.pipeline          (from the repo root with src/ on PYTHONPATH)

Outputs
  data/*.csv                 the simulated universe
  models/*.joblib            propensity pipeline, lapse pipeline
  reports/results.json       every headline number used by the app and the brief
  reports/risk_table.csv     owner-level lapse risk + segment (copilot tool input)
  reports/figures/*.png      static charts for the project brief
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from . import market, propensity, research, retention
from .config import DATA_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR, SEED
from .generate_data import build_all

warnings.filterwarnings("ignore")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="index"))
    if isinstance(o, pd.Series):
        return json.loads(o.to_json())
    raise TypeError(type(o))


def run(make_figures: bool = True) -> dict:
    for d in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    frames = build_all(DATA_DIR, SEED)
    survey, campaign, customers, sales = (frames[k] for k in ("survey", "campaign", "customers", "sales"))
    R: dict = {"seed": SEED}

    # Layer 1 - market & economics
    R["market"] = market.market_sizing()
    R["unit_economics"] = market.unit_economics_table().set_index(["sku", "channel"]).round(2)
    R["som_plan"] = market.som_plan().set_index("year").round(3)

    # Layer 2 - customer understanding
    fa = research.factor_analysis(survey)
    R["factors"] = {"kmo": fa["kmo"], "bartlett_chi2": fa["bartlett_chi2"], "bartlett_dof": fa["bartlett_dof"],
                    "bartlett_p": fa["bartlett_p"], "eigenvalues": fa["eigenvalues"], "n_factors": fa["n_factors"],
                    "loadings": fa["loadings"].round(3), "communality": fa["communality"].round(3),
                    "variance_pct": fa["variance_pct"].round(2), "cumulative_pct": fa["cumulative_pct"]}
    R["personas"] = research.persona_profiles(survey, fa["scores"])
    vw = research.van_westendorp(survey)
    R["van_westendorp"] = {k: vw[k] for k in ("PMC", "OPP", "IPP", "PME")}
    R["kano"] = research.kano(survey).set_index("feature")
    R["survey_n"] = len(survey)

    # Layer 3 - acquisition
    res, table = propensity.statsmodels_logit(campaign)
    ev = propensity.evaluate(campaign)
    R["propensity"] = {"n": len(campaign), "base_rate": ev["base_rate"], "pseudo_r2": res.prsquared,
                       "llr_p": res.llr_pvalue, "coefficients": table.round(4), "cv_auc_mean": ev["cv_auc_mean"],
                       "cv_auc_std": ev["cv_auc_std"], "holdout": ev["holdout"], "confusion": ev["confusion"],
                       "lift": ev["lift"].round(4)}
    joblib.dump(ev["model"], MODEL_DIR / "propensity.joblib")

    # Layer 4 - retention & value
    lm = retention.lapse_models(customers)
    vm = retention.value_models(customers)
    sg = retention.segments(customers)
    probs = lm["model"].predict_proba(customers[retention.NUM + retention.CAT])[:, 1]
    scen = retention.retention_scenario(customers, probs)
    R["lapse"] = {"n": len(customers), "base_rate": lm["base_rate"], "metrics": lm["metrics"].round(4),
                  "best": lm["best"], "importance": lm["importance"].round(4),
                  "odds_ratios": lm["odds_ratios"].round(3), "lift": lm["lift"].round(4), "scenario": scen}
    R["value"] = {"metrics": vm["metrics"].round(3), "coefficients": vm["coefficients"].head(10).round(3),
                  "vif_max": vm["vif_max"], "adj_r2_train": vm["adj_r2_train"]}
    R["segments"] = {"profile": sg["profile"], "silhouette": sg["silhouette"]}
    joblib.dump(lm["model"], MODEL_DIR / "lapse.joblib")
    risk = customers[["customer_id", "city", "first_sku", "first_channel", "support_tickets_m1_6",
                      "product_issue_reported", "nps_m3", "accessory_orders_m1_6"]].copy()
    risk["lapse_risk"] = probs.round(4)
    risk["segment"] = sg["labels"].values
    risk.sort_values("lapse_risk", ascending=False).to_csv(REPORT_DIR / "risk_table.csv", index=False)
    sg["profile"].to_csv(REPORT_DIR / "segments.csv")
    ev["lift"].to_csv(REPORT_DIR / "propensity_lift.csv")

    # Sales (simulated actuals) vs plan
    s = sales.assign(month=sales["date"].str[:7])
    core = s[s["family"] == "Core"]
    R["sales"] = {"net_sales_cr": s["net_sales"].sum() / 1e7, "core_units": int(core["units"].sum()),
                  "accessory_units": int(s.loc[s["family"] == "Accessories", "units"].sum()),
                  "gross_margin_pct": (s["net_sales"].sum() - s["cost"].sum()) / s["net_sales"].sum() * 100,
                  "by_city": s.groupby("city")["net_sales"].sum().round(0),
                  "by_channel": s.groupby("channel")["net_sales"].sum().round(0),
                  "by_product_units": s.groupby("product")["units"].sum(),
                  "monthly": s.groupby("month")["net_sales"].sum().round(0),
                  "lines": len(s)}

    # Copilot regression suite
    from .copilot.evaluate import run_suite
    R["copilot_eval"] = run_suite()

    (REPORT_DIR / "results.json").write_text(json.dumps(R, default=_jsonable, indent=1))
    if make_figures:
        from .figures import make_all
        make_all(R, frames, fa, vw, ev, lm, sg)
    return R


if __name__ == "__main__":
    out = run()
    print(json.dumps({"market": out["market"], "copilot": out["copilot_eval"]["summary"]}, default=_jsonable, indent=1))
