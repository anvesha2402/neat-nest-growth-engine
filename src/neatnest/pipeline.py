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

from . import market, opex, propensity, recovery, research, retention
from .config import (DATA_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR, SEED,
                     check_registry, kind_counts, params_frame)
from .generate_data import build_all

warnings.filterwarnings("ignore")


def _provenance() -> dict:
    """Stamp every result set with enough to reproduce it."""
    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    import sklearn
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5, cwd=REPORT_DIR.parent).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    timeout=5, cwd=REPORT_DIR.parent).stdout.strip())
    except Exception:                                   # pragma: no cover
        commit, dirty = None, None
    return {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED, "git_commit": commit, "git_dirty": dirty,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}


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
    R: dict = {"seed": SEED, "provenance": _provenance()}

    # Parameter registry: fail loudly if a published value has drifted from the
    # value the code actually uses.
    drift = check_registry()
    if drift:
        raise RuntimeError("Parameter registry is out of sync with the code:\n  " + "\n  ".join(drift))
    R["parameters"] = {"n": int(len(params_frame())), "by_kind": kind_counts(),
                       "note": ("Each input is tagged SOURCED, BENCHMARK or ASSUMED. Only the SOURCED "
                                "count traces to a published source; the rest are benchmark inputs and "
                                "assumptions, and the Parameters tab shows every one with its range.")}

    # Layer 1 - market & economics
    R["market"] = market.market_sizing()
    R["unit_economics"] = market.unit_economics_table("Y1").set_index(["sku", "channel"]).round(2)
    R["som_plan"] = market.som_plan().set_index("year").round(3)
    R["ebitda_bridge"] = opex.ebitda_bridge().set_index("year").round(4)
    R["opex"] = opex.opex_table().round(0)
    R["break_even"] = opex.break_even_table().round(2)
    R["tornado"] = opex.tornado("Y3").round(4)
    R["breaking_points"] = opex.breaking_points("Y3")
    R["economics_summary"] = opex.summary()

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
    vc = retention.value_model_comparison(customers)
    vm = vc["leaky"]
    sg = retention.segments(customers)
    # Out-of-fold scores: every customer is ranked by a model fitted without them,
    # so the risk table and the save scenario are not graded on their own training
    # data. The in-sample version is kept only to publish the difference.
    probs = retention.oof_lapse_scores(customers)
    probs_in_sample = lm["model"].predict_proba(customers[retention.NUM + retention.CAT])[:, 1]
    scen = retention.retention_scenario(customers, probs, scores_are_oof=True)
    scen_in_sample = retention.retention_scenario(customers, probs_in_sample, scores_are_oof=False)
    R["lapse"] = {"n": len(customers), "base_rate": lm["base_rate"], "metrics": lm["metrics"].round(4),
                  "best": lm["best"], "importance": lm["importance"].round(4),
                  "odds_ratios": lm["odds_ratios"].round(3), "lift": lm["lift"].round(4),
                  "reliability": lm["reliability"], "scenario": scen,
                  "scenario_in_sample": scen_in_sample,
                  "in_sample_overstatement_pct": round(
                      (scen_in_sample["share_of_lapsers_reached"] / scen["share_of_lapsers_reached"] - 1) * 100, 3),
                  "scoring_note": lm["note"]}
    R["value"] = {"metrics": vm["metrics"].round(3), "coefficients": vm["coefficients"].head(10).round(3),
                  "vif_max": vm["vif_max"], "adj_r2_train": vm["adj_r2_train"],
                  "leakage_comparison": vc["comparison"], "r2_drop": vc["r2_drop"],
                  "headline": vc["headline"], "caveat": vc["caveat"],
                  "forward_metrics": vc["forward"]["metrics"].round(3)}
    R["segments"] = {"profile": sg["profile"], "silhouette": sg["silhouette"], "choice": sg["choice"]}

    # Parameter recovery - the project's actual finding
    rec = recovery.recovery_report(survey, campaign, customers, fa=fa, vw=vw)
    R["recovery"] = {"coefficients": rec["coefficients"], "factors": rec["factors"],
                     "prices": rec["prices"], "summary": rec["summary"],
                     "n_terms": rec["n_terms"], "n_scored": rec["n_scored"],
                     "clean_share": rec["clean_share"], "false_positives": rec["false_positives"],
                     "headline": recovery.headline(rec)}
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

    # Copilot: the fixed regression suite AND the held-out adversarial suite.
    # The two measure different things and must be reported separately - see
    # copilot/adversarial.py for why the 30/30 is not an accuracy score.
    from .copilot.adversarial import run_adversarial
    from .copilot.evaluate import run_suite
    R["copilot_eval"] = run_suite()
    R["copilot_eval"]["what_this_measures"] = (
        "A fixed regression suite written during development, against the behaviour being "
        "developed. 30/30 means the copilot still does what it did yesterday. It is not an "
        "accuracy score and does not generalise to unseen questions.")
    R["copilot_adversarial"] = run_adversarial()
    R["copilot_adversarial"]["what_this_measures"] = (
        "Ten attacks written after the copilot was finished, with intent to produce a confident "
        "wrong answer. A case passes only if the copilot refuses, asks for clarification or "
        "reports a data error. On first run this suite scored 1/10 and the failures drove the "
        "entity, scope, recommendation and false-premise guards now in planner.py and engine.py.")

    (REPORT_DIR / "results.json").write_text(json.dumps(R, default=_jsonable, indent=1))
    if make_figures:
        from .figures import make_all
        make_all(R, frames, fa, vw, ev, lm, sg)
    return R


if __name__ == "__main__":
    out = run()
    print(json.dumps({"market": out["market"], "copilot": out["copilot_eval"]["summary"]}, default=_jsonable, indent=1))
