import json

import pandas as pd

from neatnest.config import REPORT_DIR


def results():
    return json.loads((REPORT_DIR / "results.json").read_text())


def test_propensity_beats_chance_out_of_sample():
    P = results()["propensity"]
    assert P["cv_auc_mean"] > 0.7 and P["holdout"]["auc"] > 0.7


def test_quiz_and_prior_purchases_significant_income_not():
    c = pd.DataFrame(results()["propensity"]["coefficients"]).T
    assert c.loc["fit_quiz_completed", "significant"] and c.loc["prev_home_org_purchases", "significant"]
    assert not c.loc["monthly_income_k", "significant"]


def test_lapse_model_and_top_deciles():
    L = results()["lapse"]
    m = pd.DataFrame(L["metrics"]).T
    assert m["auc"].max() > 0.75
    lift = pd.DataFrame(L["lift"]).T
    assert lift["cum_capture"].iloc[2] > 0.5


def test_segments_have_named_extremes():
    prof = pd.DataFrame(results()["segments"]["profile"]).T
    assert {"Nest Advocates", "Service-scarred"} <= set(prof.index)
    assert prof["lapse_rate"].idxmax() == "Service-scarred"


def test_simulated_year_one_close_to_plan():
    R = results()
    assert abs(R["sales"]["core_units"] - 12000) / 12000 < 0.15
