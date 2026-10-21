"""Tests for target leakage and out-of-fold scoring.

Both defects shared a shape: a number that looked like predictive skill but was
partly an artefact of how it was computed. These tests pin the honest versions
and, where the dishonest version is retained for comparison, pin the gap too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neatnest import retention
from neatnest.config import DATA_DIR


@pytest.fixture(scope="module")
def customers():
    return pd.read_csv(DATA_DIR / "customers.csv")


@pytest.fixture(scope="module")
def comparison(customers):
    return retention.value_model_comparison(customers)


# --------------------------------------------------------------------------
# Target leakage
# --------------------------------------------------------------------------
def test_forward_target_exists_and_is_not_the_leaky_one(customers):
    assert retention.VALUE_FWD in customers.columns
    assert retention.VALUE in customers.columns
    assert not customers[retention.VALUE].equals(customers[retention.VALUE_FWD])


def test_leaky_target_contains_its_own_predictors(customers):
    """The mechanism of the leak, demonstrated rather than asserted.

    value_12m is built from the first-order price, accessory spend in months 1-6
    and referrals in months 1-6. Reconstructing it from those three features plus
    the forward component should land almost exactly on the target.
    """
    from neatnest.config import SKUS, price_ex_gst
    core = customers["first_sku"].map(lambda s: price_ex_gst(SKUS[s]["mrp"]))
    rebuilt = (core + customers["accessory_spend_m1_6"] / 1.05
               + customers[retention.VALUE_FWD]
               + customers["referrals_m1_6"] * 0.6 * 1600)
    # Rounding in the generator means this is near-exact, not exact.
    assert np.corrcoef(rebuilt, customers[retention.VALUE])[0, 1] > 0.999


def test_removing_leakage_collapses_r2(comparison):
    """The headline demonstration."""
    leaky = comparison["leaky"]["metrics"].loc["OLS", "r2"]
    fwd = comparison["forward"]["metrics"].loc["OLS", "r2"]
    assert leaky > 0.85, "the leaky target should look spectacular - that is the point"
    assert fwd < 0.35, "the honest target should be modest"
    assert comparison["r2_drop"] > 0.5


def test_forward_target_shares_no_column_with_its_predictors(customers):
    """No feature used to predict forward value may appear inside it.

    value_fwd_7_12 is accessory spend from months 7-12. All features are month-6
    snapshots, so none of them can be a component of it. Checked by confirming no
    feature is perfectly collinear with the target.
    """
    feats = [c for c in retention.NUM if c in customers.columns]
    for f in feats:
        r = abs(np.corrcoef(customers[f], customers[retention.VALUE_FWD])[0, 1])
        assert r < 0.9, f"{f} correlates {r:.3f} with the forward target - possible leak"


def test_comparison_publishes_both_numbers(comparison):
    comp = comparison["comparison"]
    assert len(comp) == 2
    assert set(comp["target"]) == {retention.VALUE, retention.VALUE_FWD}
    assert comp["contains_model_features"].tolist() == [True, False]
    assert "leakage" in comparison["headline"].lower() or "identity" in comparison["headline"].lower()


def test_zero_inflation_caveat_is_warranted(customers, comparison):
    """The caveat claims most customers have zero forward value. Keep it true."""
    zero = (customers[retention.VALUE_FWD] == 0).mean()
    assert zero > 0.4
    assert "zero forward value" in comparison["caveat"]


# --------------------------------------------------------------------------
# Out-of-fold scoring
# --------------------------------------------------------------------------
def test_oof_scores_are_proper_probabilities(customers):
    p = retention.oof_lapse_scores(customers)
    assert len(p) == len(customers)
    assert ((p >= 0) & (p <= 1)).all()
    assert p.std() > 0.01, "scores should discriminate, not be constant"


def test_oof_scores_still_rank_better_than_chance(customers):
    from sklearn.metrics import roc_auc_score
    p = retention.oof_lapse_scores(customers)
    assert roc_auc_score(customers[retention.TARGET], p) > 0.72


def test_in_sample_scoring_is_optimistic_but_the_gap_is_published(customers):
    """The honest version of this fix.

    In-sample scoring inflates apparent reach. At this sample size the inflation
    is tiny, and the project publishes the small number rather than implying the
    correction was dramatic. This test pins the direction, not a magnitude.
    """
    from sklearn.pipeline import Pipeline
    oof = retention.oof_lapse_scores(customers)
    lm = retention.lapse_models(customers)
    ins = lm["model"].predict_proba(customers[retention.NUM + retention.CAT])[:, 1]
    s_oof = retention.retention_scenario(customers, oof)
    s_ins = retention.retention_scenario(customers, ins, scores_are_oof=False)
    assert s_ins["share_of_lapsers_reached"] >= s_oof["share_of_lapsers_reached"]
    assert s_oof["scores_are_oof"] is True
    assert s_ins["scores_are_oof"] is False


def test_scenario_declares_its_assumptions(customers):
    """Two of three inputs are assumptions and the output must say so."""
    p = retention.oof_lapse_scores(customers)
    s = retention.retention_scenario(customers, p)
    assert "save_rate" in s["assumptions"]
    assert "ASSUMED" in s["assumptions"]["basis"]
    assert s["lift_vs_random"] > 1.0


def test_lapse_model_warns_against_scoring_training_rows(customers):
    lm = retention.lapse_models(customers)
    assert "oof_lapse_scores" in lm["note"]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def test_calibration_is_measured_for_every_model(customers):
    lm = retention.lapse_models(customers)
    m = lm["metrics"]
    assert "brier" in m.columns and "brier_calibrated" in m.columns
    assert (m["brier"] > 0).all() and (m["brier"] < 0.25).all()


def test_recalibration_improves_the_brier_score(customers):
    """Class weighting distorts the probability scale; this quantifies it.

    If recalibration stopped helping, the claim that these scores are for ranking
    rather than probability would need revisiting.
    """
    lm = retention.lapse_models(customers)
    m = lm["metrics"]
    assert (m["brier_calibrated"] < m["brier"]).all()


def test_reliability_curve_is_published(customers):
    lm = retention.lapse_models(customers)
    rel = lm["reliability"]
    assert len(rel["mean_predicted"]) == len(rel["fraction_positive"]) >= 5


# --------------------------------------------------------------------------
# Segment naming
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_segment_names_are_unique_for_any_k(customers, k):
    """The old naming rule compared floats with == and could duplicate a label."""
    sg = retention.segments(customers, k=k)
    names = list(sg["profile"].index)
    assert len(names) == k
    assert len(set(names)) == k, f"duplicate segment names at k={k}: {names}"


def test_k_choice_is_disclosed_as_a_choice(customers):
    sg = retention.segments(customers)
    ch = sg["choice"]
    assert ch["k_used"] == 4
    assert ch["k_preferred_by_silhouette"] == 2
    assert ch["silhouette_cost_of_choice"] > 0
    assert str(ch["k_preferred_by_silhouette"]) in ch["honest_summary"]


def test_silhouette_curve_spans_enough_k_to_show_the_tradeoff(customers):
    sg = retention.segments(customers)
    assert set(sg["silhouette"]) >= {2, 3, 4, 5, 6}
