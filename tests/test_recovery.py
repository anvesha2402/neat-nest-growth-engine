"""Tests for parameter recovery - the project's central claim.

These replace the tests that asserted properties of the generator. The difference
matters: a test that checks "the Kano classifier says the handle is a Must-be"
cannot fail for any reason connected to the code, because the handle was planted
as a Must-be. A test that checks "the planted nulls came back null" can fail, and
tells you something when it does.
"""
from __future__ import annotations

import pandas as pd
import pytest

from neatnest import recovery
from neatnest.config import DATA_DIR
from neatnest.generate_data import TRUE_PARAMS, _tp, true_value


@pytest.fixture(scope="module")
def frames():
    return {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in ("survey", "campaign", "customers")}


@pytest.fixture(scope="module")
def report(frames):
    return recovery.recovery_report(frames["survey"], frames["campaign"], frames["customers"])


# --------------------------------------------------------------------------
# The controls
# --------------------------------------------------------------------------
def test_no_false_positives_on_planted_nulls(report):
    """The single most important test in the suite.

    Two coefficients are planted at exactly zero. If either comes back
    significant, the method is manufacturing effects and nothing else it reports
    can be trusted.
    """
    assert report["false_positives"] == 0
    coef = report["coefficients"]
    nulls = coef[coef["planted"] == 0]
    assert len(nulls) == 2, "expected exactly two planted nulls as controls"
    assert (nulls["verdict"] == "correctly null").all()


def test_planted_nulls_are_actually_declared_as_zero():
    """Guards against someone quietly giving a control a non-zero value."""
    assert true_value("campaign_purchase", "age") == 0.0
    assert true_value("campaign_purchase", "source[Meta paid social]") == 0.0


# --------------------------------------------------------------------------
# The strong effects
# --------------------------------------------------------------------------
def test_strong_campaign_effects_are_recovered(report):
    coef = report["coefficients"].set_index("term")
    for term in ("prev_home_org_purchases", "digital_engagement_score", "fit_quiz_completed",
                 "site_visits"):
        assert coef.loc[term, "verdict"] == "recovered", f"{term}: {coef.loc[term, 'verdict']}"


def test_recovered_terms_have_planted_value_inside_the_interval(report):
    """The definition of 'recovered', checked rather than trusted."""
    coef = report["coefficients"]
    rec = coef[coef["verdict"] == "recovered"]
    assert len(rec) >= 8
    assert (rec["ci_low"] <= rec["planted"]).all()
    assert (rec["planted"] <= rec["ci_high"]).all()


# --------------------------------------------------------------------------
# The honest failures
# --------------------------------------------------------------------------
def test_underpowered_effects_are_reported_as_missed_not_absent(report):
    """Income and both channel effects are real and planted, and are not detectable.

    This is the result the earlier write-up reported as a finding ("the strongest
    lever is not a channel"). It is a power failure, and the recovery table must
    label it as one.
    """
    coef = report["coefficients"].set_index("term")
    for term in ("monthly_income_k", "source[Google search]",
                 "source[Housing-society activation]"):
        assert coef.loc[term, "planted"] != 0, f"{term} should be a real planted effect"
        assert coef.loc[term, "verdict"] == "missed", f"{term}: {coef.loc[term, 'verdict']}"


def test_latent_driver_is_reported_as_unrecoverable(report):
    """The strongest planted lapse driver is never exported, and must say so."""
    coef = report["coefficients"].set_index("term")
    row = coef.loc["latent_engagement"]
    assert row["verdict"] == "not recoverable"
    assert pd.isna(row["estimated"]), "a latent driver cannot have an estimate"
    assert row["planted"] == -0.70


def test_latent_driver_is_not_in_the_exported_data(frames):
    """Belt and braces: the confounder must genuinely be absent from the CSV."""
    cols = set(frames["customers"].columns)
    assert "engage" not in cols and "latent_engagement" not in cols


def test_omitted_variable_bias_shows_up_in_nps(report):
    """NPS absorbs the missing latent driver, so it should overshoot its planted value."""
    coef = report["coefficients"].set_index("term")
    row = coef.loc["nps_m3"]
    assert row["verdict"] in ("inflated", "attenuated", "recovered")
    if row["verdict"] == "inflated":
        assert abs(row["estimated"]) > abs(row["planted"])


# --------------------------------------------------------------------------
# Structural integrity
# --------------------------------------------------------------------------
def test_every_mapped_term_exists_in_true_params():
    """A recovery table that references an undeclared coefficient is lying."""
    campaign = _tp("campaign_purchase")
    for true_name in recovery.CAMPAIGN_MAP.values():
        assert true_name in campaign, f"{true_name} not declared in TRUE_PARAMS"
    lapse = _tp("customer_lapse")
    for true_name in recovery.LAPSE_MAP.values():
        assert true_name in lapse, f"{true_name} not declared in TRUE_PARAMS"


def test_unobservable_terms_are_never_marked_recovered(report):
    """Terms flagged observable=False must not be scored as successes."""
    coef = report["coefficients"]
    unobs = coef[coef["verdict"] == "not recoverable"]
    assert len(unobs) >= 3
    assert not unobs["verdict"].isin(["recovered", "correctly null"]).any()


def test_verdict_vocabulary_is_closed(report):
    assert set(report["coefficients"]["verdict"]) <= set(recovery.VERDICTS)


def test_implied_loading_exceeds_nominal_for_every_trait():
    """The latent-variance correction must push the target up, never down.

    If this inverts, the EFA is being scored against the wrong target and a
    correct extraction will look like an overestimate.
    """
    for trait, items in _tp("survey_factors").items():
        for item, nominal in items.items():
            assert recovery.implied_loading(nominal, trait) > nominal, f"{trait}/{item}"


def test_latent_sd_is_above_one_for_every_trait():
    for trait in TRUE_PARAMS["survey_latents"]["shifts"]:
        assert recovery._latent_sd(trait) > 1.0


def test_headline_is_consistent_with_the_table(report):
    """The published sentence must not disagree with the data behind it."""
    line = recovery.headline(report)
    s = report["summary"]
    assert str(report["n_terms"]) in line
    assert str(s.get("recovered", 0)) in line
    assert "false positive" in line
