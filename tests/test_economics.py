"""Tests for the EBITDA bridge, the returns model and the parameter registry.

The defect these are written against is the one that mattered most in the earlier
version of this project: a per-unit CM3 that silently equalled CM2 because no
acquisition cost was ever passed in, sitting next to a three-year plan with no
fixed overhead anywhere in it.
"""
from __future__ import annotations

import pytest

from neatnest import market, opex
from neatnest.config import (ASSUMED, BENCHMARK, OPEX, PARAMS, SOM_PLAN, SOURCED,
                             TOOLING_AMORT, check_registry, gst_rate, kind_counts,
                             params_frame)
from neatnest.market import Scenario


# --------------------------------------------------------------------------
# Registry integrity
# --------------------------------------------------------------------------
def test_registry_matches_the_values_the_code_uses():
    """The published parameter table must not drift from the code.

    Every Param carries a `live` path to the value it documents, and this asserts
    the two still agree. Without it, someone edits SKUS and the Parameters tab
    quietly starts lying.
    """
    problems = check_registry()
    assert problems == [], "registry drift:\n  " + "\n  ".join(problems)


def test_every_parameter_declares_a_provenance_kind():
    for group, items in PARAMS.items():
        for name, p in items.items():
            assert p.kind in (SOURCED, BENCHMARK, ASSUMED), f"{group}/{name}"
            assert p.source and len(p.source) > 10, f"{group}/{name} has no real source string"
            assert p.drives, f"{group}/{name} does not say what it drives"


def test_sourced_parameters_are_the_minority_and_this_is_disclosed():
    """Not a style check: the project's honesty rests on this being visible.

    If SOURCED ever becomes the majority, the disclosure text on the Parameters
    tab and in the model cards needs rewriting.
    """
    k = kind_counts()
    assert k[ASSUMED] > k[SOURCED], "the model cards claim assumptions outnumber sourced inputs"
    assert len(params_frame()) >= 80


def test_ranges_bracket_their_values():
    for group, items in PARAMS.items():
        for name, p in items.items():
            if p.low is not None and p.high is not None:
                assert p.low <= p.value <= p.high, f"{group}/{name}"
                assert p.low < p.high, f"{group}/{name} has a degenerate range"


# --------------------------------------------------------------------------
# The CM3 defect
# --------------------------------------------------------------------------
def test_cm3_is_not_equal_to_cm2_when_a_customer_is_acquired():
    """The original defect, locked down.

    unit_economics_table() used to be called with no CAC, so every published CM3
    equalled CM2 and acquisition cost vanished from the per-unit view.
    """
    t = market.unit_economics_table("Y1")
    acquired = t[t["cac"] > 0]
    assert len(acquired) >= 6, "core SKUs on D2C and Amazon should all carry acquisition cost"
    assert (acquired["cm3"] < acquired["cm2"]).all()


def test_accessories_and_b2b_carry_no_acquisition_cost():
    """Accessories are sold to customers already paid for - that is why attach matters."""
    t = market.unit_economics_table("Y1").set_index("sku")
    for sku in ("LINR", "CHAR", "WHEL"):
        assert t.loc[sku, "cac"] == 0
        assert t.loc[sku, "cm3"] == pytest.approx(t.loc[sku, "cm2"])
    assert t.loc["PRO", "cac"] == 0


def test_entry_sku_loses_money_on_d2c_acquisition():
    """A real finding the old model hid, pinned so it cannot silently disappear.

    FlexFold Lite on D2C does not cover its own acquisition cost at Year-1 CAC.
    If this test starts failing, either the price ladder or the CAC changed and
    the write-up needs updating.
    """
    t = market.unit_economics_table("Y1")
    lite_d2c = t[(t["sku"] == "LITE") & (t["channel"] == "D2C")].iloc[0]
    assert lite_d2c["cm3"] < 0


def test_waterfall_still_adds_up_with_the_explicit_returns_model():
    ue = market.unit_economics("FLEX", "Amazon", cac=300)
    rebuilt = (ue["net_revenue"] - ue["cogs"] - ue["packaging"] - ue["shipping"]
               - ue["channel_fees"] - ue["returns"] - ue["cac"])
    assert rebuilt == pytest.approx(ue["cm3"])


def test_returns_model_reports_both_old_and_new_figures():
    """The change is published rather than quietly swapped in."""
    ue = market.unit_economics("FLEX", "D2C")
    assert ue["returns"] > 0 and ue["returns_legacy"] > 0
    assert ue["returns_delta"] == pytest.approx(ue["returns"] - ue["returns_legacy"])


def test_returns_cost_rises_with_the_return_rate():
    low = market.unit_economics("FLEX", "D2C", sc=Scenario(returns_mult=0.5))
    high = market.unit_economics("FLEX", "D2C", sc=Scenario(returns_mult=2.0))
    assert high["returns"] > low["returns"]
    assert high["cm2"] < low["cm2"]


# --------------------------------------------------------------------------
# GST
# --------------------------------------------------------------------------
def test_one_gst_helper_on_transaction_value():
    """The economics model and the sales simulator used to disagree on the basis."""
    assert gst_rate(2400) == 0.05
    assert gst_rate(2600) == 0.18
    # A discount can carry a unit across the threshold, which is the whole point
    # of using transaction value rather than MRP.
    assert gst_rate(2600 * (1 - 0.10)) == 0.05


# --------------------------------------------------------------------------
# The bridge
# --------------------------------------------------------------------------
def test_bridge_arithmetic_closes():
    b = opex.ebitda_bridge().set_index("year")
    for y in b.index:
        assert b.loc[y, "ebitda_cr"] == pytest.approx(b.loc[y, "cm3_cr"] - b.loc[y, "fixed_opex_cr"])


def test_fixed_opex_is_present_and_nonzero():
    """The gap that made CM3 look like profit."""
    t = opex.opex_table()
    assert (t["total"] > 0).all()
    for y in OPEX:
        assert opex.fixed_opex(y) == pytest.approx(t.loc[y, "total"])


def test_tooling_amortisation_is_derived_not_asserted():
    assert TOOLING_AMORT == pytest.approx(OPEX["Y1"]["tooling_amortisation"])
    for y in OPEX:
        assert OPEX[y]["tooling_amortisation"] == pytest.approx(TOOLING_AMORT)


def test_year_one_ebitda_is_negative():
    """The honest shape, pinned.

    Year 1 carries tooling and a team against 12,000 units. If this ever turns
    positive, either the plan or the cost schedule changed and every slide that
    says "EBITDA positive from Year 2" is wrong.
    """
    s = opex.summary()
    assert s["ebitda_cr"]["Y1"] < 0
    assert s["first_ebitda_positive_year"] == "Y2"


def test_ebitda_is_below_cm3_in_every_year():
    b = opex.ebitda_bridge().set_index("year")
    assert (b["ebitda_cr"] < b["cm3_cr"]).all()


def test_year_one_is_short_of_break_even():
    be = opex.break_even_table().loc["Y1"]
    assert be["coverage_ratio"] < 1.0
    assert be["break_even_core_units"] > be["planned_core_units"]


def test_break_even_coverage_improves_with_scale():
    be = opex.break_even_table()
    assert list(be["coverage_ratio"]) == sorted(be["coverage_ratio"])


def test_working_capital_is_reported_and_material():
    """It is an exposure, not an expense - but it must not be invisible."""
    b = opex.ebitda_bridge().set_index("year")
    assert (b["working_capital_exposure_cr"] > 0).all()
    # The model cards claim the Y3 exposure exceeds Y2 EBITDA. Keep that true.
    assert b.loc["Y3", "working_capital_exposure_cr"] > b.loc["Y2", "ebitda_cr"]


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------
def test_scenario_multipliers_move_ebitda_in_the_right_direction():
    base = opex.summary()["ebitda_cr"]["Y3"]
    worse = opex.summary(Scenario(cogs_mult=1.3))["ebitda_cr"]["Y3"]
    better = opex.summary(Scenario(cogs_mult=0.8))["ebitda_cr"]["Y3"]
    assert worse < base < better
    assert opex.summary(Scenario(opex_mult=1.4))["ebitda_cr"]["Y3"] < base
    assert opex.summary(Scenario(attach_mult=1.3))["ebitda_cr"]["Y3"] > base


def test_baseline_scenario_is_a_no_op():
    assert opex.summary(Scenario())["ebitda_cr"] == opex.summary()["ebitda_cr"]


def test_tornado_ranks_cogs_first_and_flags_its_provenance():
    """The honest headline of the sensitivity page.

    The biggest lever on Year-3 EBITDA is also one of the least grounded inputs.
    """
    tor = opex.tornado("Y3")
    assert tor.iloc[0]["parameter"] == "cogs"
    assert tor.iloc[0]["kind"] == ASSUMED
    assert (tor["swing_cr"] > 0).all()
    assert list(tor["swing_cr"]) == sorted(tor["swing_cr"], reverse=True)


def test_slider_ranges_come_from_the_registry():
    for name, (group, key) in opex.SLIDER_SOURCES.items():
        lo, hi = opex.slider_range(name)
        p = PARAMS[group][key]
        assert lo == pytest.approx(p.low / p.value, rel=1e-2)
        assert hi == pytest.approx(p.high / p.value, rel=1e-2)
        assert lo < 1.0 < hi or lo <= 1.0 <= hi


def test_breaking_point_for_cogs_is_outside_its_declared_range():
    """Year 3 is robust to any single parameter inside its plausible range."""
    bp = opex.breaking_points("Y3").set_index("parameter")
    assert bp.loc["cogs", "breaks_even_at_mult"] > 1.0
    assert not bp["within_declared_range"].any()


def test_plan_still_grows_and_stays_a_small_share_of_sam():
    p = market.som_plan()
    assert list(p["net_revenue_cr"]) == sorted(p["net_revenue_cr"])
    assert p["share_of_sam_units_pct"].max() < 10


def test_cac_is_charged_per_new_customer_not_per_unit():
    p = market.som_plan().set_index("year")
    for y in SOM_PLAN:
        assert p.loc[y, "new_d2c_customers"] <= p.loc[y, "d2c_core_units"]
    # Repeat share rises over the plan, so the gap must widen.
    gap = [(p.loc[y, "d2c_core_units"] - p.loc[y, "new_d2c_customers"]) / p.loc[y, "d2c_core_units"]
           for y in SOM_PLAN]
    assert gap == sorted(gap)
