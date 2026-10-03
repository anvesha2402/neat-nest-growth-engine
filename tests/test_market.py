from neatnest.config import GST_THRESHOLD, SKUS, price_ex_gst
from neatnest.market import market_sizing, som_plan, unit_economics


def test_all_skus_under_gst_threshold():
    assert all(s["mrp"] <= GST_THRESHOLD for s in SKUS.values())


def test_price_ex_gst_switches_rate_above_threshold():
    assert round(price_ex_gst(2100), 2) == round(2100 / 1.05, 2)
    assert round(price_ex_gst(3000), 2) == round(3000 / 1.18, 2)


def test_sam_chain_is_consistent():
    m = market_sizing()
    assert 120 < m["urban_hh_m"] < 135
    assert m["urban_wm_hh_m"] < m["urban_hh_m"]
    assert abs(m["sam_hh_m"] - 0.2 * m["urban_wm_hh_m"]) < 1e-9
    # national ownership must reconcile to the published 60 M owners
    rural = 335 - m["urban_hh_m"]
    owners = m["urban_hh_m"] * m["urban_wm_ownership"] + rural * m["urban_wm_ownership"] / 4
    assert abs(owners - 60) < 1e-6


def test_unit_economics_waterfall_adds_up():
    ue = unit_economics("FLEX", "Amazon", cac=300)
    rebuilt = ue["net_revenue"] - ue["cogs"] - ue["packaging"] - ue["shipping"] - ue["channel_fees"] - ue["returns"] - ue["cac"]
    assert abs(rebuilt - ue["cm3"]) < 1e-9
    assert ue["channel_fees"] == 1799 * 0.13 + 56


def test_hero_is_profitable_before_acquisition_on_every_channel():
    for ch in ("D2C", "Amazon"):
        assert unit_economics("FLEX", ch)["cm2"] > 0


def test_som_plan_grows_and_stays_small_share():
    p = som_plan()
    assert list(p["net_revenue_cr"]) == sorted(p["net_revenue_cr"])
    assert p["share_of_sam_units_pct"].max() < 10
