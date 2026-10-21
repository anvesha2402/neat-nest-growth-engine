"""Fixed overhead, the EBITDA bridge, break-even and sensitivity.

WHY THIS MODULE EXISTS

The earlier version of this project reported contribution after marketing (CM3)
as its bottom line. CM3 is not profit. It is what remains after the costs that
scale with a unit sold, and it says nothing about the team, the warehouse, the
moulds or the software that have to be paid for whether or not anything ships.

Reporting a healthy CM3 next to a three-year plan invites the reader to treat it
as earnings, and on this plan that reading is wrong by roughly the cost of the
company. This module closes the gap:

    net revenue
      - COGS, packaging                      -> CM1
      - shipping, channel fees, returns      -> CM2
      - acquisition (CAC + ACoS + brand)     -> CM3
      - fixed operating costs                -> EBITDA

Year 1 comes out negative. That is the expected shape for a D2C launch carrying
tooling and a team against twelve thousand units, and saying so is a stronger
claim than the one it replaces, because a plan that reaches contribution
positivity in Year 1 and EBITDA positivity in Year 2 is an ordinary, defensible
trajectory, whereas a plan that appears profitable from month one is not.

WHAT IS STILL NOT HERE
Depreciation beyond tooling, interest and tax: the bridge stops at EBITDA.
Working capital is reported as an exposure (`working_capital_exposure`) rather
than modelled as a cash flow, which means this is not a cash runway. For a
product that has to fund inventory before it sells any of it, that exposure is
the largest unmodelled risk in the project and is flagged as such in the model
cards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market
from .config import INVENTORY_DAYS, OPEX, PARAMS, SKUS, SOM_PLAN
from .market import BASE, CR, Scenario

OPEX_LINES = ["team", "rent_and_3pl_fixed", "tooling_amortisation", "platform_and_it",
              "support", "insurance_audit_legal"]


# ---------------------------------------------------------------------------
def opex_table(sc: Scenario = BASE) -> pd.DataFrame:
    """Fixed operating cost by line and year, in rupees."""
    rows = []
    for year, lines in OPEX.items():
        row = {"year": year}
        for line in OPEX_LINES:
            row[line] = lines[line] * sc.opex_mult
        row["total"] = sum(row[line] for line in OPEX_LINES)
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def fixed_opex(year: str, sc: Scenario = BASE) -> float:
    return float(sum(OPEX[year][line] for line in OPEX_LINES) * sc.opex_mult)


# ---------------------------------------------------------------------------
def ebitda_bridge(sc: Scenario = BASE) -> pd.DataFrame:
    """The full waterfall by year, in INR crore unless the column says otherwise."""
    plan = market.som_plan(sc)
    rows = []
    for _, p in plan.iterrows():
        year = p["year"]
        opex_cr = fixed_opex(year, sc) / CR
        ebitda_cr = p["cm3_cr"] - opex_cr
        rows.append({
            "year": year,
            "core_units": int(p["core_units"]),
            "accessory_units": int(p["accessory_units"]),
            "new_d2c_customers": int(p["new_d2c_customers"]),
            "net_revenue_cr": p["net_revenue_cr"],
            "cm2_cr": p["cm2_cr"],
            "cm2_pct": p["cm2_cr"] / p["net_revenue_cr"],
            "performance_marketing_cr": p["performance_marketing_cr"],
            "fixed_marketing_cr": p["fixed_marketing_cr"],
            "cm3_cr": p["cm3_cr"],
            "cm3_pct": p["cm3_pct"],
            "fixed_opex_cr": opex_cr,
            "ebitda_cr": ebitda_cr,
            "ebitda_pct": ebitda_cr / p["net_revenue_cr"],
            "ebitda_positive": bool(ebitda_cr > 0),
        })
    df = pd.DataFrame(rows)
    df["cumulative_ebitda_cr"] = df["ebitda_cr"].cumsum()
    df["working_capital_exposure_cr"] = [working_capital_exposure(y, sc) / CR for y in df["year"]]
    return df


def working_capital_exposure(year: str, sc: Scenario = BASE) -> float:
    """Inventory funding requirement: days of cover on the year's goods cost.

    Reported, not modelled. This is the money that has to exist before the
    revenue arrives, and it is not an operating expense - which is exactly why
    an EBITDA-positive year can still be cash-hungry.
    """
    p = SOM_PLAN[year]
    goods = 0.0
    for sku, w in p["mix"].items():
        units = p["core_units"] * w
        goods += units * (SKUS[sku]["cogs"] * sc.cogs_mult + SKUS[sku]["pack"])
    return float(goods * INVENTORY_DAYS / 365)


# ---------------------------------------------------------------------------
def break_even(year: str = "Y3", sc: Scenario = BASE) -> dict:
    """Units needed to cover fixed OpEx, and the contribution per core unit.

    Uses the year's blended contribution after acquisition per core unit,
    including the accessory margin each core unit drags along with it, since
    the attach rate is part of how a unit pays for itself.
    """
    plan = market.som_plan(sc).set_index("year")
    p = plan.loc[year]
    cm3_total = p["cm3_cr"] * CR
    units = p["core_units"]
    cm3_per_core_unit = cm3_total / units
    need = fixed_opex(year, sc)
    return {
        "year": year,
        "cm3_per_core_unit": cm3_per_core_unit,
        "fixed_opex": need,
        "break_even_core_units": need / cm3_per_core_unit if cm3_per_core_unit > 0 else np.inf,
        "planned_core_units": int(units),
        "coverage_ratio": units / (need / cm3_per_core_unit) if cm3_per_core_unit > 0 else 0.0,
        "ebitda_cr": p["cm3_cr"] - need / CR,
    }


def break_even_table(sc: Scenario = BASE) -> pd.DataFrame:
    return pd.DataFrame([break_even(y, sc) for y in SOM_PLAN]).set_index("year")


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
#: Which registry parameters back each slider, used to derive honest slider
#: ranges instead of inventing them at the UI layer.
SLIDER_SOURCES = {
    "cogs": ("Unit costs", "cogs_FLEX"),
    "cac": ("Acquisition and plan", "d2c_cac_Y3"),
    "returns": ("Channel economics", "returns_pct_Amazon"),
    "attach": ("Acquisition and plan", "accessory_attach_Y3"),
    "acos": ("Acquisition and plan", "amazon_acos_Y3"),
    "opex": ("Fixed operating costs", "opex_Y3_team"),
}


def slider_range(name: str) -> tuple[float, float]:
    """Multiplier range for a slider, taken from its parameter's declared range."""
    group, key = SLIDER_SOURCES[name]
    p = PARAMS[group][key]
    if p.low is None or p.high is None or not p.value:
        return (0.7, 1.4)
    return (round(p.low / p.value, 3), round(p.high / p.value, 3))


def tornado(year: str = "Y3", sc: Scenario = BASE) -> pd.DataFrame:
    """Which single parameter moves the year's EBITDA most, over its stated range.

    Each slider is swept to the low and high end of the range declared for its
    backing parameter in the registry, holding the others at the scenario
    value. This is a one-at-a-time sweep, so it shows exposure, not a joint
    probability - the combinations that matter are the ones where two of these
    go wrong together.
    """
    base_ebitda = ebitda_bridge(sc).set_index("year").loc[year, "ebitda_cr"]
    rows = []
    for name in SLIDER_SOURCES:
        lo, hi = slider_range(name)
        field = f"{name}_mult"
        out = {}
        for tag, mult in (("low", lo), ("high", hi)):
            kw = {**sc.__dict__, field: mult}
            out[tag] = ebitda_bridge(Scenario(**kw)).set_index("year").loc[year, "ebitda_cr"]
        group, key = SLIDER_SOURCES[name]
        rows.append({"parameter": name, "backing_param": key, "kind": PARAMS[group][key].kind,
                     "mult_low": lo, "mult_high": hi,
                     "ebitda_at_low_cr": out["low"], "ebitda_at_high_cr": out["high"],
                     "swing_cr": abs(out["high"] - out["low"]),
                     "worst_cr": min(out["low"], out["high"]),
                     "flips_negative": bool(min(out["low"], out["high"]) < 0 <= base_ebitda)})
    df = pd.DataFrame(rows).sort_values("swing_cr", ascending=False).reset_index(drop=True)
    df.insert(0, "base_ebitda_cr", base_ebitda)
    return df


def breaking_point(name: str, year: str = "Y3", sc: Scenario = BASE,
                   lo: float = 0.2, hi: float = 5.0, tol: float = 1e-3) -> float | None:
    """The multiplier on one parameter at which the year's EBITDA hits zero.

    Returns None when EBITDA does not cross zero anywhere in the search range,
    which is itself the useful answer for a parameter the plan is robust to.
    """
    field = f"{name}_mult"

    def f(m: float) -> float:
        return ebitda_bridge(Scenario(**{**sc.__dict__, field: m})).set_index("year").loc[year, "ebitda_cr"]

    f_lo, f_hi = f(lo), f(hi)
    if (f_lo > 0) == (f_hi > 0):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (f_lo > 0):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return round((lo + hi) / 2, 3)


def breaking_points(year: str = "Y3", sc: Scenario = BASE) -> pd.DataFrame:
    """Where each lever has to go to wipe out the year's EBITDA."""
    rows = []
    for name in SLIDER_SOURCES:
        m = breaking_point(name, year, sc)
        group, key = SLIDER_SOURCES[name]
        p = PARAMS[group][key]
        in_range = (m is not None and p.low is not None and p.high is not None
                    and p.value and p.low / p.value <= m <= p.high / p.value)
        rows.append({"parameter": name, "backing_param": key, "kind": p.kind,
                     "breaks_even_at_mult": m,
                     "implied_value": round(p.value * m, 3) if m is not None else None,
                     "within_declared_range": bool(in_range),
                     "declared_low": p.low, "declared_high": p.high})
    return pd.DataFrame(rows)


def summary(sc: Scenario = BASE) -> dict:
    """Headline numbers for the app, the deck and results.json."""
    b = ebitda_bridge(sc).set_index("year")
    first_positive = next((y for y in b.index if b.loc[y, "ebitda_positive"]), None)
    return {
        "scenario": sc.label(),
        "ebitda_cr": {y: round(float(b.loc[y, "ebitda_cr"]), 3) for y in b.index},
        "ebitda_pct": {y: round(float(b.loc[y, "ebitda_pct"]), 4) for y in b.index},
        "cm3_cr": {y: round(float(b.loc[y, "cm3_cr"]), 3) for y in b.index},
        "fixed_opex_cr": {y: round(float(b.loc[y, "fixed_opex_cr"]), 3) for y in b.index},
        "first_ebitda_positive_year": first_positive,
        "cumulative_ebitda_cr": round(float(b["ebitda_cr"].sum()), 3),
        "peak_cumulative_loss_cr": round(float(b["cumulative_ebitda_cr"].min()), 3),
        "working_capital_exposure_cr": {y: round(float(b.loc[y, "working_capital_exposure_cr"]), 3)
                                        for y in b.index},
    }


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(opex_table().round(0).to_string())
    print()
    print(ebitda_bridge().round(3).to_string(index=False))
    print()
    print(break_even_table().round(2).to_string())
    print()
    print(tornado().round(3).to_string(index=False))
    print()
    print(breaking_points().to_string(index=False))
