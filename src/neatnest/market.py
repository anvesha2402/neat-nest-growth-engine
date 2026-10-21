"""Market sizing, unit economics and the three-year SOM plan.

The per-unit waterfall runs MRP -> net revenue -> CM1 -> CM2 -> CM3. CM3 is
contribution after acquisition cost, and it is NOT profit: fixed operating
costs are subtracted in `opex.ebitda_bridge()`, which is where the number a
reader should care about lives.

Two things changed here relative to the first version of this model, both of
which moved the reported numbers:

  * CAC now flows into CM3. Previously `unit_economics_table()` was called
    without a CAC, so every published CM3 equalled CM2 and the acquisition
    cost silently vanished from the per-unit view.
  * Returns are modelled explicitly instead of as a flat percentage of net
    revenue. A returned unit loses its revenue and costs a reverse leg, but
    recovers its COGS and packaging if it can be resold. The old one-term
    approximation is retained alongside as `returns_legacy` so the difference
    is visible rather than quietly absorbed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (ACCESSORY_MIX, CHANNELS, MARKET, REPEAT_CORE_SHARE,
                     RESTOCK_RATE, REVERSE_SHIP_MULT, SKUS, SOM_PLAN,
                     gst_rate, price_ex_gst)

CR = 1e7  # 1 crore


@dataclass(frozen=True)
class Scenario:
    """Multipliers on the six inputs that actually move the answer.

    Defaults are all 1.0, i.e. the registered parameter values in config.
    The Parameters tab drives this, and `opex.tornado()` sweeps each one
    across the range declared in the parameter registry.
    """
    cogs_mult: float = 1.0
    cac_mult: float = 1.0
    returns_mult: float = 1.0
    attach_mult: float = 1.0
    acos_mult: float = 1.0
    opex_mult: float = 1.0

    def label(self) -> str:
        moved = {k: v for k, v in self.__dict__.items() if v != 1.0}
        return "baseline" if not moved else ", ".join(f"{k.replace('_mult','')} x{v:g}"
                                                      for k, v in moved.items())


BASE = Scenario()


def market_sizing(m: dict = MARKET) -> dict:
    """Top-down + bottom-up TAM, washing-machine-anchored SAM."""
    urban_hh = m["population"] * m["urban_share"] / m["urban_hh_size"]
    rural_hh = m["total_hh"] - urban_hh
    # Solve urban ownership u: urban_hh*u + rural_hh*u/ratio = wm_owner_hh
    u = m["wm_owner_hh"] / (urban_hh + rural_hh / m["urban_rural_wm_ratio"])
    urban_wm_hh = urban_hh * u
    sam_hh = urban_wm_hh * m["reachable_share"]

    annual_spend_per_hh = m["commodity_basket_price"] / m["commodity_basket_life_yrs"]
    tam_bottom_up = urban_hh * m["basket_ownership_urban"] * annual_spend_per_hh

    blended_asp = blended_core_asp_ex_gst("Y1")
    sam_units_per_year = sam_hh / m["flexfold_life_yrs"]
    sam_value = sam_units_per_year * blended_asp
    return {
        "urban_hh_m": urban_hh / 1e6,
        "urban_wm_ownership": u,
        "urban_wm_hh_m": urban_wm_hh / 1e6,
        "sam_hh_m": sam_hh / 1e6,
        "tam_bottom_up_cr": tam_bottom_up / CR,
        "tam_top_down_cr": m["topdown_tam_cr"],
        "sam_units_per_year_m": sam_units_per_year / 1e6,
        "sam_value_cr": sam_value / CR,
        "blended_core_asp_ex_gst": blended_asp,
    }


def blended_core_asp_ex_gst(year: str) -> float:
    mix = SOM_PLAN[year]["mix"]
    return sum(w * price_ex_gst(SKUS[s]["mrp"]) for s, w in mix.items())


def unit_economics(sku: str, channel: str, cac: float = 0.0, discount: float = 0.0,
                   sc: Scenario = BASE) -> dict:
    """Per-unit waterfall from MRP to contribution after acquisition cost (CM3)."""
    s, c = SKUS[sku], CHANNELS[channel]
    cogs = s["cogs"] * sc.cogs_mult
    price = s["mrp"] * (1 - discount)
    net = price / (1 + gst_rate(price))          # GST on transaction value, one shared helper
    fees = price * c["fee_pct"] + c["closing"]

    # Explicit returns: lose the net revenue, pay a reverse leg, recover the
    # goods only on the restockable share. Outbound shipping is already charged
    # on every unit in CM2, so it is not double counted here.
    r_pct = c["returns_pct"] * sc.returns_mult
    recoverable = RESTOCK_RATE * (cogs + s["pack"])
    returns = r_pct * (net + s["ship"] * REVERSE_SHIP_MULT - recoverable)
    returns_legacy = net * r_pct                 # the old one-term approximation

    cm1 = net - cogs - s["pack"]
    cm2 = cm1 - s["ship"] - fees - returns
    cm3 = cm2 - cac
    return {"sku": sku, "channel": channel, "price": price, "gst": price - net, "net_revenue": net,
            "cogs": cogs, "packaging": s["pack"], "cm1": cm1, "cm1_pct": cm1 / net,
            "shipping": s["ship"], "channel_fees": fees, "returns": returns,
            "returns_legacy": returns_legacy, "returns_delta": returns - returns_legacy,
            "cm2": cm2, "cm2_pct": cm2 / net, "cac": cac, "cm3": cm3, "cm3_pct": cm3 / net}


def unit_economics_table(year: str = "Y1", sc: Scenario = BASE) -> pd.DataFrame:
    """Per-unit economics with the year's acquisition cost actually applied.

    D2C core units carry the year's CAC. Amazon core units carry the ACoS
    equivalent, charged on price. B2B and accessories carry no acquisition
    cost: accessories are sold to customers who have already been paid for,
    which is the whole reason the attach rate matters.
    """
    p = SOM_PLAN[year]
    cac_d2c = p["d2c_cac"] * sc.cac_mult
    acos = p["amazon_acos"] * sc.acos_mult
    rows = []
    for sku in ["LITE", "FLEX", "HERI"]:
        rows.append(unit_economics(sku, "D2C", cac=cac_d2c, sc=sc))
        amazon_cac = SKUS[sku]["mrp"] * acos
        rows.append(unit_economics(sku, "Amazon", cac=amazon_cac, sc=sc))
    rows.append(unit_economics("PRO", "B2B", sc=sc))
    for sku in ["LINR", "CHAR", "WHEL"]:
        rows.append(unit_economics(sku, "D2C", cac=0.0, sc=sc))
    df = pd.DataFrame(rows)
    df.insert(2, "year", year)
    df.insert(3, "acquisition_basis", ["D2C CAC" if r.channel == "D2C" and r.cac > 0
                                       else "Amazon ACoS" if r.channel == "Amazon"
                                       else "none (no new customer acquired)"
                                       for r in df.itertuples()])
    return df


def som_plan(sc: Scenario = BASE) -> pd.DataFrame:
    """Revenue and contribution plan by year, down to CM3.

    Acquisition cost is charged per NEW CUSTOMER, not per unit. A core unit
    bought by a household already acquired carries no CAC, and accessories
    carry none at all. With a four-year durable the repeat share is small, so
    this is a modest correction - but charging CAC per unit would overstate
    acquisition spend, and the number is now explicit either way.
    """
    rows = []
    for year, p in SOM_PLAN.items():
        core_units = p["core_units"]
        repeat = REPEAT_CORE_SHARE[year]
        cac_d2c = p["d2c_cac"] * sc.cac_mult
        acos = p["amazon_acos"] * sc.acos_mult
        rev = cm2 = 0.0
        d2c_core_units = amazon_rev = 0.0
        for sku, w in p["mix"].items():
            units = core_units * w
            if sku == "PRO":
                ue = unit_economics(sku, "B2B", sc=sc)
                rev += units * ue["net_revenue"]; cm2 += units * ue["cm2"]
                continue
            consumer_share = p["channel"]["D2C"] + p["channel"]["Amazon"]
            for ch in ["D2C", "Amazon"]:
                u = units * p["channel"][ch] / consumer_share
                ue = unit_economics(sku, ch, sc=sc)
                rev += u * ue["net_revenue"]; cm2 += u * ue["cm2"]
                if ch == "D2C":
                    d2c_core_units += u
                else:
                    amazon_rev += u * ue["price"]
        new_d2c_customers = d2c_core_units * (1 - repeat)
        consumer_units = core_units * (1 - p["mix"]["PRO"])
        acc_units = consumer_units * p["accessory_attach"] * sc.attach_mult
        acc_rev = acc_cm2 = 0.0
        for sku, w in ACCESSORY_MIX.items():
            ue = unit_economics(sku, "D2C", sc=sc)
            acc_rev += acc_units * w * ue["net_revenue"]; acc_cm2 += acc_units * w * ue["cm2"]
        performance = new_d2c_customers * cac_d2c + amazon_rev * acos
        marketing = performance + p["fixed_marketing"]
        total_rev = rev + acc_rev
        total_cm2 = cm2 + acc_cm2
        rows.append({"year": year, "core_units": core_units, "accessory_units": round(acc_units),
                     "d2c_core_units": round(d2c_core_units),
                     "new_d2c_customers": round(new_d2c_customers),
                     "core_revenue_cr": rev / CR, "accessory_revenue_cr": acc_rev / CR,
                     "net_revenue_cr": total_rev / CR, "cm2_cr": total_cm2 / CR,
                     "performance_marketing_cr": performance / CR,
                     "fixed_marketing_cr": p["fixed_marketing"] / CR,
                     "marketing_cr": marketing / CR, "cm3_cr": (total_cm2 - marketing) / CR,
                     "cm3_pct": (total_cm2 - marketing) / total_rev})
    df = pd.DataFrame(rows)
    sizing = market_sizing()
    df["share_of_sam_units_pct"] = df["core_units"] / (sizing["sam_units_per_year_m"] * 1e6) * 100
    return df
