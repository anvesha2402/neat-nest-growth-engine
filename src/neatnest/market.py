"""Market sizing, unit economics and the three-year SOM plan."""
from __future__ import annotations

import pandas as pd

from .config import (ACCESSORY_MIX, CHANNELS, GST_THRESHOLD, MARKET, SKUS,
                     SOM_PLAN, price_ex_gst)

CR = 1e7  # 1 crore


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


def unit_economics(sku: str, channel: str, cac: float = 0.0, discount: float = 0.0) -> dict:
    """Per-unit waterfall from MRP to contribution after marketing (CM3)."""
    s, c = SKUS[sku], CHANNELS[channel]
    price = s["mrp"] * (1 - discount)
    gst_rate = 0.05 if price <= GST_THRESHOLD else 0.18
    net = price / (1 + gst_rate)
    fees = price * c["fee_pct"] + c["closing"]
    returns = net * c["returns_pct"]
    cm1 = net - s["cogs"] - s["pack"]
    cm2 = cm1 - s["ship"] - fees - returns
    cm3 = cm2 - cac
    return {"sku": sku, "channel": channel, "price": price, "gst": price - net, "net_revenue": net,
            "cogs": s["cogs"], "packaging": s["pack"], "cm1": cm1, "cm1_pct": cm1 / net,
            "shipping": s["ship"], "channel_fees": fees, "returns": returns, "cm2": cm2,
            "cm2_pct": cm2 / net, "cac": cac, "cm3": cm3, "cm3_pct": cm3 / net}


def unit_economics_table() -> pd.DataFrame:
    rows = []
    for sku in ["LITE", "FLEX", "HERI"]:
        for ch in ["D2C", "Amazon"]:
            rows.append(unit_economics(sku, ch))
    rows.append(unit_economics("PRO", "B2B"))
    for sku in ["LINR", "CHAR", "WHEL"]:
        rows.append(unit_economics(sku, "D2C"))
    return pd.DataFrame(rows)


def som_plan() -> pd.DataFrame:
    """Revenue and contribution plan by year."""
    rows = []
    for year, p in SOM_PLAN.items():
        core_units = p["core_units"]
        rev = cm2 = 0.0
        new_d2c_customers = 0.0
        amazon_rev = 0.0
        for sku, w in p["mix"].items():
            units = core_units * w
            if sku == "PRO":
                ue = unit_economics(sku, "B2B")
                rev += units * ue["net_revenue"]; cm2 += units * ue["cm2"]
                continue
            consumer_share = p["channel"]["D2C"] + p["channel"]["Amazon"]
            for ch in ["D2C", "Amazon"]:
                u = units * p["channel"][ch] / consumer_share
                ue = unit_economics(sku, ch)
                rev += u * ue["net_revenue"]; cm2 += u * ue["cm2"]
                if ch == "D2C":
                    new_d2c_customers += u
                else:
                    amazon_rev += u * ue["price"]
        consumer_units = core_units * (1 - p["mix"]["PRO"])
        acc_units = consumer_units * p["accessory_attach"]
        acc_rev = acc_cm2 = 0.0
        for sku, w in ACCESSORY_MIX.items():
            ue = unit_economics(sku, "D2C")
            acc_rev += acc_units * w * ue["net_revenue"]; acc_cm2 += acc_units * w * ue["cm2"]
        marketing = new_d2c_customers * p["d2c_cac"] + amazon_rev * p["amazon_acos"] + p["fixed_marketing"]
        total_rev = rev + acc_rev
        total_cm2 = cm2 + acc_cm2
        rows.append({"year": year, "core_units": core_units, "accessory_units": round(acc_units),
                     "core_revenue_cr": rev / 1e7, "accessory_revenue_cr": acc_rev / 1e7,
                     "net_revenue_cr": total_rev / 1e7, "cm2_cr": total_cm2 / 1e7,
                     "marketing_cr": marketing / 1e7, "cm3_cr": (total_cm2 - marketing) / 1e7,
                     "cm3_pct": (total_cm2 - marketing) / total_rev})
    df = pd.DataFrame(rows)
    sizing = market_sizing()
    df["share_of_sam_units_pct"] = df["core_units"] / (sizing["sam_units_per_year_m"] * 1e6) * 100
    return df
