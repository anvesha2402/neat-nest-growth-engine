"""Single source of truth for Neat Nest business assumptions.

Every number used by the market model, the data simulator, the Streamlit app
and the project brief is defined here once.

HOW TO READ THE NUMBERS IN THIS FILE
Each input is also registered in PARAMS below as a `Param` record carrying its
unit, a provenance KIND, a source string, a plausible range and what it drives.
The kinds are deliberately blunt:

  SOURCED    traceable to a named published source. The source string says which.
  BENCHMARK  a parameterised benchmark input taken from comparable products,
             categories or published rate cards. Defensible as a starting value,
             not a measurement of this business.
  ASSUMED    no external basis. A number chosen so the model can run, and the
             first thing to replace with a vendor quote or primary research.

Nothing in this file is a measurement of a real company, because there is no
real company. The point of the registry is that a reader can see which numbers
are anchored, which are borrowed, and which are invented - and can move any of
them in the Parameters tab of the app to see what the decision engine does.

`check_registry()` asserts that every registered value still equals the value
the code actually uses, so the two cannot silently drift apart. It runs in the
test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
KNOWLEDGE_DIR = ROOT / "knowledge"
ASSET_DIR = ROOT / "assets"

SEED = 2027  # one seed for the whole simulated Neat Nest universe

# --------------------------------------------------------------------------
# Brand palette (taken from the India Craft Heritage packaging system)
# --------------------------------------------------------------------------
PALETTE = {
    "indigo": "#2F415D",
    "terracotta": "#C47A4F",
    "ivory": "#F4EBDD",
    "sand": "#D5B893",
    "olive": "#7A7748",
    "charcoal": "#2F2A26",
    "rust": "#A94F39",
}

# --------------------------------------------------------------------------
# Sources referenced by tag
# --------------------------------------------------------------------------
SOURCES = {
    "SRC_GST": "Tally Solutions, GST rates for textiles after the 56th GST Council (effective 22 Sep 2025): "
               "chapter 63 made-ups 5% up to Rs 2,500 per piece, 18% above",
    "SRC_AMZ": "Shiprocket, Amazon India commission rates: Home Storage referral 13% above Rs 300; closing fee Rs 56 above Rs 1,000 (Easy Ship)",
    "SRC_HH": "Data for India, household assets: ~335 million households in 2024; ~60 million own a washing machine; "
              "urban households 4x as likely to own one as rural",
    "SRC_URB": "SBI Research, Precursor to Census 2024: population 138-142 crore, urbanisation 35-37%",
    "SRC_TUS": "MoSPI Time Use Survey 2024 (via Business Today): women 289 min/day vs men 88 min/day in unpaid domestic services; "
               "participation 81.5% vs 27.1%",
    "SRC_IKEA": "IKEA India laundry baskets category: Rs 199-2,990, no wheeled product",
    "SRC_PEP": "Pepperfry laundry basket category: Rs 449-2,799 selling price, typically 40-75% off MRP",
    "SRC_LIVZ": "Flipkart, Livzing metal rolling laundry cart: Rs 3,644 selling (MRP Rs 5,999), 9 kg",
    "SRC_JJ": "Joseph Joseph Tota 90L: US$115, two compartments, no wheels",
    "SRC_IMARC": "IMARC, India home storage market: US$1.6 bn (2025) to US$2.3 bn (2034), 4.85% CAGR",
    "SRC_TOPDOWN": "Original PBM research: India laundry-storage market derived from global US$1.2-2.9 bn, midpoint ~Rs 1,690 crore",
    "SRC_INTERVIEWS": "Neat Nest primary interviews (4 respondents, Round 2 price anchors): Rs 2,000; Rs 1,500-2,000; under Rs 1,000; Rs 1,500",
}

# --------------------------------------------------------------------------
# Product ladder (all prices MRP, GST-inclusive, in INR)
# --------------------------------------------------------------------------
GST_THRESHOLD = 2500          # SRC_GST
GST_LOW, GST_HIGH = 0.05, 0.18

SKUS = {
    "LITE": {"name": "FlexFold Lite", "family": "Core", "mrp": 899, "cogs": 290, "pack": 30,
             "ship": 80, "desc": "Fold-flat, 2 compartments, no wheels. Hostel/PG entry SKU."},
    "FLEX": {"name": "FlexFold", "family": "Core", "mrp": 1799, "cogs": 520, "pack": 45,
             "ship": 110, "desc": "Hero: tilt-and-roll wheels, 3 removable compartments, ventilated panels, "
                                 "antimicrobial liner, folds to 8 cm."},
    "HERI": {"name": "FlexFold Heritage Edition", "family": "Core", "mrp": 2299, "cogs": 545, "pack": 75,
             "ship": 110, "desc": "Hero product + regional craft sleeve and printed liner panel."},
    "PRO":  {"name": "FlexFold Pro (B2B)", "family": "Core", "mrp": 1399, "cogs": 520, "pack": 20,
             "ship": 45, "desc": "Bulk SKU for co-living operators and hostels, 50+ units."},
    "LINR": {"name": "Liner Set", "family": "Accessories", "mrp": 399, "cogs": 95, "pack": 10,
             "ship": 50, "desc": "Replacement antimicrobial liner set (3 compartments)."},
    "CHAR": {"name": "Charcoal Odour Insert (3-pack)", "family": "Accessories", "mrp": 299, "cogs": 70,
             "pack": 8, "ship": 45, "desc": "Replaceable odour-control inserts."},
    "WHEL": {"name": "Wheel Kit", "family": "Accessories", "mrp": 249, "cogs": 60, "pack": 8,
             "ship": 45, "desc": "Replacement wheels and axle; the repairability promise."},
}

# Channel economics (fractions of GST-inclusive price unless stated)
CHANNELS = {
    "D2C":    {"fee_pct": 0.02, "closing": 0, "returns_pct": 0.04},     # payment gateway ~2% (EST)
    "Amazon": {"fee_pct": 0.13, "closing": 56, "returns_pct": 0.06},    # SRC_AMZ
    "B2B":    {"fee_pct": 0.00, "closing": 0, "returns_pct": 0.01},
}

# --------------------------------------------------------------------------
# Market sizing inputs
# --------------------------------------------------------------------------
MARKET = {
    "population": 1.40e9,              # SRC_URB midpoint
    "urban_share": 0.36,               # SRC_URB midpoint of 35-37%
    "urban_hh_size": 4.0,              # EST (urban households are smaller than the 4.4 national mean)
    "total_hh": 335e6,                 # SRC_HH
    "wm_owner_hh": 60e6,               # SRC_HH
    "urban_rural_wm_ratio": 4.0,       # SRC_HH
    "basket_ownership_urban": 0.70,    # EST: share of urban households using a dedicated basket/hamper
    "commodity_basket_price": 400,     # EST: typical discounted selling price (SRC_PEP, SRC_IKEA)
    "commodity_basket_life_yrs": 1.5,  # EST from interviews ("replaced it twice this year", "joints crack after a year")
    "reachable_share": 0.20,           # EST: online, premium-capable share of urban machine-owning households
    "flexfold_life_yrs": 4.0,          # EST: design life backed by warranty
    "topdown_tam_cr": 1690,            # SRC_TOPDOWN
}

# Phased plan (units of core SKUs; accessories via attach rate)
SOM_PLAN = {
    "Y1": {"core_units": 12000, "mix": {"LITE": 0.18, "FLEX": 0.58, "HERI": 0.14, "PRO": 0.10},
           "channel": {"D2C": 0.45, "Amazon": 0.45, "B2B": 0.10}, "accessory_attach": 0.55,
           "d2c_cac": 520, "amazon_acos": 0.14, "fixed_marketing": 1.2e6},
    "Y2": {"core_units": 40000, "mix": {"LITE": 0.20, "FLEX": 0.52, "HERI": 0.18, "PRO": 0.10},
           "channel": {"D2C": 0.40, "Amazon": 0.50, "B2B": 0.10}, "accessory_attach": 0.70,
           "d2c_cac": 460, "amazon_acos": 0.12, "fixed_marketing": 3.0e6},
    "Y3": {"core_units": 100000, "mix": {"LITE": 0.20, "FLEX": 0.50, "HERI": 0.20, "PRO": 0.10},
           "channel": {"D2C": 0.38, "Amazon": 0.52, "B2B": 0.10}, "accessory_attach": 0.80,
           "d2c_cac": 420, "amazon_acos": 0.11, "fixed_marketing": 6.0e6},
}
ACCESSORY_MIX = {"LINR": 0.50, "CHAR": 0.35, "WHEL": 0.15}

PHASES = [
    ("Phase 0 - Validate", "Months -3 to 0 (Jan-Mar 2027)",
     "Van Westendorp + Kano survey (n=600), 50-unit beta with Mumbai/Pune housing societies, vendor quotes."),
    ("Phase 1 - Launch", "Months 1-6 (Apr-Sep 2027)",
     "D2C + Amazon in Mumbai and Pune; 2 heritage sleeves (Block Print, Ikat); one co-living B2B pilot."),
    ("Phase 2 - Expand", "Months 7-18 (Oct 2027-Sep 2028)",
     "Bengaluru, Hyderabad, Delhi NCR; Flipkart; all 5 regional sleeves; accessories on quick commerce."),
    ("Phase 3 - Scale", "Year 2-3",
     "Selective Home Centre shop-in-shop; institutional contracts; Year-2 roadmap features."),
]

CITIES = ["Mumbai", "Pune", "Bengaluru", "Hyderabad", "Delhi NCR"]
LAUNCH_CITIES = ["Mumbai", "Pune"]
CITY_SLEEVE = {  # which regional sleeve leads in which city
    "Mumbai": "Block Print", "Pune": "Ikat", "Bengaluru": "Ikat",
    "Hyderabad": "Kalamkari", "Delhi NCR": "Madhubani",
}


# --------------------------------------------------------------------------
# Fixed operating cost schedule (INR per year)
# --------------------------------------------------------------------------
# These are the costs that do not scale with a unit sold, and their absence was
# the single largest gap in the earlier version of this model: it reported
# contribution after marketing (CM3) and left the reader to assume that was
# profit. It is not. CM3 minus this schedule is EBITDA, and in Year 1 that
# number is negative - which is the normal shape for a D2C launch and a more
# credible claim than the one it replaces.
#
# Tooling is held as a capital figure with a life, so the annual charge is
# derived rather than asserted.
TOOLING_CAPEX = 4_000_000      # injection moulds, wheel/axle jig, pack dies
TOOLING_LIFE_YRS = 4
TOOLING_AMORT = TOOLING_CAPEX / TOOLING_LIFE_YRS

OPEX = {
    "Y1": {"team": 4_500_000, "rent_and_3pl_fixed": 900_000, "tooling_amortisation": TOOLING_AMORT,
           "platform_and_it": 350_000, "support": 250_000, "insurance_audit_legal": 400_000},
    "Y2": {"team": 7_500_000, "rent_and_3pl_fixed": 1_500_000, "tooling_amortisation": TOOLING_AMORT,
           "platform_and_it": 500_000, "support": 500_000, "insurance_audit_legal": 500_000},
    "Y3": {"team": 13_000_000, "rent_and_3pl_fixed": 2_800_000, "tooling_amortisation": TOOLING_AMORT,
           "platform_and_it": 800_000, "support": 1_000_000, "insurance_audit_legal": 700_000},
}

# Returns are modelled explicitly rather than as a flat percentage of revenue.
# A returned unit loses its net revenue, costs a reverse leg, and recovers its
# COGS and packaging only if the unit can be put back on the shelf. For an
# opened, used laundry hamper that is often not the case, so RESTOCK_RATE is
# low and is one of the six sliders in the app.
RESTOCK_RATE = 0.40           # share of returned units resold as new
REVERSE_SHIP_MULT = 1.0       # reverse leg as a multiple of outbound shipping

# Share of core units bought by customers already acquired, who carry no new CAC.
REPEAT_CORE_SHARE = {"Y1": 0.02, "Y2": 0.06, "Y3": 0.10}

# Working capital is reported as an exposure, not modelled as a cash flow.
INVENTORY_DAYS = 75


def gst_rate(transaction_value: float) -> float:
    """GST applies to transaction value, so a discount can cross the threshold.

    One helper, used by the economics model and the sales simulator alike, so
    the two cannot disagree. Note that every SKU in the current ladder sits
    below the threshold, which makes the 18% branch inactive today - the
    Heritage Edition at Rs 2,299 is the one that would cross it on a price rise.
    """
    return GST_LOW if transaction_value <= GST_THRESHOLD else GST_HIGH


def price_ex_gst(transaction_value: float) -> float:
    """Strip GST from a transaction value (MRP when there is no discount)."""
    return transaction_value / (1 + gst_rate(transaction_value))

# Chart palette: brand-adjacent hues validated for colour-vision deficiency
# (adjacent CVD dE >= 9, normal-vision dE >= 21 on a light surface).
SERIES = ["#3a6fb0", "#d0673a", "#1f9e78", "#d9a020", "#b5487a"]
INK, INK_MUTED, GRID = "#2F2A26", "#6b665f", "#e6e1d8"


# ==========================================================================
# PARAMETER REGISTRY
# ==========================================================================
SOURCED, BENCHMARK, ASSUMED = "SOURCED", "BENCHMARK", "ASSUMED"


@dataclass(frozen=True)
class Param:
    """One declared input to the decision engine."""
    value: float
    unit: str
    kind: str                    # SOURCED | BENCHMARK | ASSUMED
    source: str
    drives: str
    low: float | None = None     # plausible lower bound for sensitivity testing
    high: float | None = None    # plausible upper bound
    live: str | None = None      # dotted path to the value the code actually uses

    @property
    def spread(self) -> float | None:
        if self.low is None or self.high is None or not self.value:
            return None
        return (self.high - self.low) / self.value


def _p(value, unit, kind, source, drives, low=None, high=None, live=None) -> Param:
    return Param(value, unit, kind, source, drives, low, high, live)


# Grouped for display. `live` names where the number is consumed, and
# check_registry() verifies the two still agree.
PARAMS: dict[str, dict[str, Param]] = {
    "Price ladder": {
        f"mrp_{k}": _p(v["mrp"], "INR", BENCHMARK,
                       "Positioned against SRC_IKEA / SRC_PEP / SRC_LIVZ / SRC_JJ category ranges "
                       "and the SRC_INTERVIEWS anchors (n=4)",
                       "net revenue, GST band, every contribution margin",
                       low=round(v["mrp"] * 0.85), high=round(v["mrp"] * 1.15),
                       live=f"SKUS['{k}']['mrp']")
        for k, v in SKUS.items()
    },
    "Unit costs": {
        **{f"cogs_{k}": _p(v["cogs"], "INR/unit", ASSUMED,
                           "No vendor quote. Built up from material and moulding estimates - the first "
                           "number to replace with a real quotation",
                           "CM1, break-even, the COGS slider",
                           low=round(v["cogs"] * 0.80), high=round(v["cogs"] * 1.30),
                           live=f"SKUS['{k}']['cogs']")
           for k, v in SKUS.items()},
        **{f"pack_{k}": _p(v["pack"], "INR/unit", ASSUMED, "Estimated carton and insert cost",
                           "CM1", low=round(v["pack"] * 0.8), high=round(v["pack"] * 1.4),
                           live=f"SKUS['{k}']['pack']")
           for k, v in SKUS.items()},
        **{f"ship_{k}": _p(v["ship"], "INR/unit", BENCHMARK,
                           "Typical 3PL surface rate for the volumetric weight of a folded unit",
                           "CM2, returns cost", low=round(v["ship"] * 0.8), high=round(v["ship"] * 1.5),
                           live=f"SKUS['{k}']['ship']")
           for k, v in SKUS.items()},
    },
    "Channel economics": {
        "fee_pct_D2C": _p(CHANNELS["D2C"]["fee_pct"], "share of price", BENCHMARK,
                          "Payment gateway rate, typical Indian PG pricing",
                          "CM2 on D2C", 0.015, 0.025, "CHANNELS['D2C']['fee_pct']"),
        "fee_pct_Amazon": _p(CHANNELS["Amazon"]["fee_pct"], "share of price", SOURCED,
                             SOURCES["SRC_AMZ"], "CM2 on Amazon", 0.11, 0.15,
                             "CHANNELS['Amazon']['fee_pct']"),
        "closing_Amazon": _p(CHANNELS["Amazon"]["closing"], "INR/order", SOURCED,
                             SOURCES["SRC_AMZ"], "CM2 on Amazon", 40, 70,
                             "CHANNELS['Amazon']['closing']"),
        "returns_pct_D2C": _p(CHANNELS["D2C"]["returns_pct"], "share of units", ASSUMED,
                              "No observed return rate exists for a product that has never shipped",
                              "returns cost, the returns slider", 0.02, 0.08,
                              "CHANNELS['D2C']['returns_pct']"),
        "returns_pct_Amazon": _p(CHANNELS["Amazon"]["returns_pct"], "share of units", BENCHMARK,
                                 "Marketplace home-goods returns run materially above own-site rates",
                                 "returns cost, the returns slider", 0.04, 0.12,
                                 "CHANNELS['Amazon']['returns_pct']"),
        "restock_rate": _p(RESTOCK_RATE, "share of returned units", ASSUMED,
                           "An opened, used hamper is often not resellable as new. Wide range on purpose",
                           "returns cost - the single most uncertain input in the returns model",
                           0.20, 0.80, "RESTOCK_RATE"),
        "reverse_ship_mult": _p(REVERSE_SHIP_MULT, "x outbound ship", ASSUMED,
                                "Reverse logistics assumed to cost about one outbound leg",
                                "returns cost", 0.8, 1.5, "REVERSE_SHIP_MULT"),
    },
    "GST": {
        "gst_threshold": _p(GST_THRESHOLD, "INR/piece", SOURCED, SOURCES["SRC_GST"],
                            "which GST rate applies to a transaction", live="GST_THRESHOLD"),
        "gst_low": _p(GST_LOW, "rate", SOURCED, SOURCES["SRC_GST"], "net revenue", live="GST_LOW"),
        "gst_high": _p(GST_HIGH, "rate", SOURCED, SOURCES["SRC_GST"],
                       "net revenue above the threshold (inactive at current prices)", live="GST_HIGH"),
    },
    "Market sizing": {
        "population": _p(MARKET["population"], "people", SOURCED, SOURCES["SRC_URB"],
                         "urban households, TAM", 1.38e9, 1.42e9, "MARKET['population']"),
        "urban_share": _p(MARKET["urban_share"], "share", SOURCED, SOURCES["SRC_URB"],
                          "urban households", 0.35, 0.37, "MARKET['urban_share']"),
        "urban_hh_size": _p(MARKET["urban_hh_size"], "people/household", ASSUMED,
                            "Urban households are smaller than the 4.4 national mean; exact figure assumed",
                            "urban household count, and therefore TAM and SAM", 3.6, 4.4,
                            "MARKET['urban_hh_size']"),
        "total_hh": _p(MARKET["total_hh"], "households", SOURCED, SOURCES["SRC_HH"],
                       "rural residual in the ownership solve", live="MARKET['total_hh']"),
        "wm_owner_hh": _p(MARKET["wm_owner_hh"], "households", SOURCED, SOURCES["SRC_HH"],
                          "SAM anchor", live="MARKET['wm_owner_hh']"),
        "urban_rural_wm_ratio": _p(MARKET["urban_rural_wm_ratio"], "ratio", SOURCED,
                                   SOURCES["SRC_HH"], "urban ownership solve", 3.0, 5.0,
                                   "MARKET['urban_rural_wm_ratio']"),
        "basket_ownership_urban": _p(MARKET["basket_ownership_urban"], "share", ASSUMED,
                                     "No source for how many urban households use a dedicated hamper",
                                     "bottom-up TAM", 0.55, 0.85, "MARKET['basket_ownership_urban']"),
        "commodity_basket_price": _p(MARKET["commodity_basket_price"], "INR", BENCHMARK,
                                     "Typical discounted selling price, SRC_PEP and SRC_IKEA ranges",
                                     "bottom-up TAM", 300, 550, "MARKET['commodity_basket_price']"),
        "commodity_basket_life_yrs": _p(MARKET["commodity_basket_life_yrs"], "years", ASSUMED,
                                        "From SRC_INTERVIEWS remarks on breakage (n=4), not a durability test",
                                        "replacement frequency, bottom-up TAM", 1.0, 2.5,
                                        "MARKET['commodity_basket_life_yrs']"),
        "reachable_share": _p(MARKET["reachable_share"], "share", ASSUMED,
                              "Online, premium-capable share of urban machine-owning households. "
                              "A judgement call that moves SAM proportionally",
                              "SAM, and the share-of-SAM claim in the plan", 0.10, 0.30,
                              "MARKET['reachable_share']"),
        "flexfold_life_yrs": _p(MARKET["flexfold_life_yrs"], "years", ASSUMED,
                                "Design life implied by the warranty, not a tested figure",
                                "SAM units per year", 3.0, 6.0, "MARKET['flexfold_life_yrs']"),
        "topdown_tam_cr": _p(MARKET["topdown_tam_cr"], "INR crore", SOURCED,
                             SOURCES["SRC_TOPDOWN"], "the top-down TAM cross-check",
                             live="MARKET['topdown_tam_cr']"),
        "inventory_days": _p(INVENTORY_DAYS, "days of COGS", ASSUMED,
                             "Assumed stock cover for an imported-tooling, domestically moulded product",
                             "working-capital exposure (reported, not modelled)", 45, 120,
                             "INVENTORY_DAYS"),
    },
    "Acquisition and plan": {
        **{f"core_units_{y}": _p(p["core_units"], "units/year", ASSUMED,
                                 "A PLAN, not a forecast. No demand model stands behind these volumes",
                                 "every revenue and EBITDA figure",
                                 low=round(p["core_units"] * 0.6), high=round(p["core_units"] * 1.4),
                                 live=f"SOM_PLAN['{y}']['core_units']")
           for y, p in SOM_PLAN.items()},
        **{f"d2c_cac_{y}": _p(p["d2c_cac"], "INR/new customer", BENCHMARK,
                              "Performance-marketing cost per new D2C customer, benchmarked on Indian "
                              "D2C home categories. Excludes the brand spend booked as fixed_marketing",
                              "CM3, break-even, the CAC slider",
                              low=round(p["d2c_cac"] * 0.7), high=round(p["d2c_cac"] * 1.6),
                              live=f"SOM_PLAN['{y}']['d2c_cac']")
           for y, p in SOM_PLAN.items()},
        **{f"amazon_acos_{y}": _p(p["amazon_acos"], "share of Amazon GMV", BENCHMARK,
                                  "Advertising cost of sale typical of a contested home-storage keyword set",
                                  "CM3 on Amazon, the ACoS slider",
                                  low=round(p["amazon_acos"] * 0.7, 3), high=round(p["amazon_acos"] * 1.6, 3),
                                  live=f"SOM_PLAN['{y}']['amazon_acos']")
           for y, p in SOM_PLAN.items()},
        **{f"accessory_attach_{y}": _p(p["accessory_attach"], "accessory units per core unit", ASSUMED,
                                       "The attach rate carries a large share of lifetime margin and has "
                                       "no empirical basis whatsoever. Treat as the key risk",
                                       "accessory revenue, CM2, the attach slider",
                                       low=round(p["accessory_attach"] * 0.5, 3),
                                       high=min(1.2, round(p["accessory_attach"] * 1.3, 3)),
                                       live=f"SOM_PLAN['{y}']['accessory_attach']")
           for y, p in SOM_PLAN.items()},
        **{f"fixed_marketing_{y}": _p(p["fixed_marketing"], "INR/year", ASSUMED,
                                      "Brand, content and agency spend not attributable to a customer. "
                                      "Kept separate from CAC to avoid double counting acquisition",
                                      "CM3", low=p["fixed_marketing"] * 0.6, high=p["fixed_marketing"] * 1.5,
                                      live=f"SOM_PLAN['{y}']['fixed_marketing']")
           for y, p in SOM_PLAN.items()},
        **{f"repeat_core_share_{y}": _p(v, "share of core units", ASSUMED,
                                        "Repeat core purchases of a four-year durable are rare; assumed low "
                                        "and rising as the base grows",
                                        "how many core units attract new-customer CAC",
                                        low=0.0, high=round(v * 2.5, 3) if v else 0.05,
                                        live=f"REPEAT_CORE_SHARE['{y}']")
           for y, v in REPEAT_CORE_SHARE.items()},
    },
    "Fixed operating costs": {
        f"opex_{y}_{line}": _p(val, "INR/year",
                               BENCHMARK if line in ("rent_and_3pl_fixed", "platform_and_it") else ASSUMED,
                               {"team": "Headcount plan priced at market salaries for a 5-to-15 person D2C team",
                                "rent_and_3pl_fixed": "Small office plus the fixed component of a 3PL contract",
                                "tooling_amortisation": f"Derived: TOOLING_CAPEX {TOOLING_CAPEX:,.0f} over "
                                                        f"{TOOLING_LIFE_YRS} years",
                                "platform_and_it": "Storefront, apps, analytics and hosting subscriptions",
                                "support": "Customer-service tooling and part-time agent cost",
                                "insurance_audit_legal": "Audit, compliance, trademark and insurance"}[line],
                               "fixed OpEx, EBITDA, break-even, the OpEx slider",
                               low=val * 0.7, high=val * 1.4, live=f"OPEX['{y}']['{line}']")
        for y, lines in OPEX.items() for line, val in lines.items()
    },
    "Tooling": {
        "tooling_capex": _p(TOOLING_CAPEX, "INR", ASSUMED,
                            "Mould, jig and die investment. No quotation obtained",
                            "annual amortisation charge", 2.5e6, 7e6, "TOOLING_CAPEX"),
        "tooling_life_yrs": _p(TOOLING_LIFE_YRS, "years", ASSUMED,
                               "Assumed mould life in cycles", "annual amortisation charge", 3, 6,
                               "TOOLING_LIFE_YRS"),
    },
}

#: The six inputs the Parameters tab exposes as live sliders, because these are
#: the ones that actually move Year-3 EBITDA. Confirmed by opex.tornado().
SLIDERS = ["cogs", "cac", "returns", "attach", "acos", "opex"]


def params_frame():
    """Flat table of the registry, for the app and the model cards."""
    import pandas as pd
    rows = []
    for group, items in PARAMS.items():
        for name, p in items.items():
            rows.append({"group": group, "parameter": name, "value": p.value, "unit": p.unit,
                         "kind": p.kind, "low": p.low, "high": p.high,
                         "range_width_vs_value": round(p.spread, 2) if p.spread is not None else None,
                         "drives": p.drives, "source": p.source})
    return pd.DataFrame(rows)


def kind_counts() -> dict:
    return params_frame()["kind"].value_counts().to_dict()


def check_registry() -> list[str]:
    """Verify each registered value still equals the value the code uses.

    Returns a list of human-readable mismatches; empty means the registry is
    faithful. Called by the test suite so a hand-edit to SKUS or OPEX cannot
    leave the published parameter table lying.
    """
    scope = {"SKUS": SKUS, "CHANNELS": CHANNELS, "MARKET": MARKET, "SOM_PLAN": SOM_PLAN,
             "OPEX": OPEX, "REPEAT_CORE_SHARE": REPEAT_CORE_SHARE, "GST_THRESHOLD": GST_THRESHOLD,
             "GST_LOW": GST_LOW, "GST_HIGH": GST_HIGH, "RESTOCK_RATE": RESTOCK_RATE,
             "REVERSE_SHIP_MULT": REVERSE_SHIP_MULT, "INVENTORY_DAYS": INVENTORY_DAYS,
             "TOOLING_CAPEX": TOOLING_CAPEX, "TOOLING_LIFE_YRS": TOOLING_LIFE_YRS}
    problems = []
    for group, items in PARAMS.items():
        for name, p in items.items():
            if not p.live:
                continue
            try:
                actual = eval(p.live, {"__builtins__": {}}, scope)  # noqa: S307 - fixed literal paths
            except Exception as exc:                                # pragma: no cover
                problems.append(f"{group}/{name}: cannot resolve live path {p.live!r} ({exc})")
                continue
            if not _close(actual, p.value):
                problems.append(f"{group}/{name}: registry says {p.value!r} but {p.live} is {actual!r}")
            if p.low is not None and p.high is not None and not (p.low <= p.value <= p.high):
                problems.append(f"{group}/{name}: value {p.value!r} outside its stated range "
                                f"[{p.low!r}, {p.high!r}]")
            if p.kind not in (SOURCED, BENCHMARK, ASSUMED):
                problems.append(f"{group}/{name}: unknown kind {p.kind!r}")
    return problems


def _close(a, b, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b
