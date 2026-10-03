"""Single source of truth for Neat Nest business assumptions.

Every number used by the market model, the data simulator, the Streamlit app
and the project brief is defined here once. Each assumption carries a source
tag: SRC_* for published sources, EST_* for labelled estimates that need
validation (vendor quotes, primary research).
"""
from __future__ import annotations

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


def price_ex_gst(mrp: float) -> float:
    rate = GST_LOW if mrp <= GST_THRESHOLD else GST_HIGH
    return mrp / (1 + rate)

# Chart palette: brand-adjacent hues validated for colour-vision deficiency
# (adjacent CVD dE >= 9, normal-vision dE >= 21 on a light surface).
SERIES = ["#3a6fb0", "#d0673a", "#1f9e78", "#d9a020", "#b5487a"]
INK, INK_MUTED, GRID = "#2F2A26", "#6b665f", "#e6e1d8"
