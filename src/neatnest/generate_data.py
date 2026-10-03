"""Simulated Neat Nest launch-year data universe.

All four datasets come from one seed so that they describe the same fictional
company. They are SYNTHETIC: the structure and effect sizes are designed to
mirror the patterns found in the original coursework and the primary
interviews, not observed from a real business.

  survey.csv     600 launch-audience respondents (Phase 0): factor items,
                 Van Westendorp price points, Kano answers, laundry context
  campaign.csv   5,000 prospects reached by the Phase-1 launch campaign
  customers.csv  8,000 registered FlexFold owners (QR warranty registration),
                 month-6 snapshot features + month 7-12 lapse outcome
  sales.csv      daily order lines, Apr 2027 - Mar 2028, by city x SKU x channel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CITIES, CITY_SLEEVE, DATA_DIR, LAUNCH_CITIES, SEED, SKUS,
                     price_ex_gst)

PERSONAS = ["Anjali (dual-income 2BHK manager)", "Hostel/PG student",
            "Young couple, first apartment", "Senior household"]


# ---------------------------------------------------------------------------
def make_survey(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    persona = rng.choice(PERSONAS, size=n, p=[0.42, 0.22, 0.24, 0.12])
    is_student = persona == PERSONAS[1]
    is_senior = persona == PERSONAS[3]

    # Three independent latent traits (standard normal)
    dig = rng.normal(0, 1, n) + 0.5 * is_student - 0.6 * is_senior
    spend = rng.normal(0, 1, n) - 1.0 * is_student + 0.3 * (persona == PERSONAS[0])
    loyal = rng.normal(0, 1, n) + 0.3 * is_senior

    def item(f, loading, mean, sd, lo=None, hi=None, rnd=None):
        x = loading * f + np.sqrt(1 - loading ** 2) * rng.normal(0, 1, n)
        x = mean + sd * x
        if lo is not None or hi is not None:
            x = np.clip(x, lo, hi)
        return np.round(x, rnd) if rnd is not None else x

    age = np.where(is_student, rng.integers(18, 24, n),
                   np.where(is_senior, rng.integers(58, 75, n), rng.integers(25, 46, n)))
    df = pd.DataFrame({
        "respondent_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "city": rng.choice(LAUNCH_CITIES + ["Bengaluru", "Hyderabad", "Delhi NCR"], n,
                           p=[0.32, 0.22, 0.18, 0.14, 0.14]),
        "persona": persona,
        "age": age,
        # Digital engagement items
        "website_visits_month": item(dig, 0.85, 9, 4, 0, None, 0),
        "social_interactions_month": item(dig, 0.82, 22, 9, 0, None, 0),
        "email_open_rate": item(dig, 0.74, 28, 10, 0, 100, 1),
        # Spending power items
        "monthly_income_k": item(spend, 0.80, 95, 35, 12, None, 0),
        "home_org_spend_12m": item(spend, 0.78, 9000, 4200, 0, None, -2),
        "spending_score": item(spend, 0.76, 60, 15, 1, 100, 0),
        # Loyalty & promotion responsiveness items
        "brand_loyalty": item(loyal, 0.80, 3.0, 1.0, 1, 5, 0),
        "promo_response": item(loyal, 0.78, 2.5, 1.0, 0, 5, 1),
        "referral_likelihood": item(loyal, 0.72, 6.5, 2.0, 0, 10, 0),
    })

    # Laundry context
    df["loads_per_week"] = np.clip(np.round(rng.normal(4.2, 1.3, n) - 1.5 * is_student), 1, 9).astype(int)
    df["floors_to_machine"] = np.where(is_student, rng.integers(1, 4, n), rng.integers(0, 2, n))
    df["home_type"] = np.where(is_student, "Hostel/PG",
                               rng.choice(["1BHK", "2BHK", "3BHK"], n, p=[0.30, 0.52, 0.18]))
    df["current_basket_price"] = np.round(np.clip(rng.lognormal(np.log(420), 0.45, n), 120, 3000), -1)
    df["basket_broke_last_2y"] = (rng.random(n) < 0.58).astype(int)

    # Van Westendorp price points (Rs) for the hero FlexFold, after seeing the concept
    base = 1650 * np.exp(0.16 * spend + 0.10 * df["basket_broke_last_2y"] - 0.12 * is_student
                         + rng.normal(0, 0.18, n))
    df["vw_too_cheap"] = np.round(base * rng.uniform(0.38, 0.50, n), -1)
    df["vw_cheap"] = np.round(base * rng.uniform(0.70, 0.85, n), -1)
    df["vw_expensive"] = np.round(base * rng.uniform(1.12, 1.30, n), -1)
    df["vw_too_expensive"] = np.round(base * rng.uniform(1.45, 1.75, n), -1)

    # Kano: functional / dysfunctional answers (1 like, 2 must-be, 3 neutral, 4 live-with, 5 dislike)
    kano_profile = {  # probability of each Kano category per feature
        "reinforced_handle": {"M": 0.55, "O": 0.25, "A": 0.05, "I": 0.15},
        "wheels_tilt_roll": {"M": 0.15, "O": 0.50, "A": 0.25, "I": 0.10},
        "three_compartments": {"M": 0.10, "O": 0.45, "A": 0.25, "I": 0.20},
        "ventilated_antimicrobial": {"M": 0.15, "O": 0.35, "A": 0.38, "I": 0.12},
        "fold_flat_8cm": {"M": 0.08, "O": 0.30, "A": 0.42, "I": 0.20},
        "heritage_sleeve": {"M": 0.02, "O": 0.10, "A": 0.40, "I": 0.48},
        "weight_sensor_app": {"M": 0.02, "O": 0.08, "A": 0.25, "I": 0.65},
    }
    answer_map = {"M": (2, 5), "O": (1, 5), "A": (1, 3), "I": (3, 3)}
    for feat, probs in kano_profile.items():
        cats = rng.choice(list(probs), n, p=list(probs.values()))
        noise = rng.random(n) < 0.08  # some inconsistent answers -> Questionable/Reverse
        func = np.array([answer_map[c][0] for c in cats])
        dysf = np.array([answer_map[c][1] for c in cats])
        func = np.where(noise, rng.integers(1, 6, n), func)
        dysf = np.where(noise, rng.integers(1, 6, n), dysf)
        df[f"kano_{feat}_f"] = func
        df[f"kano_{feat}_d"] = dysf
    return df


# ---------------------------------------------------------------------------
def make_campaign(rng: np.random.Generator, n: int = 5000) -> pd.DataFrame:
    source = rng.choice(["Meta paid social", "Google search", "Creator partnership", "Housing-society activation"],
                        n, p=[0.42, 0.26, 0.20, 0.12])
    age = np.clip(np.round(rng.normal(33, 8, n)), 18, 70)
    income = np.round(np.clip(rng.lognormal(np.log(85), 0.45, n), 15, 600))
    engagement = np.clip(np.round(rng.normal(52, 18, n)), 0, 100)
    prev = rng.poisson(1.6, n)
    visits = np.clip(rng.poisson(2.2 + engagement / 40), 1, None)
    quiz = (rng.random(n) < 1 / (1 + np.exp(-(-1.6 + 0.025 * (engagement - 50) + 0.25 * visits)))).astype(int)
    z = (-6.05 + 0.42 * prev + 0.032 * engagement + 1.05 * quiz + 0.07 * visits
         + 0.0015 * (income - 85) + 0.0 * (age - 33)
         + 0.35 * (source == "Housing-society activation") + 0.20 * (source == "Google search"))
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "prospect_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "city": rng.choice(LAUNCH_CITIES, n, p=[0.6, 0.4]),
        "source": source, "age": age.astype(int), "monthly_income_k": income.astype(int),
        "digital_engagement_score": engagement.astype(int), "prev_home_org_purchases": prev,
        "site_visits": visits, "fit_quiz_completed": quiz, "purchased": y,
    })


# ---------------------------------------------------------------------------
def make_customers(rng: np.random.Generator, n: int = 8000) -> pd.DataFrame:
    sku = rng.choice(["LITE", "FLEX", "HERI"], n, p=[0.20, 0.62, 0.18])
    channel = rng.choice(["D2C", "Amazon"], n, p=[0.52, 0.48])
    city = rng.choice(CITIES, n, p=[0.36, 0.24, 0.16, 0.12, 0.12])
    age = np.where(sku == "LITE", rng.integers(18, 26, n), np.clip(np.round(rng.normal(36, 9, n)), 22, 75)).astype(int)
    engage = rng.normal(0, 1, n)  # latent engagement
    logins = np.clip(np.round(rng.gamma(2.0, 1.1, n) * np.exp(0.45 * engage)), 0, 25)
    wa_open = np.clip(np.round(45 + 15 * engage + rng.normal(0, 10, n)), 0, 100)
    acc_orders = rng.poisson(np.exp(-0.6 + 0.45 * engage + 0.25 * (sku == "HERI")))
    acc_spend = np.round(acc_orders * rng.uniform(260, 420, n), -1)
    delay = np.clip(np.round(rng.gamma(1.2, 1.2, n) + 1.2 * (channel == "Amazon") * rng.random(n)), 0, 12)
    issue = (rng.random(n) < 0.08 + 0.05 * (sku == "LITE")).astype(int)
    tickets = rng.poisson(0.4 + 1.6 * issue + 0.12 * delay)
    review = (rng.random(n) < 1 / (1 + np.exp(-(-0.9 + 0.5 * engage - 0.6 * issue)))).astype(int)
    referrals = rng.poisson(np.exp(-1.1 + 0.5 * engage + 0.35 * (sku == "HERI") - 0.8 * issue))
    nps = np.clip(np.round(8 + 1.1 * engage - 2.4 * issue - 0.25 * delay + rng.normal(0, 1.4, n)), 0, 10)
    discount = np.round(np.clip(rng.beta(2, 9, n) * 100 + 8 * (channel == "Amazon"), 0, 60), 1)

    z = (-1.75 - 0.70 * engage + 0.55 * tickets + 0.16 * delay + 1.20 * issue
         - 0.45 * np.minimum(acc_orders, 3) - 0.40 * np.minimum(referrals, 3) - 0.30 * (sku == "HERI")
         + 0.35 * (sku == "LITE") + 0.010 * discount - 0.12 * (nps - 8))
    lapse = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    core_price = np.array([price_ex_gst(SKUS[s]["mrp"]) for s in sku])
    # 12-month realised value: first order + accessories (incl. months 7-12) + referred first orders
    future_acc = rng.poisson(np.exp(-0.4 + 0.45 * engage) * (1 - 0.75 * lapse)) * rng.uniform(260, 420, n)
    value_12m = np.round(core_price + acc_spend / 1.05 + future_acc / 1.05 + referrals * 0.6 * 1600, 0)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "city": city, "first_sku": sku, "first_channel": channel, "age": age,
        "care_hub_logins_m1_6": logins.astype(int), "whatsapp_open_rate": wa_open,
        "accessory_orders_m1_6": acc_orders, "accessory_spend_m1_6": acc_spend,
        "delivery_delay_days": delay.astype(int), "product_issue_reported": issue,
        "support_tickets_m1_6": tickets, "review_written": review, "referrals_m1_6": referrals,
        "nps_m3": nps.astype(int), "discount_pct_first_order": discount,
        "lapsed_m7_12": lapse, "value_12m": value_12m,
    })


# ---------------------------------------------------------------------------
def make_sales(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2027-04-01", "2028-03-31", freq="D")
    city_start = {"Mumbai": "2027-04-01", "Pune": "2027-04-01", "Bengaluru": "2027-10-01",
                  "Hyderabad": "2027-10-01", "Delhi NCR": "2027-10-01"}
    city_weight = {"Mumbai": 1.0, "Pune": 0.65, "Bengaluru": 0.75, "Hyderabad": 0.6, "Delhi NCR": 0.7}
    base_daily = {"LITE": 2.3, "FLEX": 7.4, "HERI": 1.8, "LINR": 4.6, "CHAR": 3.3, "WHEL": 1.0}
    chan_split = {"D2C": 0.47, "Amazon": 0.53}
    disc = {"D2C": 0.03, "Amazon": 0.08}
    rows = []
    for city in CITIES:
        start = pd.Timestamp(city_start[city])
        for d in days:
            if d < start:
                continue
            age_m = (d - start).days / 30.4
            ramp = min(1.0, 0.35 + 0.16 * age_m)           # launch ramp
            season = 1.25 if d.month in (10, 11) else (1.15 if d.month in (6, 7, 8) else 1.0)  # Diwali, monsoon
            weekend = 1.15 if d.dayofweek >= 5 else 1.0
            for sku, b in base_daily.items():
                mult = 1.0
                d_extra = 0.0
                if sku in ("LINR", "CHAR", "WHEL"):
                    mult *= min(1.0, 0.2 + 0.12 * age_m)    # accessories lag the installed base
                if sku == "HERI" and city == "Hyderabad":
                    mult *= 1.6                             # Kalamkari sleeve resonates locally
                    if d >= pd.Timestamp("2028-03-01"):
                        mult *= 0.12                        # PLANTED: sleeve stock-out in March
                if city == "Hyderabad" and d >= pd.Timestamp("2028-03-01") and sku in ("FLEX", "LITE"):
                    mult *= 0.82                            # spill-over: fewer visits, softer core demand
                    d_extra = 0.07                          # deeper Amazon discounting to compensate
                for ch, share in chan_split.items():
                    lam = b * city_weight[city] * ramp * season * weekend * share * mult
                    units = rng.poisson(lam)
                    if units == 0:
                        continue
                    mrp = SKUS[sku]["mrp"]
                    dpct = disc[ch] + (d_extra if ch == "Amazon" else 0) + rng.choice([0, 0, 0, 0.05])
                    gross = units * mrp * (1 - dpct)
                    net = gross / (1.05 if mrp * (1 - dpct) <= 2500 else 1.18)
                    cost = units * (SKUS[sku]["cogs"] + SKUS[sku]["pack"])
                    rows.append((d.date().isoformat(), city, sku, SKUS[sku]["name"], SKUS[sku]["family"], ch,
                                 CITY_SLEEVE[city] if sku == "HERI" else "", units, mrp, round(dpct, 3),
                                 round(net, 2), cost))
    # B2B co-living / hostel orders: monthly bulk orders
    for m in pd.date_range("2027-05-01", "2028-03-01", freq="MS"):
        for city in ["Pune", "Mumbai", "Bengaluru"]:
            if m < pd.Timestamp(city_start[city]) + pd.DateOffset(months=1):
                continue
            if rng.random() < 0.6:
                units = int(rng.choice([50, 60, 80, 100]))
                net = units * SKUS["PRO"]["mrp"] / 1.05
                rows.append(((m + pd.Timedelta(days=int(rng.integers(3, 25)))).date().isoformat(), city, "PRO",
                             SKUS["PRO"]["name"], "Core", "B2B", "", units, SKUS["PRO"]["mrp"], 0.0,
                             round(net, 2), units * (SKUS["PRO"]["cogs"] + SKUS["PRO"]["pack"])))
    df = pd.DataFrame(rows, columns=["date", "city", "sku", "product", "family", "channel", "sleeve",
                                     "units", "mrp", "discount_pct", "net_sales", "cost"])
    df = df.sort_values(["date", "city", "sku", "channel"]).reset_index(drop=True)
    df.insert(0, "line_id", [f"L{i:06d}" for i in range(1, len(df) + 1)])
    return df


def build_all(out_dir=DATA_DIR, seed: int = SEED) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = {"survey": make_survey(rng), "campaign": make_campaign(rng),
              "customers": make_customers(rng), "sales": make_sales(rng)}
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return frames


if __name__ == "__main__":
    f = build_all()
    for k, v in f.items():
        print(k, v.shape)
