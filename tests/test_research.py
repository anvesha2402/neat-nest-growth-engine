import numpy as np
import pandas as pd

from neatnest import research
from neatnest.config import DATA_DIR


def test_kmo_and_bartlett_on_independent_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 6)))
    k, _ = research.kmo(X)
    chi2, dof, p = research.bartlett(X)
    assert dof == 15 and p > 0.01 and k < 0.6


def test_factor_analysis_recovers_three_factors():
    fa = research.factor_analysis(pd.read_csv(DATA_DIR / "survey.csv"))
    assert fa["n_factors"] == 3 and fa["kmo"] > 0.7
    L = fa["loadings"]
    assert L.loc["website_visits_month", "Digital Engagement"] > 0.7
    assert L.loc["monthly_income_k", "Spending Power"] > 0.7
    assert L.loc["brand_loyalty", "Loyalty & Promo Response"] > 0.7


def test_varimax_preserves_communalities():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 3))
    R = research.varimax(A)
    assert np.allclose((A ** 2).sum(1), (R ** 2).sum(1))


def test_van_westendorp_ordering():
    vw = research.van_westendorp(pd.read_csv(DATA_DIR / "survey.csv"))
    assert vw["PMC"] <= vw["OPP"] <= vw["IPP"] <= vw["PME"]
    assert vw["PMC"] <= 1799 <= vw["PME"]


def test_kano_classifies_handle_as_must_be():
    k = research.kano(pd.read_csv(DATA_DIR / "survey.csv")).set_index("feature")
    assert k.loc["reinforced_handle", "category"] == "Must-be"
    assert k.loc["weight_sensor_app", "category"] == "Indifferent"
