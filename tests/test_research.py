import numpy as np
import pandas as pd

from neatnest import recovery, research
from neatnest.config import DATA_DIR
from neatnest.generate_data import TRUE_PARAMS, true_value


def test_kmo_and_bartlett_on_independent_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 6)))
    k, _ = research.kmo(X)
    chi2, dof, p = research.bartlett(X)
    assert dof == 15 and p > 0.01 and k < 0.6


def test_factor_analysis_recovers_the_three_planted_factors():
    """Recovery test: three factors were planted, so three should come back.

    This is a legitimate check because the target is declared in TRUE_PARAMS and
    the extraction is independent of it.
    """
    fa = research.factor_analysis(pd.read_csv(DATA_DIR / "survey.csv"))
    assert fa["n_factors"] == 3 and fa["kmo"] > 0.7
    L = fa["loadings"]
    assert L.loc["website_visits_month", "Digital Engagement"] > 0.7
    assert L.loc["monthly_income_k", "Spending Power"] > 0.7
    assert L.loc["brand_loyalty", "Loyalty & Promo Response"] > 0.7


def test_every_item_loads_on_its_intended_factor():
    """The structural claim, checked item by item against the planted assignment."""
    fr = recovery.factor_recovery(pd.read_csv(DATA_DIR / "survey.csv"))
    planted = fr[fr["factor"] != "(none planted)"]
    assert planted["loads_on_intended_factor"].all()
    # Estimated loadings should sit near the *recoverable* target, not the nominal
    # one - the latent variance correction matters (see recovery.implied_loading).
    assert (planted["abs_error"] < 0.2).all()


def test_age_has_no_planted_loading_but_cross_loads_anyway():
    """Documents a real artefact rather than asserting it away.

    `age` carries no planted loading, yet picks up a cross-loading because age is
    set by persona and persona shifts all three latents. If this ever drops to
    zero the generator has changed.
    """
    fr = recovery.factor_recovery(pd.read_csv(DATA_DIR / "survey.csv")).set_index("item")
    assert fr.loc["age", "nominal_loading"] == 0.0
    assert fr.loc["age", "max_cross_loading"] > 0.1


def test_varimax_preserves_communalities():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 3))
    R = research.varimax(A)
    assert np.allclose((A ** 2).sum(1), (R ** 2).sum(1))


def test_van_westendorp_ordering():
    """A structural property of the method, independent of the planted values."""
    vw = research.van_westendorp(pd.read_csv(DATA_DIR / "survey.csv"))
    assert vw["PMC"] <= vw["OPP"] <= vw["IPP"] <= vw["PME"]


def test_van_westendorp_recovers_the_planted_anchor():
    """The price band is constructed around an anchor, so recovering it is the test.

    The indifference price point is the crossing that should land on the anchor.
    Stating it this way makes the circularity explicit instead of presenting the
    band as a willingness-to-pay finding.
    """
    survey = pd.read_csv(DATA_DIR / "survey.csv")
    anchor = true_value("van_westendorp", "anchor_inr")
    pr = recovery.price_recovery(survey).set_index("statistic")
    ipp = pr.loc["IPP - Indifference price point", "ratio_to_anchor"]
    assert 0.95 <= ipp <= 1.05, f"IPP should recover the planted anchor of {anchor}, got ratio {ipp}"


def test_kano_decode_table_is_correct():
    """Tests the DECODE LOGIC, which is real code, not the planted categories.

    The previous version of this test asserted that `reinforced_handle` comes back
    "Must-be". It was planted with M = 0.55, so that test only confirmed that
    sampling from a distribution with mode M returns M - it could not fail for any
    reason related to the code. These cases exercise the functional/dysfunctional
    lookup against the standard Kano table instead.
    """
    cases = {(1, 5): "O", (2, 5): "M", (1, 3): "A", (3, 3): "I", (5, 1): "R", (1, 1): "Q"}
    for (f, d), expected in cases.items():
        assert research.KANO_TABLE[f][d - 1] == expected, f"functional={f}, dysfunctional={d}"


def test_kano_output_is_a_printout_of_its_inputs():
    """Documents the circularity rather than hiding it.

    The only quantity the Kano analysis genuinely estimates is the inconsistent-
    answer rate, which shows up as Questionable and Reverse responses. Everything
    else is the planted category distribution read back out.
    """
    k = research.kano(pd.read_csv(DATA_DIR / "survey.csv")).set_index("feature")
    planted_rate = TRUE_PARAMS["kano_categories"]["inconsistent_answer_rate"]
    n = 600
    for feat in k.index:
        odd = (k.loc[feat, "Reverse"] + k.loc[feat, "Questionable"]) / n
        assert odd < planted_rate * 3, f"{feat}: inconsistent share {odd:.3f} far above planted rate"
