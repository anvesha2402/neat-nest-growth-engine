"""Layer 2 - Customer understanding.

Exploratory factor analysis (implemented from first principles in numpy so the
maths is visible and there is no dependency on factor_analyzer), Van Westendorp
price sensitivity and Kano feature classification on the Phase-0 survey.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FACTOR_ITEMS = ["website_visits_month", "social_interactions_month", "email_open_rate",
                "monthly_income_k", "home_org_spend_12m", "spending_score",
                "brand_loyalty", "promo_response", "referral_likelihood", "age"]
FACTOR_NAMES = ["Digital Engagement", "Spending Power", "Loyalty & Promo Response"]


# ----------------------------- factor analysis -----------------------------
def kmo(df: pd.DataFrame) -> tuple[float, pd.Series]:
    r = np.corrcoef(df.values, rowvar=False)
    inv = np.linalg.inv(r)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    np.fill_diagonal(r, 0); np.fill_diagonal(partial, 0)
    r2, p2 = r ** 2, partial ** 2
    overall = r2.sum() / (r2.sum() + p2.sum())
    per_item = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    return float(overall), pd.Series(per_item, index=df.columns)


def bartlett(df: pd.DataFrame) -> tuple[float, int, float]:
    n, p = df.shape
    r = np.corrcoef(df.values, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(r))
    dof = p * (p - 1) // 2
    return float(chi2), dof, float(stats.chi2.sf(chi2, dof))


def varimax(loadings: np.ndarray, gamma: float = 1.0, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    p, k = loadings.shape
    rot = np.eye(k)
    d = 0.0
    for _ in range(iters):
        lam = loadings @ rot
        u, s, vh = np.linalg.svd(loadings.T @ (lam ** 3 - (gamma / p) * lam @ np.diag((lam ** 2).sum(0))))
        rot = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return loadings @ rot


def factor_analysis(survey: pd.DataFrame, n_factors: int | None = None) -> dict:
    X = survey[FACTOR_ITEMS].astype(float)
    k_overall, k_items = kmo(X)
    chi2, dof, p = bartlett(X)
    r = np.corrcoef(X.values, rowvar=False)
    eigval, eigvec = np.linalg.eigh(r)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    if n_factors is None:
        n_factors = int((eigval > 1).sum())
    raw = eigvec[:, :n_factors] * np.sqrt(eigval[:n_factors])  # principal-component extraction
    rot = varimax(raw)
    # order and sign factors so the naming is stable
    lead = {"Digital Engagement": "website_visits_month", "Spending Power": "monthly_income_k",
            "Loyalty & Promo Response": "brand_loyalty"}
    cols, names = [], []
    for name, item in lead.items():
        j = int(np.argmax(np.abs(rot[FACTOR_ITEMS.index(item)])))
        sign = np.sign(rot[FACTOR_ITEMS.index(item), j])
        cols.append(rot[:, j] * sign); names.append(name)
    L = pd.DataFrame(np.column_stack(cols), index=FACTOR_ITEMS, columns=names)
    communality = (L ** 2).sum(axis=1)
    ss = (L ** 2).sum(axis=0)
    # Bartlett-style factor scores via regression on standardised items
    Z = (X - X.mean()) / X.std(ddof=0)
    W = np.linalg.solve(r, L.values)
    scores = pd.DataFrame(Z.values @ W, columns=names, index=survey.index)
    return {"kmo": k_overall, "kmo_items": k_items, "bartlett_chi2": chi2, "bartlett_dof": dof,
            "bartlett_p": p, "eigenvalues": eigval, "n_factors": n_factors, "loadings": L,
            "communality": communality, "variance_pct": ss / len(FACTOR_ITEMS) * 100,
            "cumulative_pct": float(ss.sum() / len(FACTOR_ITEMS) * 100), "scores": scores}


def persona_profiles(survey: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([survey[["persona"]], scores], axis=1)
    prof = df.groupby("persona").mean().round(2)
    prof["n"] = survey["persona"].value_counts()
    return prof.sort_values("n", ascending=False)


# ----------------------------- Van Westendorp ------------------------------
def van_westendorp(survey: pd.DataFrame, grid=None) -> dict:
    grid = np.arange(400, 3601, 10) if grid is None else grid
    tc = np.array([(survey["vw_too_cheap"] >= g).mean() for g in grid])       # "too cheap" (descending)
    ch = np.array([(survey["vw_cheap"] >= g).mean() for g in grid])            # "cheap" (descending)
    ex = np.array([(survey["vw_expensive"] <= g).mean() for g in grid])        # "expensive" (ascending)
    te = np.array([(survey["vw_too_expensive"] <= g).mean() for g in grid])    # "too expensive" (ascending)

    def cross(a, b):
        idx = int(np.argmin(np.abs(a - b)))
        return float(grid[idx])
    curves = pd.DataFrame({"price": grid, "Too cheap": tc, "Cheap": ch, "Expensive": ex, "Too expensive": te})
    return {"curves": curves,
            "PMC": cross(tc, ex),   # point of marginal cheapness
            "OPP": cross(tc, te),   # optimal price point
            "IPP": cross(ch, ex),   # indifference price point
            "PME": cross(ch, te)}   # point of marginal expensiveness


# ----------------------------------- Kano ----------------------------------
KANO_TABLE = {  # rows = functional answer 1..5, cols = dysfunctional 1..5
    1: ["Q", "A", "A", "A", "O"],
    2: ["R", "I", "I", "I", "M"],
    3: ["R", "I", "I", "I", "M"],
    4: ["R", "I", "I", "I", "M"],
    5: ["R", "R", "R", "R", "Q"],
}
KANO_LABEL = {"M": "Must-be", "O": "Performance", "A": "Attractive (delighter)", "I": "Indifferent",
              "R": "Reverse", "Q": "Questionable"}


def kano(survey: pd.DataFrame) -> pd.DataFrame:
    feats = sorted({c[5:-2] for c in survey.columns if c.startswith("kano_")})
    rows = []
    for f in feats:
        cats = [KANO_TABLE[a][b - 1] for a, b in zip(survey[f"kano_{f}_f"], survey[f"kano_{f}_d"])]
        s = pd.Series(cats).value_counts()
        a, o, m, i = (s.get(k, 0) for k in "AOMI")
        better = (a + o) / max(a + o + m + i, 1)
        worse = -(o + m) / max(a + o + m + i, 1)
        top = s.drop(labels=[x for x in ("Q", "R") if x in s.index]).idxmax()
        rows.append({"feature": f, **{KANO_LABEL[k]: int(s.get(k, 0)) for k in "MOAIRQ"},
                     "category": KANO_LABEL[top], "better": round(better, 3), "worse": round(worse, 3)})
    return pd.DataFrame(rows).sort_values("better", ascending=False).reset_index(drop=True)
