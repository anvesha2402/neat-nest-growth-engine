"""How much of each headline number is the seed?

Every model metric in this project comes from one simulated draw under SEED =
2027, and the earlier version of the write-up quoted those metrics to two or
three decimal places. That precision is not real: it describes one sample from
a generator, and a different seed would move it.

This script re-runs the whole simulation and model stack across several seeds
and reports mean, min and max for each headline metric, so the project can
publish a range instead of a false decimal.

Two things worth noting in the output:

  * The ECONOMICS are deterministic. Market sizing, unit economics and the
    EBITDA bridge are computed from the parameter registry, not from the
    simulated data, so they do not move with the seed at all. Their uncertainty
    is parameter uncertainty, which is what opex.tornado() measures instead.

  * The RECOVERY VERDICTS are the interesting column. If a planted effect is
    recovered under one seed and missed under another, it was marginal all
    along, and that is worth knowing before anyone quotes it.

    python scripts/seed_stability.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neatnest import propensity, recovery, research, retention  # noqa: E402
from neatnest.config import REPORT_DIR, SEED  # noqa: E402
from neatnest.generate_data import build_all  # noqa: E402

warnings.filterwarnings("ignore")


def one_seed(seed: int, out_dir: Path) -> dict:
    frames = build_all(out_dir, seed)
    survey, campaign, customers = frames["survey"], frames["campaign"], frames["customers"]

    ev = propensity.evaluate(campaign)
    _, ptable = propensity.statsmodels_logit(campaign)
    lm = retention.lapse_models(customers)
    vc = retention.value_model_comparison(customers)
    sg = retention.segments(customers)
    fa = research.factor_analysis(survey)
    vw = research.van_westendorp(survey)
    rep = recovery.recovery_report(survey, campaign, customers, fa=fa, vw=vw)
    s = rep["summary"]

    return {
        "seed": seed,
        "campaign_base_rate": ev["base_rate"],
        "campaign_buyers": int(campaign["purchased"].sum()),
        "propensity_cv_auc": ev["cv_auc_mean"],
        "propensity_holdout_auc": ev["holdout"]["auc"],
        "quiz_odds_ratio": float(ptable.loc["fit_quiz_completed", "odds_ratio"]),
        "lapse_base_rate": lm["base_rate"],
        "lapse_auc": float(lm["metrics"]["auc"].max()),
        "lapse_brier": float(lm["metrics"].loc[lm["best"], "brier"]),
        "value_r2_leaky": float(vc["leaky"]["metrics"].loc["OLS", "r2"]),
        "value_r2_forward": float(vc["forward"]["metrics"].loc["OLS", "r2"]),
        "silhouette_k2": sg["silhouette"][2],
        "silhouette_k4": sg["silhouette"][4],
        "vw_opp": float(vw["OPP"]),
        "vw_ipp": float(vw["IPP"]),
        "factors_extracted": int(fa["n_factors"]),
        "kmo": float(fa["kmo"]),
        "recovered": s.get("recovered", 0),
        "correctly_null": s.get("correctly null", 0),
        "missed": s.get("missed", 0),
        "biased": s.get("attenuated", 0) + s.get("inflated", 0),
        "false_positives": rep["false_positives"],
    }


def run(n_seeds: int = 5, base_seed: int = SEED) -> pd.DataFrame:
    seeds = [base_seed + i for i in range(n_seeds)]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for i, s in enumerate(seeds, 1):
            print(f"  seed {s} ({i}/{len(seeds)}) ...", flush=True)
            rows.append(one_seed(s, out))
    return pd.DataFrame(rows).set_index("seed")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.agg(["mean", "min", "max", "std"]).T
    stats["range"] = stats["max"] - stats["min"]
    stats["canonical_seed"] = df.loc[SEED] if SEED in df.index else np.nan
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="write reports/seed_stability.csv and .json")
    a = ap.parse_args()

    print(f"Re-running the full stack across {a.seeds} seeds (base {SEED}).")
    df = run(a.seeds)
    stats = summarise(df)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    print("\nPER-SEED")
    print(df.round(4).to_string())
    print("\nSUMMARY (mean / min / max / range)")
    print(stats[["mean", "min", "max", "range", "canonical_seed"]].round(4).to_string())

    print("\nHOW TO QUOTE THESE")
    for k, dp in [("propensity_cv_auc", 2), ("lapse_auc", 2), ("value_r2_leaky", 2),
                  ("value_r2_forward", 2), ("silhouette_k4", 2), ("quiz_odds_ratio", 2)]:
        r = stats.loc[k]
        print(f"  {k:24s} {r['mean']:.{dp}f} +/- {r['range'] / 2:.{dp}f}   "
              f"(seen {r['min']:.{dp}f} to {r['max']:.{dp}f})")

    unstable = stats.loc[["recovered", "missed", "biased", "false_positives"], "range"]
    print("\nRECOVERY VERDICT STABILITY")
    for k, v in unstable.items():
        print(f"  {k:18s} varies by {v:.0f} across seeds"
              + ("  <- marginal terms, do not quote a single-seed count" if v else "  <- stable"))

    if a.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORT_DIR / "seed_stability.csv")
        payload = {"n_seeds": a.seeds, "base_seed": SEED,
                   "per_seed": json.loads(df.to_json(orient="index")),
                   "summary": json.loads(stats[["mean", "min", "max", "range"]].to_json(orient="index"))}
        (REPORT_DIR / "seed_stability.json").write_text(json.dumps(payload, indent=1))
        print(f"\nWrote {REPORT_DIR / 'seed_stability.csv'} and seed_stability.json")
