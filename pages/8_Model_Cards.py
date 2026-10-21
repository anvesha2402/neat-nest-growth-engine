import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import (fig_layout, frame, header, note, page_setup,  # noqa: E402
                                report_csv, results)
from neatnest.config import SERIES  # noqa: E402

page_setup("Model Cards")
R = results()

header("Disclosures", "Model cards and parameter recovery",
       "Every effect size in this project was planted by the simulator before any model saw it. "
       "This page reports how much of it the models got back, and where they failed.")

note("<b>There is no real company and no real data.</b> FlexFold is a designed product concept and all four "
     "datasets are simulated from parameters declared in <code>TRUE_PARAMS</code>. So no number here is a finding "
     "about the market. Each one is an attempt to recover a value that was written down first — and the attempts "
     "fail in four distinct ways, which is the actual result.")

# ---------------------------------------------------------------- recovery
st.subheader("Parameter recovery")
rec = R["recovery"]
st.markdown(f"**{rec['headline']}**")

c = frame(rec["coefficients"])
order = ["recovered", "correctly null", "attenuated", "inflated", "missed", "not recoverable",
         "FALSE POSITIVE"]
counts = {v: int((c["verdict"] == v).sum()) for v in order if (c["verdict"] == v).sum()}
cols = st.columns(len(counts) + 1)
for i, (v, n) in enumerate(counts.items()):
    cols[i].metric(v.title(), n)
cols[-1].metric("Clean share", f"{rec['clean_share'] * 100:.0f}%",
                help="Recovered or correctly-null, as a share of terms that were recoverable at all.")

show = c[["block", "term", "planted", "estimated", "ci_low", "ci_high", "p_value", "verdict", "note"]]
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"planted": st.column_config.NumberColumn("Planted", format="%.4f"),
                            "estimated": st.column_config.NumberColumn("Estimated", format="%.4f"),
                            "ci_low": st.column_config.NumberColumn("CI low", format="%.3f"),
                            "ci_high": st.column_config.NumberColumn("CI high", format="%.3f"),
                            "p_value": st.column_config.NumberColumn("p", format="%.4f"),
                            "note": st.column_config.TextColumn("Why", width="large")})

st.markdown("""
**How to read the verdicts.** *Recovered* means the confidence interval covers the planted value.
*Missed* means the planted effect is real and the model cannot see it at this sample size — a power
failure, not a modelling error. *Not recoverable* means the driver is latent or was planted through a
different functional form than the model assumes, so the surrounding coefficients are confounded rather
than wrong. The two planted nulls (`age`, `source[Meta paid social]`) are controls: a method that cannot
hold a true zero at zero is not measuring anything.
""")

st.markdown("##### The two results worth dwelling on")
a, b = st.columns(2)
with a:
    st.markdown("""
**Two real channel effects were planted and neither is detectable.**
With 5,000 prospects, a 7.5% base rate and four channels, a 0.35 log-odds channel effect cannot be
separated from zero. An earlier version of this work reported *"the strongest lever is not a channel"*
as a finding. It is not a finding — it is a power failure. The honest statement is that **at this sample
size the design cannot detect the channel effects it contains.**
    """)
with b:
    st.markdown("""
**The strongest driver of lapse is invisible.**
A latent engagement variable with a planted coefficient of −0.70 is never written to `customers.csv`.
It leaks into logins, WhatsApp opens, accessory orders, referrals and NPS, so all of those coefficients
are confounded by it. The visible symptom: NPS comes back at −0.21 against a planted −0.12, having
absorbed part of the missing driver. Textbook omitted-variable bias.
    """)

# ---------------------------------------------------------------- leakage
st.divider()
st.subheader("Target leakage, demonstrated")
V = R["value"]
lk = frame(V["leakage_comparison"])
l1, l2 = st.columns([2, 3])
with l1:
    st.metric("OLS R² with leakage", f"{lk['r2_ols'].iloc[0]:.3f}")
    st.metric("OLS R² without it", f"{lk['r2_ols'].iloc[1]:.3f}", delta=f"−{V['r2_drop']:.3f}")
with l2:
    st.dataframe(lk[["target", "contains_model_features", "leaking_features", "r2_ols", "zero_share"]],
                 hide_index=True, width="stretch",
                 column_config={"target": "Target", "contains_model_features": "Leaks?",
                                "leaking_features": st.column_config.TextColumn("Features inside the target",
                                                                               width="medium"),
                                "r2_ols": st.column_config.NumberColumn("OLS R²", format="%.3f"),
                                "zero_share": st.column_config.NumberColumn("Zero share", format="%.0%")})
st.markdown(f"**{V['headline']}**")
st.caption(V["caveat"])

# ---------------------------------------------------------------- k choice
st.divider()
st.subheader("Why k = 4 when the fit prefers k = 2")
S = R["segments"]
ch = S["choice"]
k1, k2 = st.columns([3, 2])
with k1:
    sil = {int(k): v for k, v in S["silhouette"].items()}
    ks = sorted(sil)
    fig = go.Figure()
    fig.add_scatter(x=ks, y=[sil[k] for k in ks], mode="lines+markers", name="Silhouette",
                    line={"color": SERIES[0], "width": 2})
    fig.add_scatter(x=[ch["k_used"]], y=[sil[ch["k_used"]]], mode="markers+text",
                    text=[f"k={ch['k_used']} used"], textposition="bottom center",
                    marker={"color": SERIES[1], "size": 14}, name="Chosen")
    fig.add_scatter(x=[ch["k_preferred_by_silhouette"]], y=[sil[ch["k_preferred_by_silhouette"]]],
                    mode="markers+text", text=[f"k={ch['k_preferred_by_silhouette']} best fit"],
                    textposition="top center", marker={"color": SERIES[2], "size": 14}, name="Best fit")
    fig.update_layout(xaxis_title="k", yaxis_title="Silhouette", showlegend=False)
    st.plotly_chart(fig_layout(fig, "The full silhouette curve, not just the chosen k", 330), theme=None)
with k2:
    st.metric("Silhouette cost of the choice", f"{ch['silhouette_cost_of_choice']:.3f}")
    st.markdown(f"**{ch['honest_summary']}**")
    st.caption(ch["rationale"])

# ---------------------------------------------------------------- calibration
st.divider()
st.subheader("Calibration, measured rather than asserted")
L = R["lapse"]
met = frame(L["metrics"])
cal1, cal2 = st.columns([2, 3])
with cal1:
    st.dataframe(met[["auc", "brier", "brier_calibrated"]].round(4), width="stretch",
                 column_config={"auc": "AUC", "brier": "Brier",
                                "brier_calibrated": "Brier (isotonic)"})
    st.caption("Both models are class-weighted for ranking, which deliberately distorts the probability "
               "scale. Isotonic recalibration improves the Brier score by about 13%, so the raw scores "
               "rank well and should not be read as probabilities.")
with cal2:
    rel = L.get("reliability")
    if rel:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                        line={"dash": "dot", "color": "#b9b3a8"})
        fig.add_scatter(x=rel["mean_predicted"], y=rel["fraction_positive"], mode="lines+markers",
                        name=L["best"], line={"color": SERIES[0]})
        fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed lapse rate")
        st.plotly_chart(fig_layout(fig, "Reliability curve, held-out split", 330), theme=None)

st.markdown(f"""
**Out-of-fold scoring, and an honest footnote.** The pipeline previously scored every customer with a
model refitted on all rows, so the risk table and the save scenario were graded on their own training
data. Scores are now out of fold. The correction moved the headline reach by
**{L['in_sample_overstatement_pct']:.2f}%** — the leakage was real and the fix is right, but at
n = {L['n']:,} with a linear model it barely mattered. Both numbers are published, because the point is
that this was measured rather than assumed in either direction.
""")

# ---------------------------------------------------------------- copilot
st.divider()
st.subheader("Copilot: two evaluations that measure different things")
ce, ca = R["copilot_eval"]["summary"], R["copilot_adversarial"]["summary"]
e1, e2 = st.columns(2)
with e1:
    st.metric("Fixed regression suite", ce["total"])
    st.markdown("""
Written **during** development, against the behaviour being developed. This number means the copilot
still does what it did yesterday. **It is not an accuracy score** and it does not generalise to unseen
questions. Reporting it as accuracy would be the single most misleading claim in this project.
    """)
with e2:
    st.metric("Held-out adversarial suite", ca["total"], delta="was 1/10 on first run")
    st.markdown("""
Ten attacks written **after** the copilot was finished, with intent to produce a confident wrong answer.
A case passes only if the copilot refuses, asks for clarification, or reports a data error.
    """)

st.markdown("""
##### What the adversarial suite found on first run

It scored **1 of 10**, and the failures were not cosmetic:

- Asked **"What were net sales in Chennai?"** — a city with no data — the planner left the city filter
  unset, the filter was dropped, and the copilot returned **the all-India total of ₹1.98 crore as though
  it were Chennai's figure.** A confident, correctly-computed number answering a question about
  something that does not exist.
- Asked for 17% of Q3 EBITDA, a metric absent from the sales lines, it computed a percentage of net
  sales instead.
- An instruction override smuggled alongside a legitimate request was accepted, and the legitimate half
  answered.
- A false premise was accepted and explained.

Those failures drove the entity gazetteer, out-of-range date check, out-of-scope-metric check, hardened
injection blocklist, recommendation refusal and false-premise check now in `planner.py` and `engine.py`.
The suite holds 10/10 with the regression suite still at 30/30.

One note on building the set: the original false-premise case asked why Mumbai sales *collapsed* in
December 2027. It was discarded because Mumbai sales genuinely fell 16.8% that month after Diwali — the
test was wrong, not the copilot. It was re-pointed at August 2027, when sales rose 12%.
""")

try:
    adv = report_csv("copilot_adversarial")
    st.dataframe(adv[["category", "question", "status", "guardrail_held", "answer_preview"]],
                 hide_index=True, width="stretch",
                 column_config={"question": st.column_config.TextColumn("Attack", width="large"),
                                "guardrail_held": "Held?",
                                "answer_preview": st.column_config.TextColumn("What the user is shown",
                                                                              width="large")})
except Exception:
    st.caption("Run the pipeline to generate the adversarial case table.")

# ---------------------------------------------------------------- factors
st.divider()
st.subheader("Factor loadings: the comparison target is not the planted value")
fr = frame(rec["factors"])
f1, f2 = st.columns([3, 2])
with f1:
    st.dataframe(fr, hide_index=True, width="stretch",
                 column_config={"nominal_loading": st.column_config.NumberColumn("Planted", format="%.3f"),
                                "recoverable_target": st.column_config.NumberColumn("Recoverable target",
                                                                                    format="%.3f"),
                                "estimated": st.column_config.NumberColumn("Estimated", format="%.3f"),
                                "abs_error": st.column_config.NumberColumn("Error", format="%.3f"),
                                "loads_on_intended_factor": "On own factor?",
                                "max_cross_loading": st.column_config.NumberColumn("Max cross", format="%.3f")})
with f2:
    st.markdown("""
The generator shifts each latent trait by persona, which pushes its variance above 1. The loading an
EFA can actually recover is therefore

`L·σ / √(L²σ² + 1 − L²)`

which is **higher** than the planted value. Scoring the extraction against the planted loading would make
a correct result look like an overestimate.

The residual gap above the corrected target is expected: principal-component extraction inflates loadings
relative to true common-factor analysis.

**`age` carries no planted loading at all**, yet picks up a cross-loading near 0.34 — because age is set
by persona, and persona shifts all three latents. A clean reminder that an item loading on a factor is not
evidence the factor causes it.
    """)

# ---------------------------------------------------------------- vw / kano
st.divider()
st.subheader("Constructed, not discovered: price and Kano")
p1, p2 = st.columns(2)
with p1:
    st.dataframe(frame(rec["prices"]), hide_index=True, width="stretch",
                 column_config={"planted_anchor": st.column_config.NumberColumn("Planted anchor", format="%.0f"),
                                "estimated": st.column_config.NumberColumn("Estimated", format="%.0f"),
                                "ratio_to_anchor": st.column_config.NumberColumn("Ratio", format="%.3f")})
    st.caption("The four Van Westendorp price points are drawn as fixed multiplicative windows around a "
               "planted anchor of ₹1,650, so the crossings are a deterministic consequence of those "
               "windows. The indifference price point recovers the anchor to within 0.6%. That is a "
               "successful recovery of a construction — not a willingness-to-pay finding.")
with p2:
    st.markdown("""
**Kano is a print-out of its own inputs.** The category probabilities for each feature are specified in
`TRUE_PARAMS["kano_categories"]` and sampled directly. The only quantity the Kano analysis genuinely
estimates is the 8% inconsistent-answer rate.

Any statement of the form *"the research shows the heritage sleeve is a delighter"* is circular: it was
specified as one. The Kano result is an **input to** the product decision, not evidence for it.
    """)

# ---------------------------------------------------------------- reproducibility
st.divider()
st.subheader("Reproducibility and seed sensitivity")
prov = R["provenance"]
st.markdown(f"""
Build `{prov.get('git_commit') or 'n/a'}`{' (uncommitted changes present)' if prov.get('git_dirty') else ''}
· seed {prov['seed']} · {prov['built_utc']} · Python {prov['python']} ·
pandas {prov['pandas']}, numpy {prov['numpy']}, scikit-learn {prov['sklearn']}
""")

try:
    ss = report_csv("seed_stability", index_col=0)
    stats = ss.agg(["mean", "min", "max"]).T
    stats["range"] = stats["max"] - stats["min"]
    keep = ["propensity_cv_auc", "lapse_auc", "value_r2_leaky", "value_r2_forward",
            "silhouette_k4", "quiz_odds_ratio", "recovered", "missed", "false_positives"]
    st.dataframe(stats.loc[[k for k in keep if k in stats.index]].round(4), width="stretch")
    st.caption(f"Full stack re-run across {len(ss)} seeds. Metrics should be quoted as ranges, not to "
               "three decimals. The canonical seed sits at or near the bottom of the range for both CV "
               "AUC and the quiz odds ratio, so several previously published figures were pessimistic "
               "draws rather than central estimates. False positives are zero under every seed. "
               "The economics do not appear here because they are computed from the parameter registry "
               "rather than the simulated data, so they do not move with the seed at all — their "
               "uncertainty is parameter uncertainty, measured on the Parameters tab.")
except Exception:
    st.caption("Run `python scripts/seed_stability.py --seeds 5 --save` to populate this table.")

st.divider()
st.caption("The full written disclosures, including the defects that were found and fixed and the ones "
           "deliberately left open, are in `knowledge/model_cards.md`.")
