import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import fig_layout, frame, header, model, note, page_setup, results  # noqa: E402
from neatnest.config import SERIES  # noqa: E402
from neatnest.propensity import score_prospect  # noqa: E402

page_setup("Campaign Propensity")
R = results()
P = R["propensity"]
header("Layer 3 · Acquisition (binary logistic regression)", "Who buys after the launch campaign?",
       f"{P['n']:,} prospects reached in Phase 1 (Mumbai and Pune). Base conversion {P['base_rate'] * 100:.1f}%. "
       "statsmodels gives interpretable odds ratios; a scikit-learn pipeline gives an honest out-of-sample score.")

k = st.columns(4)
k[0].metric("Model significance", f"LLR p < 0.001", help=f"McFadden pseudo R-squared {P['pseudo_r2']:.3f}")
k[1].metric("5-fold CV AUC", f"{P['cv_auc_mean']:.3f}", help=f"SD {P['cv_auc_std']:.3f}")
k[2].metric("Holdout AUC", f"{P['holdout']['auc']:.3f}", help=f"n = {P['holdout']['n_test']:,}")
lift = frame(P["lift"])
k[3].metric("Buyers in top 3 deciles", f"{lift['cum_capture'].iloc[2] * 100:.0f}%")

coef = frame(P["coefficients"])
coef = coef.sort_values("odds_ratio")
f = go.Figure()
for sig, color, name in [(True, SERIES[0], "p < 0.05"), (False, "#b9b3a8", "not significant")]:
    sub = coef[coef["significant"] == sig]
    f.add_scatter(x=sub["odds_ratio"], y=[i.replace("_", " ") for i in sub.index], mode="markers", name=name,
                  marker={"color": color, "size": 11},
                  error_x={"type": "data", "symmetric": False, "array": sub["ci_high"] - sub["odds_ratio"],
                           "arrayminus": sub["odds_ratio"] - sub["ci_low"], "color": color},
                  hovertemplate="%{y}: OR %{x:.3f}<extra></extra>")
f.add_vline(x=1, line={"dash": "dash", "color": "#6b665f"})
f.update_xaxes(type="log", title="Odds ratio (95% CI)")
a, b = st.columns([1.2, 1])
a.plotly_chart(fig_layout(f, "Purchase drivers", 400), theme=None)
fl = go.Figure(go.Bar(x=lift.index.astype(str), y=lift["conv_rate"] * 100, marker_color=SERIES[0],
                      hovertemplate="Decile %{x}: %{y:.1f}%<extra></extra>"))
fl.add_hline(y=P["base_rate"] * 100, line={"dash": "dash", "color": SERIES[1]}, annotation_text="average")
fl.update_xaxes(title="Propensity decile (1 = highest)"); fl.update_yaxes(title="Conversion %")
b.plotly_chart(fig_layout(fl, "Holdout conversion by decile", 400), theme=None)

sig = coef[coef["significant"]].sort_values("odds_ratio", ascending=False)
lines = []
for v, r in sig.iterrows():
    unit = {"fit_quiz_completed": "completing the 'Will it fit?' quiz", "prev_home_org_purchases": "each previous home-organisation purchase",
            "site_visits": "each site visit", "digital_engagement_score": "each engagement point"}.get(v, v)
    lines.append(f"- **{unit}** multiplies the odds of buying by **{r['odds_ratio']:.2f}** ({(r['odds_ratio'] - 1) * 100:+.0f}%), p = {r['p_value']:.4f}")
st.markdown("**Significant drivers, holding the others constant**\n" + "\n".join(lines))
eng10 = sig.loc["digital_engagement_score", "odds_ratio"] ** 10 if "digital_engagement_score" in sig.index else None
if eng10:
    st.caption(f"A 10-point rise in engagement multiplies the odds by {eng10:.2f}. Income, age and acquisition source are not significant once behaviour is known.")
note("The RACE <i>Act</i>-stage quiz built to answer fit anxiety is the strongest single signal. Because quiz-takers "
     "self-select, this is association, not proof that the quiz causes purchase. A randomised quiz-prompt test in "
     "Phase 2 would settle it.")

st.subheader("Score a prospect")
c = st.columns(4)
age = c[0].number_input("Age", 18, 75, 32)
inc = c[1].number_input("Monthly income (Rs '000)", 10, 600, 90)
eng = c[2].slider("Digital engagement score", 0, 100, 65)
prev = c[3].number_input("Previous home-org purchases", 0, 12, 2)
c2 = st.columns(4)
visits = c2[0].number_input("Site visits", 1, 30, 4)
quiz = c2[1].selectbox("Completed 'Will it fit?' quiz", ["Yes", "No"]) == "Yes"
src = c2[2].selectbox("Source", ["Meta paid social", "Google search", "Creator partnership", "Housing-society activation"])
p = score_prospect(model("propensity"), age=age, monthly_income_k=inc, digital_engagement_score=eng,
                   prev_home_org_purchases=prev, site_visits=visits, fit_quiz_completed=int(quiz), source=src)
c2[3].metric("Propensity score", f"{p * 100:.0f} / 100")
st.caption("The class-balanced model ranks prospects well but inflates raw probabilities, so read the score as a rank, not a calibrated chance.")
h = P["holdout"]
with st.expander("Holdout classification metrics at a 0.5 threshold"):
    st.write(pd.DataFrame([h]).drop(columns=["n_test"]).round(3))
    st.caption("Precision is low at 0.5 because buyers are rare (about 8%). The model is used for ranking and decile targeting, not as a yes/no classifier.")
