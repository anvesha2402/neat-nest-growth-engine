import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import fig_layout, frame, header, note, page_setup, report_csv, results  # noqa: E402
from neatnest.config import SERIES  # noqa: E402

page_setup("Retention & CLV")
R = results()
L, V, S = R["lapse"], R["value"], R["segments"]
header("Layer 4 · Retention and customer value", "Keeping owners active after the sale",
       f"FlexFold is bought once every few years, so monthly churn is the wrong lens. A registered owner "
       f"(QR warranty sign-up, n={L['n']:,}) has **lapsed** when, in months 7-12, they buy no accessory, make no "
       f"referral and do not use the care hub. Features are a month-6 snapshot, so the model cannot see the future. "
       f"Base lapse rate: {L['base_rate'] * 100:.1f}%.")

met = frame(L["metrics"])
k = st.columns(4)
k[0].metric("Best model", L["best"].split(" (")[0], help="Chosen on holdout AUC")
k[1].metric("Holdout AUC", f"{met.loc[L['best'], 'auc']:.3f}", help=f"5-fold CV AUC {met.loc[L['best'], 'cv_auc']:.3f}")
k[2].metric("Recall on lapsers", f"{met.loc[L['best'], 'recall'] * 100:.0f}%")
lift = frame(L["lift"])
k[3].metric("Lapsers in riskiest 30%", f"{lift['cum_capture'].iloc[2] * 100:.0f}%")
st.dataframe(met.style.format("{:.3f}"), width="stretch")
note("The Random Forest does not beat the logistic model. When a simpler, explainable model performs as well, it is the "
     "one deployed, so customer-care teams can see why a customer was flagged.")

imp = pd.Series(L["importance"]).head(8)[::-1]
a, b = st.columns(2)
fi = go.Figure(go.Bar(x=imp.values, y=[i.replace("_", " ") for i in imp.index], orientation="h", marker_color=SERIES[0],
                      hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
fi.update_xaxes(title="Permutation importance (AUC drop)")
a.plotly_chart(fig_layout(fi, "What predicts lapse", 380), theme=None)
odds = pd.Series(L["odds_ratios"])
odds = odds[[i for i in odds.index if not i.startswith(("city_", "first_"))]].sort_values()
fo = go.Figure(go.Bar(x=odds.values, y=[i.replace("_", " ") for i in odds.index], orientation="h",
                      marker_color=[SERIES[1] if v > 1 else SERIES[2] for v in odds.values],
                      hovertemplate="%{y}: OR %{x:.2f} per SD<extra></extra>"))
fo.add_vline(x=1, line={"dash": "dash", "color": "#6b665f"})
fo.update_xaxes(title="Odds ratio per 1 SD (logistic)")
b.plotly_chart(fig_layout(fo, "Direction of effect", 380), theme=None)

st.subheader("Owner segments (K-Means, k=4)")
prof = frame(S["profile"])
fs = go.Figure()
for i, (name, r) in enumerate(prof.iterrows()):
    fs.add_scatter(x=[r["lapse_rate"] * 100], y=[r["value_12m"]], mode="markers+text", name=name, text=[name],
                   textposition="top center", marker={"size": max(r["share"] * 140, 14), "color": SERIES[i]},
                   hovertemplate=f"{name}<br>{r['share'] * 100:.0f}% of owners<br>lapse %{{x:.0f}}%<br>value Rs %{{y:,.0f}}<extra></extra>")
fs.update_xaxes(title="Lapse rate %", range=[0, 80]); fs.update_yaxes(title="Mean 12-month value (Rs)")
st.plotly_chart(fig_layout(fs, "Lapse vs value by segment", 400), theme=None)
st.dataframe(prof.style.format("{:,.2f}"), width="stretch")
sil = S["silhouette"]
st.caption(f"Silhouette for k=4 is {sil['4']:.2f} (k=2 scores {sil['2']:.2f}). The segments overlap, so they are "
           "planning groups for retention plays, not hard customer types.")

st.subheader("Retention scenario")
risk = report_csv("risk_table")
c = st.columns(3)
share = c[0].slider("Contact the riskiest share of owners", 0.1, 1.0, 0.3, 0.05)
save = c[1].slider("Save rate among contacted lapsers (assumption)", 0.05, 0.5, 0.2, 0.05)
margin = c[2].number_input("Accessory + referral margin per saved owner (Rs)", 200, 3000, 900, 50)
top = risk.head(int(len(risk) * share))
reach = top["lapse_risk"].sum()
lapsers_total = risk["lapse_risk"].sum()
m = st.columns(3)
m[0].metric("Owners contacted", f"{len(top):,}")
m[1].metric("Expected lapsers reached", f"{reach:,.0f}", help=f"{reach / lapsers_total * 100:.0f}% of expected lapsers")
m[2].metric("Margin protected", f"Rs {reach * save * margin / 1e5:,.2f} lakh")
st.caption("Uses each owner's predicted risk as the expected chance of lapsing. Validate the save rate with a random holdout before scaling.")

st.subheader("12-month customer value (regression)")
vm = frame(V["metrics"])
st.dataframe(vm.style.format("{:,.3f}"), width="stretch")
st.caption(f"R-squared is high partly by construction: value includes the first order and early accessory spend. "
           f"The useful output is the ranking of drivers. Max VIF after removing the duplicate accessory-orders "
           f"feature: {V['vif_max']:.1f}.")
st.dataframe(frame(V["coefficients"]).style.format("{:,.3f}"), width="stretch")

st.subheader("Highest-risk owners")
seg = st.multiselect("Segment", sorted(risk["segment"].unique()), default=[])
view = risk[risk["segment"].isin(seg)] if seg else risk
st.dataframe(view.head(50), hide_index=True, width="stretch")
