import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest import research  # noqa: E402
from neatnest.app_utils import data, fig_layout, header, note, page_setup  # noqa: E402
from neatnest.config import SERIES  # noqa: E402

page_setup("Customer Factors")
header("Layer 2 · Customer understanding (factor analysis)", "Three dimensions behind ten survey answers",
       "Phase-0 survey of 600 launch-audience respondents. Nine behavioural and attitudinal items (plus age) are "
       "reduced to latent factors, which are then used to test whether the STP personas really differ.")

sv = data("survey")
fa = research.factor_analysis(sv)
k = st.columns(4)
k[0].metric("KMO (sampling adequacy)", f"{fa['kmo']:.2f}", help="> 0.7 is good")
k[1].metric("Bartlett chi-square", f"{fa['bartlett_chi2']:,.0f}", help=f"df={fa['bartlett_dof']}, p < 0.001")
k[2].metric("Factors retained", fa["n_factors"], help="Eigenvalue > 1 and scree elbow agree")
k[3].metric("Variance explained", f"{fa['cumulative_pct']:.1f}%")
note("This fixes the original coursework's weakness: with two items per factor, KMO was stuck at 0.50. "
     "Designing the survey with three items per construct lifts it to the 'good' band.")

a, b = st.columns([1, 1.3])
ev = fa["eigenvalues"]
f1 = go.Figure(go.Scatter(x=list(range(1, len(ev) + 1)), y=ev, mode="lines+markers", line={"color": SERIES[0], "width": 2},
                          marker={"size": 9}, hovertemplate="Factor %{x}: %{y:.2f}<extra></extra>"))
f1.add_hline(y=1, line={"color": SERIES[1], "dash": "dash"})
f1.update_xaxes(title="Factor", dtick=1); f1.update_yaxes(title="Eigenvalue")
a.plotly_chart(fig_layout(f1, "Scree plot", 360), theme=None)
L = fa["loadings"]
f2 = go.Figure(go.Heatmap(z=L.values, x=L.columns, y=[i.replace("_", " ") for i in L.index], colorscale="Blues",
                          zmin=0, zmax=1, text=np.round(L.values, 2), texttemplate="%{text}", showscale=False,
                          hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>"))
f2.update_yaxes(autorange="reversed")
b.plotly_chart(fig_layout(f2, "Varimax-rotated loadings", 360), theme=None)

st.markdown("""
| Factor | Items (loading > 0.8) | Marketing use |
|---|---|---|
| **Digital Engagement** | website visits, social interactions, email opens | Paid social and creator reach; propensity feature |
| **Spending Power** | income, home-organisation spend, spending score | Price ladder (Lite vs hero vs Heritage) |
| **Loyalty & Promo Response** | brand loyalty, promo response, referral likelihood | Referral programme and accessory replenishment |

Age loads weakly everywhere (communality %.2f). It describes *who* someone is, not *how* they behave, so it is left out of targeting.
""" % fa["communality"]["age"])

st.subheader("Do the personas differ?")
prof = research.persona_profiles(sv, fa["scores"])
f3 = go.Figure()
for i, c in enumerate(["Digital Engagement", "Spending Power", "Loyalty & Promo Response"]):
    f3.add_bar(x=[p.split(" (")[0] for p in prof.index], y=prof[c], name=c, marker_color=SERIES[i],
               hovertemplate="%{x}: %{y:.2f} SD<extra>" + c + "</extra>")
f3.update_yaxes(title="Mean factor score (SD)")
st.plotly_chart(fig_layout(f3, "Persona factor profiles", 380), theme=None)
note("Students are digitally reachable but cash-poor, which is the case for FlexFold Lite at Rs 899. Seniors are hard to "
     "reach online but loyal and referral-prone, which suits housing-society activations. Anjali's edge is spending "
     "power, the right audience for the hero product.")

st.subheader("Kano feature classification")
kano = research.kano(sv)
kano_show = kano[["feature", "category", "better", "worse", "Must-be", "Performance", "Attractive (delighter)", "Indifferent"]]
st.dataframe(kano_show, hide_index=True, width="stretch")
st.caption("Better = (A+O)/(A+O+M+I); Worse = -(O+M)/(A+O+M+I). Reverse and questionable answers are excluded.")
