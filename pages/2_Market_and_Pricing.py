import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest import research  # noqa: E402
from neatnest.app_utils import asset, data, fig_layout, frame, header, note, page_setup, results  # noqa: E402
from neatnest.config import PHASES, SERIES, SKUS, SOURCES  # noqa: E402
from neatnest.market import unit_economics  # noqa: E402

page_setup("Market & Pricing")
R = results()
m = R["market"]
header("Layer 1 · Market, pricing and entry", "Sizing the market, then pricing it",
       "The market is sized from washing-machine ownership: households that carry laundry to a machine are the ones "
       "with a transport problem. Every input is either a cited source or a labelled estimate.")
st.image(asset("diagrams/layer_strip_1.png"), width="stretch")

k = st.columns(4)
k[0].metric("TAM, urban laundry storage (Rs cr)", f"{m['tam_top_down_cr']:,.0f}-{m['tam_bottom_up_cr']:,.0f}",
            help="Top-down (global share) vs bottom-up (households x ownership x replacement)")
k[1].metric("Urban washing-machine households", f"{m['urban_wm_hh_m']:.1f} M",
            help=f"Derived: 60 M owners nationally, urban 4x rural; urban ownership {m['urban_wm_ownership'] * 100:.0f}%")
k[2].metric("SAM households", f"{m['sam_hh_m']:.1f} M", help="20% reachable online at a Rs 1,500+ price point (estimate)")
k[3].metric("SAM annual value", f"Rs {m['sam_value_cr']:,.0f} cr", help=f"{m['sam_units_per_year_m']:.2f} M units/yr at blended ASP")

st.subheader("Price acceptance (Van Westendorp, n=600)")
vw = research.van_westendorp(data("survey"))
c = vw["curves"]
fig = go.Figure()
for col, colr, dash in [("Too cheap", SERIES[2], "dash"), ("Cheap", SERIES[2], "solid"),
                        ("Expensive", SERIES[1], "solid"), ("Too expensive", SERIES[1], "dash")]:
    fig.add_scatter(x=c["price"], y=c[col] * 100, name=col, line={"color": colr, "dash": dash, "width": 2},
                    hovertemplate="Rs %{x:,}: %{y:.0f}%<extra>" + col + "</extra>")
fig.add_vrect(x0=vw["PMC"], x1=vw["PME"], fillcolor="#D5B893", opacity=0.3, line_width=0)
fig.add_vline(x=1799, line={"color": SERIES[0], "width": 2}, annotation_text="FlexFold Rs 1,799")
fig.update_xaxes(range=[500, 3200], title="Price (Rs)"); fig.update_yaxes(title="% of respondents")
st.plotly_chart(fig_layout(fig, f"Acceptable range Rs {vw['PMC']:,.0f}-{vw['PME']:,.0f} · OPP Rs {vw['OPP']:,.0f} · IPP Rs {vw['IPP']:,.0f}"), theme=None)
note("Rs 1,799 sits inside the acceptable range but above the indifference point. The launch therefore leads with "
     "total cost of ownership and the three-year warranty rather than discounts. Heritage Edition (Rs 2,299) sits above "
     "the hero range by design; the SMART objective is an A/B test against a plain sleeve.")

st.subheader("Unit economics calculator")
a, b, c2, d = st.columns(4)
sku = a.selectbox("SKU", list(SKUS), index=1, format_func=lambda s: SKUS[s]["name"])
ch = b.selectbox("Channel", ["D2C", "Amazon", "B2B"])
disc = c2.slider("Discount off MRP", 0, 30, 0, 1) / 100
cac = d.slider("Acquisition cost per unit (Rs)", 0, 1200, 0 if ch == "B2B" else 520, 20)
ue = unit_economics(sku, ch, cac=cac, discount=disc)
steps = [("Price paid", ue["price"]), ("GST", -ue["gst"]), ("COGS", -ue["cogs"]), ("Packaging", -ue["packaging"]),
         ("Shipping", -ue["shipping"]), ("Channel fees", -ue["channel_fees"]), ("Returns", -ue["returns"]),
         ("Acquisition", -ue["cac"])]
wf = go.Figure(go.Waterfall(x=[s for s, _ in steps] + ["CM3"], y=[v for _, v in steps] + [0],
                            measure=["absolute"] + ["relative"] * (len(steps) - 1) + ["total"],
                            connector={"line": {"color": "#e6e1d8"}},
                            increasing={"marker": {"color": SERIES[0]}}, decreasing={"marker": {"color": SERIES[1]}},
                            totals={"marker": {"color": SERIES[2] if ue["cm3"] >= 0 else SERIES[4]}},
                            text=[f"{v:,.0f}" for _, v in steps] + [f"{ue['cm3']:,.0f}"], textposition="outside"))
st.plotly_chart(fig_layout(wf, f"CM2 Rs {ue['cm2']:,.0f} ({ue['cm2_pct'] * 100:.0f}% of net) · CM3 Rs {ue['cm3']:,.0f} ({ue['cm3_pct'] * 100:.0f}%)", 360), theme=None)
if ue["price"] > 2500:
    st.warning("Above Rs 2,500 per piece, textile made-ups attract 18% GST instead of 5%. Keep every SKU at or below Rs 2,499.")

st.subheader("Three-year plan")
som = frame(R["som_plan"])
show = som[["core_units", "accessory_units", "net_revenue_cr", "cm2_cr", "marketing_cr", "cm3_cr", "cm3_pct", "share_of_sam_units_pct"]].copy()
show.columns = ["Core units", "Accessory units", "Net revenue (Rs cr)", "CM2 (Rs cr)", "Marketing (Rs cr)",
                "CM3 (Rs cr)", "CM3 %", "Share of SAM units %"]
show["CM3 %"] = show["CM3 %"] * 100
st.dataframe(show.style.format("{:,.2f}"), width="stretch")
st.caption("CM3 is contribution after variable marketing and is shown before team, rent and tooling overheads.")

st.subheader("Market entry")
for name, when, what in PHASES:
    st.markdown(f"**{name}** · {when}  \n{what}")

with st.expander("Sources and labelled estimates"):
    for kx, v in SOURCES.items():
        st.markdown(f"- **{kx}**: {v}")
    st.markdown("- **EST** values (urban household size, basket ownership and life, reachable share, unit costs, "
                "payment-gateway fee, returns, CAC) are assumptions to validate with Phase-0 research and vendor quotes.")
