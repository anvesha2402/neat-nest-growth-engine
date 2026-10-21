import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import fig_layout, header, note, page_setup  # noqa: E402
from neatnest.config import PARAMS, SERIES, kind_counts, params_frame  # noqa: E402
from neatnest.market import Scenario  # noqa: E402
from neatnest import opex  # noqa: E402

page_setup("Parameters")

header("Switchboard", "Parameters and sensitivity",
       "Every financial input in this project is a declared parameter with a provenance tag, a source "
       "and a plausible range. Move the six that matter and watch the decision engine respond.")

pf = params_frame()
kinds = kind_counts()

note("These are <b>parameterised benchmark inputs</b>, not measurements of a real business. "
     f"Of {len(pf)} registered inputs, only <b>{kinds.get('SOURCED', 0)}</b> trace to a named published "
     f"source; <b>{kinds.get('BENCHMARK', 0)}</b> are benchmarks from comparable products and rate cards, "
     f"and <b>{kinds.get('ASSUMED', 0)}</b> are assumptions with no external basis. The assumptions are "
     "the work programme: each one is something a vendor quote or a Phase-0 test would replace.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registered inputs", len(pf))
k2.metric("SOURCED", kinds.get("SOURCED", 0), help="Traceable to a named published source")
k3.metric("BENCHMARK", kinds.get("BENCHMARK", 0), help="From comparable products, categories or rate cards")
k4.metric("ASSUMED", kinds.get("ASSUMED", 0), help="No external basis — replace these first")

# ------------------------------------------------------------------ sliders
st.divider()
st.subheader("Live scenario")
st.caption("Each slider moves one group of inputs as a multiple of its registered value. The ranges come "
           "from each parameter's declared plausible range in the registry, not from the interface.")

LABELS = {"cogs": "COGS", "cac": "D2C CAC", "returns": "Return rate",
          "attach": "Accessory attach rate", "acos": "Amazon ACoS", "opex": "Fixed OpEx"}

cols = st.columns(3)
vals = {}
for i, name in enumerate(["cogs", "cac", "returns", "attach", "acos", "opex"]):
    lo, hi = opex.slider_range(name)
    group, key = opex.SLIDER_SOURCES[name]
    p = PARAMS[group][key]
    with cols[i % 3]:
        vals[f"{name}_mult"] = st.slider(
            f"{LABELS[name]} (×)", min_value=float(min(lo, 1.0)), max_value=float(max(hi, 1.0)),
            value=1.0, step=0.01,
            help=f"Anchored on {key} = {p.value:,.4g} {p.unit} [{p.kind}]. "
                 f"Declared range {p.low:,.4g} to {p.high:,.4g}. Drives: {p.drives}")

sc = Scenario(**vals)
bridge = opex.ebitda_bridge(sc).set_index("year")
base = opex.ebitda_bridge().set_index("year")

if sc.label() != "baseline":
    st.caption(f"Scenario: **{sc.label()}** · reset every slider to 1.00 for the registered baseline.")

# ------------------------------------------------------------------ bridge
st.subheader("EBITDA bridge")
m = st.columns(4)
for i, y in enumerate(bridge.index):
    delta = bridge.loc[y, "ebitda_cr"] - base.loc[y, "ebitda_cr"]
    m[i].metric(f"{y} EBITDA", f"₹{bridge.loc[y, 'ebitda_cr']:.2f} cr",
                delta=(f"{delta:+.2f} vs baseline" if abs(delta) > 5e-3 else None))
first = next((y for y in bridge.index if bridge.loc[y, "ebitda_positive"]), None)
m[3].metric("EBITDA positive from", first or "never",
            help="On the registered baseline this is Y2. Year 1 is negative, which is the normal shape "
                 "for a D2C launch carrying tooling and a team against 12,000 units.")

wf_year = st.radio("Waterfall year", list(bridge.index), horizontal=True, index=2)
r = bridge.loc[wf_year]
steps = [("Net revenue", r["net_revenue_cr"]),
         ("CM2 (after COGS, pack, ship, fees, returns)", r["cm2_cr"] - r["net_revenue_cr"]),
         ("Performance marketing", -r["performance_marketing_cr"]),
         ("Brand & content", -r["fixed_marketing_cr"]),
         ("Fixed operating costs", -r["fixed_opex_cr"])]
wf = go.Figure(go.Waterfall(
    orientation="v", measure=["absolute"] + ["relative"] * (len(steps) - 1) + ["total"],
    x=[s[0] for s in steps] + ["EBITDA"], y=[s[1] for s in steps] + [None],
    text=[f"{v:+.2f}" if i else f"{v:.2f}" for i, (_, v) in enumerate(steps)]
         + [f"{r['ebitda_cr']:.2f}"],
    textposition="outside",
    decreasing={"marker": {"color": SERIES[1]}}, increasing={"marker": {"color": SERIES[2]}},
    totals={"marker": {"color": SERIES[2] if r["ebitda_cr"] >= 0 else SERIES[4]}}))
st.plotly_chart(fig_layout(wf, f"{wf_year}: net revenue to EBITDA (₹ crore)", 400), theme=None)

st.dataframe(bridge[["core_units", "new_d2c_customers", "net_revenue_cr", "cm2_cr", "cm3_cr",
                     "fixed_opex_cr", "ebitda_cr", "ebitda_pct", "cumulative_ebitda_cr",
                     "working_capital_exposure_cr"]].round(3), width="stretch",
             column_config={"core_units": "Core units", "new_d2c_customers": "New D2C customers",
                            "net_revenue_cr": "Net revenue ₹cr", "cm2_cr": "CM2 ₹cr", "cm3_cr": "CM3 ₹cr",
                            "fixed_opex_cr": "Fixed OpEx ₹cr", "ebitda_cr": "EBITDA ₹cr",
                            "ebitda_pct": st.column_config.NumberColumn("EBITDA %", format="%.1%"),
                            "cumulative_ebitda_cr": "Cumulative ₹cr",
                            "working_capital_exposure_cr": "Working capital ₹cr"})

st.markdown("""
**CM3 is not profit.** It is what remains after the costs that scale with a unit sold. An earlier version
of this model stopped there, which invited the reader to treat it as earnings — and on this plan that
reading is wrong by roughly the cost of the company. The bridge stops at EBITDA: no interest, tax, or
depreciation beyond tooling.

**Working capital is reported, not modelled.** It is inventory days multiplied by the year's goods cost —
money that has to exist before the revenue arrives. At Year 3 that exposure exceeds Year 2 EBITDA, so
this is not a cash runway, and for a product that must fund inventory before selling any of it, this is
the largest unmodelled risk in the project.
""")

# ------------------------------------------------------------------ break-even
st.divider()
st.subheader("Break-even")
be = opex.break_even_table(sc)
st.dataframe(be[["cm3_per_core_unit", "fixed_opex", "break_even_core_units", "planned_core_units",
                 "coverage_ratio"]].round(2), width="stretch",
             column_config={"cm3_per_core_unit": "CM3 per core unit ₹",
                            "fixed_opex": "Fixed OpEx ₹", "break_even_core_units": "Break-even units",
                            "planned_core_units": "Planned units",
                            "coverage_ratio": st.column_config.NumberColumn("Coverage", format="%.2f")})
y1 = be.loc["Y1"]
st.caption(f"Year 1 needs about {y1['break_even_core_units']:,.0f} core units to cover fixed overhead "
           f"against a plan of {y1['planned_core_units']:,.0f} — a coverage ratio of "
           f"{y1['coverage_ratio']:.2f}. The Year-1 plan is structurally short of break-even, and saying "
           f"so is a stronger claim than a plan that looks profitable from month one.")

# ------------------------------------------------------------------ tornado
st.divider()
st.subheader("What actually moves the answer")
tor = opex.tornado("Y3", sc)
fig = go.Figure()
b0 = float(tor["base_ebitda_cr"].iloc[0])
fig.add_bar(y=[LABELS[p] for p in tor["parameter"]],
            x=[lo - b0 for lo in tor["ebitda_at_low_cr"]], orientation="h",
            name="At the low end of its range", marker={"color": SERIES[2]})
fig.add_bar(y=[LABELS[p] for p in tor["parameter"]],
            x=[hi - b0 for hi in tor["ebitda_at_high_cr"]], orientation="h",
            name="At the high end", marker={"color": SERIES[1]})
fig.update_layout(barmode="overlay", xaxis_title="Change in Y3 EBITDA vs this scenario (₹ crore)",
                  yaxis={"autorange": "reversed"})
st.plotly_chart(fig_layout(fig, f"One-at-a-time sweep over each parameter's declared range "
                                f"(Y3 baseline ₹{b0:.2f} cr)", 360), theme=None)

st.dataframe(tor[["parameter", "backing_param", "kind", "mult_low", "mult_high", "ebitda_at_low_cr",
                  "ebitda_at_high_cr", "swing_cr", "flips_negative"]].round(3), hide_index=True,
             width="stretch",
             column_config={"parameter": "Lever", "backing_param": "Anchored on", "kind": "Provenance",
                            "swing_cr": "Swing ₹cr", "flips_negative": "Flips negative?"})

top = tor.iloc[0]
st.markdown(f"""
**The largest single lever on Year-3 EBITDA is {LABELS[top['parameter']]}**, with a swing of
₹{top['swing_cr']:.2f} crore across its declared range — and it is anchored on `{top['backing_param']}`,
an **{top['kind']}** input with no vendor quotation behind it. The most consequential number in the model
is also one of the least grounded. That is the honest headline of this page.

This is a **one-at-a-time** sweep, so it shows exposure, not a joint probability. No single parameter
drives Year 3 negative within its declared range; the risk is two of them moving together.
""")

st.markdown("##### Where each lever has to go to wipe out Year-3 EBITDA")
bp = opex.breaking_points("Y3", sc)
st.dataframe(bp, hide_index=True, width="stretch",
             column_config={"parameter": "Lever", "backing_param": "Anchored on", "kind": "Provenance",
                            "breaks_even_at_mult": st.column_config.NumberColumn("Breaks even at (×)",
                                                                                 format="%.2f"),
                            "implied_value": "Implied value",
                            "within_declared_range": "Inside its declared range?",
                            "declared_low": "Range low", "declared_high": "Range high"})
st.caption("A blank means EBITDA does not cross zero anywhere in the search range, which is itself the "
           "useful answer for a parameter the plan is robust to.")

# ------------------------------------------------------------------ registry
st.divider()
st.subheader("The full registry")
f1, f2 = st.columns([1, 2])
kind_pick = f1.multiselect("Provenance", ["SOURCED", "BENCHMARK", "ASSUMED"],
                           default=["SOURCED", "BENCHMARK", "ASSUMED"])
group_pick = f2.multiselect("Group", sorted(pf["group"].unique()), default=sorted(pf["group"].unique()))
view = pf[pf["kind"].isin(kind_pick) & pf["group"].isin(group_pick)]
st.dataframe(view, hide_index=True, width="stretch",
             column_config={"group": "Group", "parameter": "Parameter", "value": "Value", "unit": "Unit",
                            "kind": "Provenance", "low": "Range low", "high": "Range high",
                            "range_width_vs_value": st.column_config.NumberColumn("Range width ÷ value",
                                                                                  format="%.2f"),
                            "drives": st.column_config.TextColumn("What it drives", width="medium"),
                            "source": st.column_config.TextColumn("Source / basis", width="large")})
st.caption("`config.check_registry()` asserts that every value above still equals the value the code "
           "actually uses, and the build fails if any has drifted. It runs in the test suite, so this "
           "table cannot quietly go stale.")
