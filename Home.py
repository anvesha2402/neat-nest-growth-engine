"""Neat Nest Growth Engine: Streamlit entry point (Home)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from neatnest.app_utils import asset, card, header, note, page_setup, results  # noqa: E402

page_setup("Home")
R = results()
m, som = R["market"], R["som_plan"]

c1, c2 = st.columns([1.35, 1])
with c1:
    header("Integrated capstone · Brand to AI",
           "Neat Nest Growth Engine",
           "**FlexFold** is a fold-flat, wheeled, three-compartment laundry system for India's space-constrained "
           "apartments. This app follows one brand from consumer insight to an AI copilot. The same simulated "
           "launch-year data runs through every layer, so the numbers on one page explain the numbers on the next.")
    st.markdown("**Tidy Starts Here.**")
with c2:
    st.image(asset("product_hallway.jpg"), width="stretch")

k = st.columns(4)
k[0].metric("Serviceable households (SAM)", f"{m['sam_hh_m']:.1f} M", help="Urban, washing-machine owning, reachable online at Rs 1,500+")
k[1].metric("Hero price (GST-incl.)", "Rs 1,799", help="Inside the Van Westendorp acceptable range")
k[2].metric("Year-3 net revenue plan", f"Rs {som['Y3']['net_revenue_cr']:.1f} cr",
            help=f"{int(som['Y3']['core_units']):,} core units, ~{som['Y3']['share_of_sam_units_pct']:.1f}% of annual SAM demand")
k[3].metric("Copilot regression suite", R["copilot_eval"]["summary"]["total"], help="Offline mode, fixed question set")

st.subheader("Five layers, one business")
cols = st.columns(5)
layers = [
    ("1 · Brand & Product", "Problem, market, STP, JTBD, Kano, concept selection, CBBE, packaging, RACE."),
    ("2 · Customer Factors", f"Survey n={R['survey_n']}: three latent dimensions, Van Westendorp, Kano."),
    ("3 · Acquisition", f"Launch campaign n={R['propensity']['n']:,}: who buys, and where to spend."),
    ("4 · Retention & Value", f"{R['lapse']['n']:,} owners: lapse risk, 12-month value, segments."),
    ("5 · Nest Copilot", "Guarded agent + RAG: Python computes, the LLM only plans and explains."),
]
for col, (t, b) in zip(cols, layers):
    with col:
        card(t, b)

st.write("")
st.subheader("How the layers hand off")
st.markdown(
    "- **Brand → Factors:** the STP personas (Anjali, students, young couples, seniors) are tested against "
    "latent survey dimensions, not assumed.\n"
    "- **Factors → Acquisition:** *Digital Engagement* and *prior home-organisation spend* become propensity "
    "features, and the RACE *Will it fit?* quiz turns out to be the strongest single conversion signal.\n"
    "- **Acquisition → Retention:** for a durable product, churn is redefined as *lapse*: no accessory, referral "
    "or care-hub activity in months 7-12. That is the CBBE *Resonance* layer, made measurable.\n"
    "- **Everything → Copilot:** brand book, PRD, KPI rules and GTM playbook are the retrieval corpus, and the "
    "propensity, lapse and segment outputs are callable tools.")
note("All business data in this app is a <b>seeded simulation</b> of Neat Nest's launch year (Apr 2027 to Mar 2028), "
     "built to mirror patterns from primary interviews and published benchmarks. Market inputs are cited on the "
     "Market & Pricing page. Use the sidebar to move between layers.")
