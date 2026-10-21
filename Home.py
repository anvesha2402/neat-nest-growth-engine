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
    header("Methods portfolio · Brand to AI",
           "Neat Nest Growth Engine",
           "**FlexFold** is a fold-flat, wheeled, three-compartment laundry system for India's space-constrained "
           "apartments. This app is a RevOps decision framework — brand insight through to a guarded AI copilot — "
           "demonstrated on a seeded simulation whose every effect size was declared before any model saw it.")
    st.markdown("**Tidy Starts Here.**")
with c2:
    st.image(asset("product_hallway.jpg"), width="stretch")

eco = R["economics_summary"]
k = st.columns(4)
k[0].metric("Serviceable households (SAM)", f"{m['sam_hh_m']:.1f} M",
            help="Urban, washing-machine owning, reachable online at Rs 1,500+. Rests on a reachable-share "
                 "assumption of 0.20 that moves SAM proportionally — see the Parameters tab.")
k[1].metric("Year-3 net revenue plan", f"Rs {som['Y3']['net_revenue_cr']:.1f} cr",
            help=f"{int(som['Y3']['core_units']):,} core units, ~{som['Y3']['share_of_sam_units_pct']:.1f}% of "
                 f"annual SAM demand. A plan, not a forecast: no demand model stands behind these volumes.")
k[2].metric("Year-1 EBITDA", f"Rs {eco['ebitda_cr']['Y1']:.2f} cr",
            delta=f"EBITDA positive from {eco['first_ebitda_positive_year']}", delta_color="off",
            help="After fixed overhead — team, rent and 3PL, tooling amortisation, platform, support, "
                 "insurance. Contribution after marketing (CM3) is not profit.")
k[3].metric("Planted effects recovered", f"{R['recovery']['clean_share'] * 100:.0f}%",
            delta=f"{R['recovery']['false_positives']} false positives", delta_color="off",
            help="Recovered or correctly-null, as a share of the planted coefficients that were recoverable "
                 "at all. See the Model Cards tab.")

st.subheader("Five layers, one business")
cols = st.columns(5)
layers = [
    ("1 · Brand & Product", "Problem, market, STP, JTBD, Kano, concept selection, CBBE, packaging, RACE."),
    ("2 · Customer Factors", f"Survey n={R['survey_n']}: three latent dimensions, Van Westendorp, Kano."),
    ("3 · Acquisition", f"Launch campaign n={R['propensity']['n']:,}: who buys, and which planted effects "
                        f"are even detectable at this size."),
    ("4 · Retention & Value", f"{R['lapse']['n']:,} owners: lapse risk, forward value, segments."),
    ("5 · Nest Copilot", "Guarded agent + RAG: Python computes, the LLM only plans and explains."),
]
for col, (t, b) in zip(cols, layers):
    with col:
        card(t, b)

st.write("")
st.subheader("Reference architecture")
st.image(asset("diagrams/architecture_master.png"), width="stretch")
st.caption("Data & knowledge → analytics & ML → AI agent layer → application layer → business outcomes. "
           "Open **🧭 Architecture** in the sidebar for the full diagram and the frameworks/keyword index "
           "behind every layer.")

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
note("<b>There is no real company and no real data.</b> FlexFold is a designed product concept, and all four "
     "datasets are a seeded simulation of a launch year (Apr 2027 to Mar 2028) whose every effect size was "
     "declared in advance. So no number here is a finding about the market — each one is an attempt to recover a "
     "value that was written down first. The <b>Model Cards</b> tab reports how well that worked, including the "
     "effects no model could detect; the <b>Parameters</b> tab shows all 89 inputs with their provenance and "
     "lets you move the six that matter.<br><br>"
     "The four datasets are <b>independent panels of one scenario, not a linked funnel</b>: they share a seed, "
     "not a population, so they cannot be added up or traced as a cohort.")
