import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import asset, card, fig_layout, header, note, page_setup  # noqa: E402
from neatnest.config import SERIES  # noqa: E402

page_setup("Brand & Product")
header("Layer 1 · Brand & product management", "A category nobody engineered",
       "Laundry storage is used several times a week but is priced and built as a disposable commodity. The labour "
       "around it is invisible: India's Time Use Survey 2024 records women spending **289 minutes a day** on unpaid "
       "domestic services against **88 for men**. Because nobody counts that labour, nobody has designed for it.")
st.image(asset("diagrams/layer_strip_1.png"), width="stretch")

st.subheader("Four connected failures")
c = st.columns(4)
for col, (t, b) in zip(c, [
    ("Space", "Sized for the worst week, in the way every other week. Capacity is fixed at purchase."),
    ("Movement", "Handles built for lifting, not hauling through doorways and down stairs."),
    ("Durability", "Joints and handles fail within a year, which trains buyers to buy the cheapest."),
    ("Hygiene", "Sealed shapes trap damp and odour in humid, poorly ventilated apartments."),
]):
    with col:
        card(t, b)

st.subheader("Competitive teardown: nobody solves more than two")
comp = pd.DataFrame({
    "Player": ["Cello / Milton", "Unbranded marketplace", "IKEA India", "Brabantia", "Joseph Joseph Tota", "Umbra Cinch", "Neat Nest FlexFold"],
    "Price (India / intl.)": ["Rs 157-1,401", "Rs 145-258", "Rs 199-2,990", "US$44-265", "US$115", "US$35-50", "Rs 899-2,299"],
    "Space": ["-", "-", "partial", "-", "partial", "-", "yes"],
    "Movement": ["-", "-", "-", "-", "-", "yes", "yes"],
    "Durability": ["-", "-", "partial", "yes", "-", "-", "yes"],
    "Hygiene": ["-", "-", "-", "-", "partial", "-", "yes"],
})
st.dataframe(comp, hide_index=True, width="stretch")
note("IKEA India's laundry range has no wheeled product, and the only wheeled options on Indian marketplaces are 9 kg metal carts "
     "selling at about Rs 3,600. The Rs 1,200-2,500 wheeled, fold-flat space is empty.")

st.subheader("Concept selection: try your own weights")
st.caption("The original matrix scored four finalist concepts on five weighted criteria. Move the weights to test how robust the choice is.")
crit = ["Solves all 4 dimensions", "Feasibility at target price", "Differentiation", "Fit to Anjali", "Answers durability trigger"]
default_w = [30, 20, 20, 15, 15]
scores = pd.DataFrame({"C1 Integrated System": [5, 2, 5, 5, 4], "C2 Mobility-First": [2, 4, 3, 3, 3],
                       "C3 Space-Smart Collapsible": [2, 5, 2, 3, 2], "C4 Hygiene Hard Shell": [3, 3, 4, 4, 5]}, index=crit)
wc = st.columns(5)
w = [wc[i].slider(crit[i], 0, 50, default_w[i], 5, key=f"w{i}") for i in range(5)]
wsum = sum(w) or 1
tot = (scores.mul([x / wsum for x in w], axis=0)).sum()
fig = go.Figure(go.Bar(x=tot.values, y=tot.index, orientation="h",
                       marker_color=[SERIES[1] if v == tot.max() else SERIES[0] for v in tot.values],
                       text=[f"{v:.2f}" for v in tot.values], textposition="outside"))
fig.update_xaxes(range=[0, 5.3], title="Weighted score (1-5)")
st.plotly_chart(fig_layout(fig, f"Winner: {tot.idxmax()}", 300), theme=None)
note("With the original weights the Integrated System wins 4.25 to 3.65. It only loses when feasibility "
     "carries roughly half of the total weight (other weights scaled down proportionally), which is why the PRD invests in cost engineering rather than cutting features.")

st.subheader("The product")
a, b, c3 = st.columns(3)
a.image(asset("product_studio.jpg"), caption="FlexFold: 3 removable compartments, tilt-and-roll", width="stretch")
b.image(asset("product_three_modes.jpg"), caption="Standing, transport and folded modes", width="stretch")
c3.image(asset("product_folded.jpg"), caption="Folds to 8 cm beside the machine", width="stretch")

ladder = pd.DataFrame({
    "SKU": ["FlexFold Lite", "FlexFold (hero)", "FlexFold Heritage Edition", "FlexFold Pro (B2B, 50+)",
            "Liner Set", "Charcoal Odour Insert (3-pack)", "Wheel Kit"],
    "MRP": ["Rs 899", "Rs 1,799", "Rs 2,299", "Rs 1,399", "Rs 399", "Rs 299", "Rs 249"],
    "For": ["Hostel/PG students", "Anjali, dual-income 2BHK", "Gifting, design-led buyers", "Co-living, hostels",
            "All owners", "All owners", "All owners (repairability)"],
})
st.dataframe(ladder, hide_index=True, width="stretch")

st.subheader("Brand: Neat Nest")
b1, b2 = st.columns([1, 1])
with b1:
    st.markdown("""
**Tagline:** Tidy Starts Here.  ·  **Product line:** Carry Less. Roll More.

**Purpose:** give the person who absorbs invisible household labour a product that finally works, and a
presentation that treats their home with dignity.

**Values:** Quiet Respect · Earned Durability · Woven Heritage · Honest Simplicity · Everyday Dignity ·
Grounded Optimism · Considered Growth

**CBBE path:** salience at trigger moments (a broken basket, a new apartment) → performance proof (warranty,
tests) → judgements of quality and superiority → resonance through accessories, referrals and community.
""")
with b2:
    st.image(asset("packaging_flexfold_ikat.jpg"), caption="Heritage Edition sleeve: Ikat collection", width="stretch")

st.markdown("**India Craft Heritage sleeves:** one sleeve structure, five regional craft stories. Each city's landing page leads with its own sleeve.")
g = st.columns(4)
for col, (img, cap) in zip(g, [("sleeve_block_print.jpg", "Block Print · Mumbai"), ("sleeve_ikat.jpg", "Ikat · Pune, Bengaluru"),
                               ("sleeve_kalamkari_pattern.jpg", "Kalamkari · Hyderabad"), ("sleeve_bandhani.jpg", "Bandhani · Phase 2")]):
    col.image(asset(img), caption=cap, width="stretch")
