"""Reference architecture and the frameworks/keyword index behind every layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import asset, card, header, note, page_setup  # noqa: E402

page_setup("Architecture", "🧭")
header("System map", "Reference architecture",
       "How data, analytics, the AI agent and the app fit together — and the marketing, research and "
       "technical frameworks used at each layer. Read top to bottom: raw data and knowledge feed four "
       "analytics layers, the analytics outputs become callable tools and a retrieval corpus for Nest "
       "Copilot, and everything is served through one Streamlit app.")

st.image(asset("diagrams/architecture_master.png"), width="stretch")
note("Five bands: <b>data &amp; knowledge</b> (what was collected or written), <b>analytics &amp; ML</b> "
     "(four independent layers, each originally a separate piece of coursework), the <b>AI agent layer</b> "
     "(Nest Copilot's five-step bounded pipeline), the <b>application layer</b> (the Streamlit app, optionally "
     "calling Claude), and the <b>business outcomes</b> each layer is accountable for.")

st.subheader("Frameworks, methods & tech stack")
st.caption("A reference index: every named framework, model and library used somewhere in this project, grouped by layer.")

tabs = st.tabs(["1 · Brand & Market", "2 · Customer Factors", "3 · Acquisition", "4 · Retention & Value",
                "5 · Nest Copilot (AI)", "Engineering"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        card("Marketing & strategy frameworks",
             "STP (Segmentation-Targeting-Positioning) · Jobs To Be Done with Forces of Progress "
             "(push / pull / anxiety / habit) · Kano model · Van Westendorp Price Sensitivity Meter · "
             "TAM-SAM-SOM market sizing · weighted concept-selection matrix · Keller's CBBE pyramid "
             "(salience, performance, judgements, resonance) · brand architecture (master brand + "
             "product family) · RACE digital marketing framework (Reach-Act-Convert-Engage) · SMART objectives.")
    with c2:
        card("Commercial & financial methods",
             "GST-aware pricing ladder (HSN 6307, 5%/18% threshold at Rs 2,500) · contribution-margin "
             "waterfall (CM1 / CM2 / CM3) · channel economics (D2C vs. marketplace fee structures) · "
             "unit economics (CAC, COGS, packaging, shipping) · 3-year SOM phasing · competitive teardown.")

with tabs[1]:
    card("Exploratory Factor Analysis (built from scratch in numpy)",
         "Kaiser-Meyer-Olkin (KMO) sampling adequacy · Bartlett's test of sphericity · principal-axis "
         "extraction · Varimax (orthogonal) rotation · eigenvalues &amp; scree criterion · communalities · "
         "factor scores · persona profiling against latent dimensions. Validated against Kano classification "
         "and Van Westendorp price bands collected in the same survey instrument.")

with tabs[2]:
    card("Binary logistic regression (acquisition / propensity)",
         "statsmodels Logit for interpretable coefficients &amp; odds ratios · scikit-learn Pipeline "
         "(preprocessing + model, no leakage) · stratified train/test split &amp; k-fold cross-validation · "
         "ROC-AUC, holdout evaluation · McFadden pseudo-R² · decile lift / gains charts for targeting · "
         "multicollinearity checks (VIF).")

with tabs[3]:
    card("Retention, value & segmentation",
         "Logistic regression vs. Random Forest classifier (lapse risk) · permutation feature importance · "
         "OLS, Ridge &amp; Lasso regression (12-month customer value) · VIF multicollinearity diagnostics · "
         "K-Means clustering with silhouette scoring (owner segments) · Customer Lifetime Value (CLV) "
         "modelling · \"lapse\" redefinition of churn for durable, infrequently-repurchased goods.")

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        card("Agentic & RAG architecture",
             "Bounded / guarded LLM agent pattern (Python computes, the LLM only plans and explains) · "
             "Retrieval-Augmented Generation via TF-IDF + cosine similarity with a title-term bonus · "
             "forced JSON-schema tool calls (Anthropic Messages API, Claude) · prompt-injection &amp; "
             "scope-blocklisting guardrails · hallucination checks (no-digits-in-prose rule, citation-ID "
             "verification) · offline-first design with an optional live-LLM mode.")
    with c2:
        card("Evaluation & governance",
             "Fixed-question regression test suite (30/30 offline) · model cards per layer · mocked-client "
             "unit tests for the live Claude path · graceful degradation (ambiguous question → clarify; "
             "forecast request → decline; missing data → error, never a silent zero) · downloadable audit "
             "record per answer.")

with tabs[5]:
    card("Engineering & MLOps",
         "Python package layout (src/ layout, src/neatnest) · scikit-learn Pipelines for leakage-free "
         "modelling · joblib model persistence · pytest + Streamlit AppTest (38 tests) · GitHub Actions CI · "
         "Streamlit multi-page apps · matplotlib &amp; Plotly data visualisation on a validated, "
         "colour-vision-deficient-safe brand palette · reproducible, seeded data simulation.")

st.subheader("Five pieces of coursework, one system")
rows = pd.DataFrame({
    "Original assignment": ["Product & Brand Management", "Factor Analysis", "Binary Logistic Regression",
                            "Advanced Analytics (churn/retention)", "Generative & Agentic AI / RAG"],
    "Became, in this system": ["Layer 1 — brand, product, market & pricing", "Layer 2 — customer factors",
                               "Layer 3 — acquisition / propensity", "Layer 4 — retention & value",
                               "Layer 5 — Nest Copilot"],
    "Pages": ["0, 1, 2", "3", "4", "5", "6"],
}, index=range(1, 6))
st.dataframe(rows, width="stretch")
note("See the <b>Methods &amp; Limits</b> page for data provenance and honest limits, and the capstone brief "
     "for the full narrative connecting brand strategy to the AI layer.")
