import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import header, note, page_setup, results  # noqa: E402

page_setup("Methods & Limits")
R = results()
header("Transparency", "Methods, model cards and limits",
       "What each layer is, what data it used, how it was validated, and what it cannot claim.")

note("This page is the summary. The full disclosures — including the defects that were found and fixed, and "
     "the ones deliberately left open — are on the <b>Model Cards</b> tab and in "
     "<code>knowledge/model_cards.md</code>. The <b>Parameters</b> tab carries all 89 inputs with their "
     "provenance and ranges.")

st.subheader("Data provenance")
st.markdown(f"""
| Dataset | Rows | What it represents | Origin |
|---|---|---|---|
| `survey.csv` | {R['survey_n']} | Phase-0 launch-audience survey: factor items, Van Westendorp, Kano | Simulated (seed {R['seed']}) |
| `campaign.csv` | {R['propensity']['n']:,} | Prospects reached by the Phase-1 launch campaign | Simulated |
| `customers.csv` | {R['lapse']['n']:,} | Registered FlexFold owners: month-6 snapshot + months 7-12 outcome | Simulated |
| `sales.csv` | {R['sales']['lines']:,} | Daily order lines, Apr 2027 to Mar 2028, city x SKU x channel | Simulated, with one planted anomaly |
| Market inputs | 89 parameters | Households, ownership, GST, marketplace fees, competitor prices, costs, CAC, overhead | {R['parameters']['by_kind'].get('SOURCED', 0)} sourced · {R['parameters']['by_kind'].get('BENCHMARK', 0)} benchmark · {R['parameters']['by_kind'].get('ASSUMED', 0)} assumed |
| Brand, product, persona, interviews | 4 interviews | Brand strategy, packaging, positioning, interview transcripts | Original product and brand work |

**The four datasets are independent panels of one scenario, not a linked funnel.** They share a seed, not a
population: no campaign prospect becomes a customer, no customer appears in the sales lines, and the persona,
SKU and city mixes differ because each file was drawn separately. They cannot be added up, reconciled against
each other, or traced as a cohort.

Every effect size in the simulation is a parameter declared in `TRUE_PARAMS` before any model ran. The
simulation is **not** evidence that those effects exist in the market; what it supports is a test of whether
these methods can recover a known process, which is what the **Model Cards** tab reports.
""")

st.subheader("Model cards")
P, L, V = R["propensity"], R["lapse"], R["value"]
cards = pd.DataFrame([
    {"Model": "Factor analysis", "Type": "Unsupervised, principal extraction + varimax (numpy)",
     "Validation": f"KMO {R['factors']['kmo']:.2f}; Bartlett p<0.001; {R['factors']['cumulative_pct']:.1f}% variance",
     "Use": "Persona profiling, targeting features", "Do not use for": "Individual-level decisions"},
    {"Model": "Campaign propensity", "Type": "Binary logistic regression (statsmodels + sklearn pipeline)",
     "Validation": f"CV AUC {P['cv_auc_mean']:.3f}; holdout AUC {P['holdout']['auc']:.3f}",
     "Use": "Decile ranking of prospects", "Do not use for": "Calibrated purchase probabilities"},
    {"Model": "Lapse classifier", "Type": f"{L['best']} (vs Random Forest)",
     "Validation": f"Holdout AUC {pd.DataFrame(L['metrics']).T.loc[L['best'], 'auc']:.3f}; month-6 features, months 7-12 label",
     "Use": "Prioritising retention outreach", "Do not use for": "Proving an offer causes retention"},
    {"Model": "Customer value", "Type": "OLS / Ridge / Lasso, two targets",
     "Validation": f"R2 {pd.DataFrame(V['metrics']).T.loc['OLS', 'r2']:.3f} on the leaky target, "
                   f"{pd.DataFrame(V['forward_metrics']).T.loc['OLS', 'r2']:.3f} on the forward-only one",
     "Use": "Driver ranking on the forward target",
     "Do not use for": "Anything based on the 0.93 - that figure is target leakage"},
    {"Model": "Owner segments", "Type": "K-Means (k=4) on 6 standardised features",
     "Validation": f"Silhouette {R['segments']['silhouette']['4']:.2f} at k=4; "
                   f"{R['segments']['silhouette']['2']:.2f} at k=2, which the fit prefers",
     "Use": "Retention play design", "Do not use for": "Treating segments as fixed types"},
    {"Model": "Nest Copilot", "Type": "Single bounded agent; TF-IDF RAG; optional Claude planner/explainer",
     "Validation": f"Regression suite {R['copilot_eval']['summary']['total']} (not an accuracy score); "
                   f"held-out adversarial {R['copilot_adversarial']['summary']['total']}, was 1/10",
     "Use": "Fast, traceable descriptive analysis", "Do not use for": "Forecasts, causal claims, recommendations"},
])
st.dataframe(cards, hide_index=True, width="stretch")

st.subheader("Copilot evaluation: two suites, measuring different things")
e1, e2 = st.columns(2)
e1.metric("Fixed regression suite", R["copilot_eval"]["summary"]["total"],
          help="Written during development, against the behaviour being developed.")
e2.metric("Held-out adversarial suite", R["copilot_adversarial"]["summary"]["total"],
          delta="was 1/10 on first run", delta_color="off",
          help="Written after the copilot was finished, with intent to break it.")
st.caption("The 30 questions were written **during** development, so that number means the copilot still does "
           "what it did yesterday. It is not an accuracy score. The adversarial suite was written afterwards to "
           "break it and scored 1 of 10 on its first run — most notably returning the all-India sales total as "
           "the answer to a question about a city with no data. The failures it exposed are detailed on the "
           "**Model Cards** tab. Live Claude mode is covered by mocked-client unit tests only; real-model "
           "accuracy, latency and cost are not measured anywhere in this project.")
ev = pd.DataFrame(R["copilot_eval"]["cases"])
st.dataframe(ev, hide_index=True, width="stretch")

st.subheader("Known limitations")
st.markdown(f"""
- **Every effect size is planted, not found.** `TRUE_PARAMS` declares the data-generating process and
  `recovery.py` scores the models against it. Of {R['recovery']['n_terms']} planted coefficients:
  {R['recovery']['summary'].get('recovered', 0)} recovered, {R['recovery']['summary'].get('correctly null', 0)}
  planted nulls correctly null, {R['recovery']['summary'].get('missed', 0)} missed on power,
  {R['recovery']['summary'].get('not recoverable', 0)} structurally unrecoverable,
  **{R['recovery']['false_positives']} false positives**.
- **Two real channel effects are undetectable at this sample size.** Reporting "the strongest lever is not a
  channel" as a finding was a power failure, not a result.
- **The strongest driver of lapse is never exported.** A latent engagement variable confounds every
  engagement-derived coefficient; NPS overshoots its planted value because it absorbs part of the missing driver.
- **Only {R['parameters']['by_kind'].get('SOURCED', 0)} of 89 parameters trace to a published source.**
  {R['parameters']['by_kind'].get('ASSUMED', 0)} are assumptions. The biggest lever on Year-3 EBITDA is COGS,
  which is one of them.
- **CM3 is not profit. Year-1 EBITDA is {R['economics_summary']['ebitda_cr']['Y1']:.2f} cr** and Year-1
  break-even needs about {R['break_even']['Y1']['break_even_core_units']:,.0f} core units against a plan of
  12,000. Working capital is reported as an exposure, not modelled — the largest unmodelled risk here.
- **No price-elasticity mechanism exists** anywhere in the simulator, so a competitor price war can be
  described but not modelled.
- **The price band recovers a planted anchor; the Kano result is a print-out of its own inputs.** Neither is
  market evidence.
- **Propensity and lapse scores rank, and are not calibrated probabilities** — Brier
  {pd.DataFrame(L['metrics']).T.loc[L['best'], 'brier']:.3f}, improving to
  {pd.DataFrame(L['metrics']).T.loc[L['best'], 'brier_calibrated']:.3f} under isotonic recalibration.
  The quiz effect is associational: prospects self-select into the quiz.
- **The value regression's 0.93 is target leakage.** The honest forward-only figure is
  {pd.DataFrame(V['forward_metrics']).T.loc['OLS', 'r2']:.3f}.
- **k = 4 segments is a presentation choice**, costing {R['segments']['choice']['silhouette_cost_of_choice']:.3f}
  of silhouette against the k = 2 the fit prefers.
- **Metrics are seed-sensitive** and should be quoted as ranges, not to three decimals.
- **Copilot parser** is keyword-based offline; its guards fail closed but are a gazetteer and regexes, not
  comprehension. Live mode is untested against a real model.
- **The primary interview base is 4 respondents.** Any price anchor drawn from it is indicative only.
""")

st.subheader("What this project is")
st.markdown("""
A methods portfolio and a RevOps decision framework, built end to end on one brand: positioning and product
definition, customer research, acquisition modelling, retention analytics and a guarded AI copilot, connected
so that each layer's output becomes the next layer's input.

It is demonstrated on a seeded simulation rather than live business data, which is a real constraint and is
disclosed throughout rather than buried. The thing on offer is the decision system and its validation logic —
the parameter registry, the recovery table, the EBITDA bridge, the sensitivity analysis and the guardrail
testing — all of which transfer unchanged to Phase-0 and Phase-1 field data.
""")
