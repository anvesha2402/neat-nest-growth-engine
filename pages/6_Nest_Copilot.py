import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from neatnest.app_utils import data, fig_layout, header, note, page_setup, report_csv  # noqa: E402
from neatnest.config import SERIES  # noqa: E402
from neatnest.copilot import Copilot  # noqa: E402
from neatnest.copilot import llm as llm_mod  # noqa: E402
from neatnest.copilot.engine import fmt_metric, inr  # noqa: E402
from neatnest.copilot.tools import DataError, MAX_MB  # noqa: E402

page_setup("Nest Copilot", "🤖")
header("Layer 5 · AI decision layer", "Nest Copilot",
       "Ask a question about Neat Nest's sales, customers or playbook in plain English. Python calculates every number. "
       "The language model (live mode only) may only choose an approved plan and explain evidence that has already "
       "been calculated.")

with st.sidebar:
    st.header("Data and settings")
    up = st.file_uploader("Sales CSV (optional)", type=["csv"], help=f"Same columns as data/sales.csv, max {MAX_MB} MB")
    mode = st.radio("Mode", ["Offline demo", "Live AI (Claude)"], help="Offline needs no key and no internet.")
    consent = False
    if mode.startswith("Live"):
        consent = st.checkbox("I agree to send my question, a summary of the calculated evidence and the retrieved "
                              "rules to Anthropic. The full CSV is never sent.")
        if not llm_mod.available():
            st.warning("ANTHROPIC_API_KEY is not set, so the live steps will fall back to offline.")
    st.caption("Do not upload personal data. This is a classroom prototype without user accounts.")


@st.cache_resource
def get_copilot(_bytes: bytes | None, live: bool):
    sales = pd.read_csv(io.BytesIO(_bytes)) if _bytes else data("sales")
    art = {"propensity_lift": report_csv("propensity_lift", 0), "risk_table": report_csv("risk_table"),
           "segments": report_csv("segments", 0)}
    client = None
    if live and llm_mod.available():
        client = llm_mod.ClaudeClient()
    return Copilot(sales, artifacts=art, llm=client)


raw = None
if up is not None:
    if up.size > MAX_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_MB} MB."); st.stop()
    raw = up.getvalue()
try:
    cp = get_copilot(raw, mode.startswith("Live"))
except DataError as e:
    st.error(f"Upload rejected: {e}"); st.stop()

ctx = cp.ctx
st.caption(f"Data: {'uploaded file' if raw else 'simulated Neat Nest launch year'} · {len(cp.df):,} order lines · "
           f"{ctx['months'][0]} to {ctx['months'][-1]} · {len(ctx['cities'])} cities · INR, GST excluded")

examples = ["Why did sales decline in Hyderabad?", "Compare February and March net sales by product in Hyderabad",
            "Which city has the highest gross profit?", "Show monthly units for accessories",
            "Who should we target in the launch campaign?", "Which customers are at risk of lapsing in Pune?",
            "What warranty does FlexFold carry?", "Which city is best?", "Predict next year sales"]
st.write("**Try:**")
ec = st.columns(3)
for i, q in enumerate(examples):
    if ec[i % 3].button(q, key=f"ex{i}", width="stretch"):
        st.session_state["q"] = q

with st.form("ask"):
    q = st.text_input("Your question", value=st.session_state.get("q", ""), max_chars=1000)
    go_btn = st.form_submit_button("Ask", type="primary")
if go_btn:
    st.session_state["q"] = q
    st.session_state["res"] = cp.ask(q, mode="live" if mode.startswith("Live") else "offline", consent=consent)

res = st.session_state.get("res")
if res is None:
    st.stop()

if res.status in ("clarify", "unsupported", "blocked", "data_error"):
    icon = {"clarify": "❓", "unsupported": "🚫", "blocked": "🔒", "data_error": "⚠️"}[res.status]
    st.info(f"{icon} {res.plan.get('message', '')}")
    for w in res.warnings:
        st.caption(w)
    st.stop()

ev, plan = res.evidence, res.plan
for a in ev.assumptions:
    st.info(a)
st.subheader(ev.title)
m = plan["metric"]
if plan["operation"] in ("explain", "compare") and ev.headline:
    keys = [k for k in ev.headline if k != "change_pct"]
    c = st.columns(3)
    c[0].metric(keys[0], fmt_metric(m, ev.headline[keys[0]]))
    c[1].metric(keys[1], fmt_metric(m, ev.headline[keys[1]]))
    if ev.headline.get("change_pct") is not None:
        c[2].metric("Change", f"{ev.headline['change_pct']:+.2f}%")
elif plan["operation"] == "total":
    st.metric(m.replace("_", " ").title(), fmt_metric(m, ev.facts["value"]))

ex = res.explanation
if ex.get("summary"):
    st.markdown(ex["summary"])
if ex.get("actions"):
    st.markdown("**Suggested next checks**\n" + "\n".join(f"- {x}" for x in ex["actions"]))
if ex.get("limitation"):
    st.caption(f"Limitation: {ex['limitation']}")
st.caption(f"Mode: {res.mode} · explanation by {res.explained_by} · {res.seconds * 1000:.0f} ms")
for w in res.warnings:
    st.warning(w)

t = ev.table
if t is not None and len(t):
    if plan["operation"] == "explain":
        fig = go.Figure()
        fig.add_bar(y=t.index, x=t["prev_sales"], name=plan["previous_month"], orientation="h", marker_color=SERIES[0])
        fig.add_bar(y=t.index, x=t["cur_sales"], name=plan["month"], orientation="h", marker_color=SERIES[1])
        fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="Net sales (Rs)")
        st.plotly_chart(fig_layout(fig, "Net sales by product", 360), theme=None)
        st.markdown("**By channel**")
        st.dataframe(ev.facts["channel_table"].style.format("{:,.0f}"), width="stretch")
    elif ev.chart and ev.chart["type"] in ("bar", "line") and ev.chart["y"] in t.columns:
        x = t.index.astype(str)
        trace = go.Bar(x=x, y=t[ev.chart["y"]], marker_color=SERIES[0]) if ev.chart["type"] == "bar" else \
            go.Scatter(x=x, y=t[ev.chart["y"]], mode="lines+markers", line={"color": SERIES[0], "width": 2})
        st.plotly_chart(fig_layout(go.Figure(trace), None, 340), theme=None)
    elif plan["operation"] == "compare" and plan.get("group_by"):
        fig = go.Figure()
        fig.add_bar(x=t.index, y=t[plan["previous_month"]], name=plan["previous_month"], marker_color=SERIES[0])
        fig.add_bar(x=t.index, y=t[plan["month"]], name=plan["month"], marker_color=SERIES[1])
        st.plotly_chart(fig_layout(fig, None, 340), theme=None)
    st.markdown("**Calculated evidence**")
    num = t.select_dtypes("number").columns
    st.dataframe(t.style.format({c: "{:,.2f}" for c in num}), width="stretch")
    st.download_button("Download evidence (CSV)", t.to_csv().encode(), "evidence.csv", "text/csv")

with st.expander(f"Retrieved knowledge ({len(res.sources)})"):
    for s in res.sources:
        st.markdown(f"**{s['id']}** · {s['title']} · *{s['doc']}* · score {s['score']}")
        st.caption(s["text"])
with st.expander("Pipeline record (audit trail)"):
    st.json(plan)
    st.write({"usage": res.usage})
    record = {"question": res.question, "mode": res.mode, "plan": plan, "sources": [s["id"] for s in res.sources],
              "explanation": ex, "warnings": res.warnings}
    st.download_button("Download record (JSON)", json.dumps(record, indent=1, default=str), "copilot_record.json")
note("Guardrails: request blocklist · JSON-schema plans restricted to 9 operations and 5 metrics · values checked "
     "against the data · missing months raise an error instead of counting as zero · live prose containing digits or "
     "unknown source IDs is discarded · consent gate before anything leaves the app.")
