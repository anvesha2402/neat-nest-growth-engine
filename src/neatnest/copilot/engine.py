"""Nest Copilot request lifecycle.

question -> safety pre-check -> plan (offline keywords or Claude) -> validate
-> execute approved tool in Python -> retrieve policy passages -> explain
(template or Claude, with output checks) -> result with full audit record.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from . import tools as T
from .knowledge import Retriever
from .planner import offline_plan, validate_plan


def inr(x: float, dp: int = 0) -> str:
    """Indian digit grouping, e.g. 1234567 -> 12,34,567."""
    neg = x < 0
    s = f"{abs(x):.{dp}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        whole = f"{head},{tail}"
    return ("-" if neg else "") + "Rs " + whole + (f".{frac}" if frac else "")


def fmt_metric(m: str, v: float) -> str:
    if m in ("net_sales", "gross_profit"):
        return inr(v)
    if m in ("gross_margin", "discount"):
        return f"{v:.2f}%"
    return f"{v:,.0f}"


@dataclass
class Result:
    question: str
    mode: str
    plan: dict
    status: str
    evidence: T.Evidence | None = None
    explanation: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    explained_by: str = "template"


ACTIONS = {
    "explain": ["Check city stock and sleeve inventory for the product with the largest fall (POLICY_STOCKOUT).",
                "Compare price per unit and average discount with the previous month (POLICY_DISCOUNT).",
                "Ask the city team about listing status, local events and competitor offers before concluding."],
    "compare": ["Look at the group with the largest change first.",
                "Split that group by product and channel before drawing conclusions."],
    "breakdown": ["Check whether the ranking holds on gross profit as well as net sales."],
    "trend": ["Read peaks against the season (Diwali, monsoon) and city launch dates before comparing months."],
    "targeting": ["Put retargeting and creator-code budget on the top three deciles first (GTM_TARGETING).",
                  "Keep a random holdout so that true incremental conversion can be measured."],
    "customer_risk": ["Contact the highest-risk customers first using the segment play in GTM_RETENTION.",
                      "Resolve open product issues before sending any offer.",
                      "Hold out a random control group to measure the real save rate."],
    "segment_profile": ["Match each segment to its retention play in GTM_RETENTION."],
}


class Copilot:
    def __init__(self, sales: pd.DataFrame, retriever: Retriever | None = None, artifacts: dict | None = None,
                 llm=None):
        self.df = T.validate_sales(sales)
        self.retriever = retriever or Retriever()
        self.art = artifacts or {}
        self.llm = llm
        self.ctx = {"months": sorted(self.df["month"].unique()), "cities": sorted(self.df["city"].unique()),
                    "products": sorted(self.df["product"].unique()), "channels": sorted(self.df["channel"].unique()),
                    "families": sorted(self.df["family"].unique())}

    # ------------------------------------------------------------------ run
    def ask(self, question: str, mode: str = "offline", consent: bool = False) -> Result:
        t0 = time.perf_counter()
        question = (question or "").strip()[:1000]
        res = Result(question=question, mode=mode, plan={}, status="ok")
        if not question:
            res.status, res.plan = "clarify", {"status": "clarify", "message": "Please type a question."}
            return res
        live = mode == "live"
        if live and not consent:
            res.status = "blocked"
            res.plan = {"status": "unsupported", "message": "Tick the data-sharing consent box to use Live AI mode."}
            res.warnings.append("Nothing was sent to the model provider.")
            return res
        # plan
        plan = offline_plan(question, self.ctx)
        if live and self.llm is not None and plan["status"] == "ok":
            try:
                plan = self.llm.plan(question, self.ctx)
            except Exception as e:  # noqa: BLE001
                res.warnings.append(f"Live planner unavailable ({type(e).__name__}); used the offline planner.")
        try:
            plan = validate_plan(plan, self.ctx)
        except ValueError as e:
            res.warnings.append(f"Plan rejected: {e}. Used the offline planner.")
            plan = validate_plan(offline_plan(question, self.ctx), self.ctx)
        res.plan = plan
        if plan["status"] != "ok":
            res.status = plan["status"]
            res.seconds = time.perf_counter() - t0
            return res
        # execute
        try:
            if plan["operation"] == "explain" and not any(plan.get(d) for d in T.DIMENSIONS):
                city = T.find_largest_decline(self.df, plan["month"], plan["previous_month"])
                if city:
                    plan["city"] = city
                    plan["message"] = (plan.get("message", "") + f" No scope named, so the city with the largest "
                                       f"decline ({city}) was selected.").strip()
            res.evidence = self._execute(plan)
            if plan.get("message"):
                res.evidence.assumptions.append(plan["message"])
        except T.DataError as e:
            res.status = "data_error"
            res.plan = {**plan, "message": str(e)}
            res.seconds = time.perf_counter() - t0
            return res
        # retrieve
        query = question if plan["operation"] == "knowledge" else f"{question} {plan['operation']} {plan['metric']}"
        hits = self.retriever.search(query, k=3 if plan["operation"] == "knowledge" else 2)
        res.sources = [{"id": c.id, "title": c.title, "doc": c.doc, "score": round(s, 4), "text": c.text}
                       for c, s in hits]
        # explain
        res.explanation = self._template(plan, res)
        if live and self.llm is not None and plan["operation"] != "knowledge":
            try:
                gen = self.llm.explain(question, self._summary(res.evidence),
                                       [{"id": s["id"], "text": s["text"]} for s in res.sources])
                ok, why = self._check(gen, {s["id"] for s in res.sources})
                if ok:
                    res.explanation, res.explained_by = gen, "claude"
                else:
                    res.warnings.append(f"Generated explanation rejected ({why}); template shown instead.")
            except Exception as e:  # noqa: BLE001
                res.warnings.append(f"Live explanation unavailable ({type(e).__name__}); template shown instead.")
        if self.llm is not None:
            res.usage = list(self.llm.usage)
        res.seconds = time.perf_counter() - t0
        return res

    # -------------------------------------------------------------- helpers
    def _execute(self, plan: dict) -> T.Evidence:
        op = plan["operation"]
        if op in ("total", "breakdown", "trend", "compare", "explain"):
            return getattr(T, f"op_{op}")(self.df, plan)
        if op == "knowledge":
            return T.Evidence(title="Answer from the Neat Nest knowledge base")
        if op == "targeting":
            lift = self.art.get("propensity_lift")
            if lift is None:
                raise T.DataError("Propensity model artifacts are not loaded.")
            top3 = float(lift["cum_capture"].iloc[2])
            return T.Evidence(title="Launch-campaign targeting by propensity decile", table=lift.round(3),
                              chart={"type": "bar", "x": "decile", "y": "conv_rate"},
                              facts={"top3_capture": top3, "top_lift": float(lift["lift"].iloc[0])})
        if op == "customer_risk":
            risk = self.art.get("risk_table")
            if risk is None:
                raise T.DataError("Retention model artifacts are not loaded.")
            sub = risk
            if plan.get("city"):
                sub = sub[sub["city"] == plan["city"]]
            top = sub.sort_values("lapse_risk", ascending=False).head(25)
            return T.Evidence(title="Highest lapse-risk FlexFold owners", table=top.reset_index(drop=True),
                              facts={"n": len(sub), "mean_risk": float(sub["lapse_risk"].mean()),
                                     "top_segment": top["segment"].mode().iat[0]})
        if op == "segment_profile":
            prof = self.art.get("segments")
            if prof is None:
                raise T.DataError("Segment artifacts are not loaded.")
            return T.Evidence(title="Owner segments (K-Means)", table=prof,
                              facts={"worst": prof["lapse_rate"].idxmax(), "best": prof["value_12m"].idxmax()})
        raise T.DataError("Unsupported operation")

    @staticmethod
    def _summary(ev: T.Evidence) -> dict:
        facts = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in ev.facts.items()}
        return {"title": ev.title, "facts": facts}

    @staticmethod
    def _check(gen: dict, allowed_ids: set) -> tuple[bool, str]:
        prose = " ".join([gen.get("summary", ""), gen.get("limitation", "")] + list(gen.get("actions", [])))
        if re.search(r"\d", prose):
            return False, "contained digits"
        if not set(gen.get("source_ids", [])) <= allowed_ids:
            return False, "cited an unknown source"
        if len(gen.get("actions", [])) > 3:
            return False, "more than three actions"
        return True, ""

    def _template(self, plan: dict, res: Result) -> dict:
        op, ev, m = plan["operation"], res.evidence, plan["metric"]
        f = ev.facts if ev else {}
        lim = "These are observed patterns in simulated data; they show where something changed, not why."
        if op == "knowledge":
            if not res.sources:
                return {"summary": "The knowledge base has no passage that answers this. Try naming the topic "
                                   "(pricing, warranty, tagline, entry phases, KPI).", "actions": [], "limitation": ""}
            return {"summary": res.sources[0]["text"], "actions": [],
                    "limitation": "Approved knowledge-base text, quoted without modification."}
        if op == "explain":
            share = f" ({f['share_of_decline']:.0f}% of the fall)" if f.get("share_of_decline") else ""
            disc = (f" Average discount moved from {f['discount_prev']:.1f}% to {f['discount_cur']:.1f}%."
                    if abs(f["discount_cur"] - f["discount_prev"]) >= 0.5 else "")
            s = (f"Net sales in {f['scope']} moved from {inr(f['prev'])} to {inr(f['cur'])} "
                 f"({(f['cur'] - f['prev']) / f['prev'] * 100:+.1f}%). The largest contribution is {f['top_product']}"
                 f" at {inr(f['top_change'])}{share}, with units {f['top_units_pct']:+.0f}%.{disc}")
            return {"summary": s, "actions": ACTIONS["explain"], "limitation": lim}
        if op == "total":
            return {"summary": f"{m.replace('_', ' ').capitalize()} for the selected scope is "
                               f"{fmt_metric(m, f['value'])}.", "actions": [], "limitation": lim}
        if op == "breakdown":
            word = "lowest" if f.get("ascending") else "highest"
            return {"summary": f"{f['leader']} has the {word} {m.replace('_', ' ')} at {fmt_metric(m, f['value'])}.",
                    "actions": ACTIONS["breakdown"], "limitation": lim}
        if op == "trend":
            return {"summary": f"{m.replace('_', ' ').capitalize()} peaked in {f['peak']} at "
                               f"{fmt_metric(m, f['peak_value'])}.", "actions": ACTIONS["trend"], "limitation": lim}
        if op == "compare":
            if m in ("gross_margin", "discount"):
                s = f"{m.replace('_', ' ').capitalize()} moved from {f['prev']:.2f}% to {f['cur']:.2f}% ({f['cur'] - f['prev']:+.2f} pp)."
            else:
                s = (f"{m.replace('_', ' ').capitalize()} moved from {fmt_metric(m, f['prev'])} to "
                     f"{fmt_metric(m, f['cur'])} ({(f['cur'] - f['prev']) / f['prev'] * 100:+.1f}%).")
            return {"summary": s, "actions": ACTIONS["compare"], "limitation": lim}
        if op == "targeting":
            return {"summary": f"The top three propensity deciles hold {f['top3_capture'] * 100:.0f}% of buyers in the "
                               f"holdout; the top decile converts at {f['top_lift']:.1f}x the average.",
                    "actions": ACTIONS["targeting"],
                    "limitation": "Scores rank prospects; they are not guaranteed purchase probabilities."}
        if op == "customer_risk":
            return {"summary": f"Average predicted lapse risk across {f['n']:,} owners is {f['mean_risk'] * 100:.0f}%. "
                               f"The highest-risk list is dominated by the '{f['top_segment']}' segment.",
                    "actions": ACTIONS["customer_risk"],
                    "limitation": "Risk scores show association, not cause; only a holdout test proves a save."}
        if op == "segment_profile":
            return {"summary": f"'{f['best']}' carries the highest 12-month value; '{f['worst']}' has the highest lapse rate.",
                    "actions": ACTIONS["segment_profile"],
                    "limitation": "Segments overlap (low silhouette); treat them as planning groups, not hard types."}
        return {"summary": "", "actions": [], "limitation": lim}
