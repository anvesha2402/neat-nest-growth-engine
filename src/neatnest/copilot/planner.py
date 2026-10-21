"""Question -> validated plan. Offline keyword planner plus a strict validator
that every plan (offline or LLM-produced) must pass before anything runs."""
from __future__ import annotations

import re

from .tools import DIMENSIONS, METRICS

OPERATIONS = ["total", "breakdown", "trend", "compare", "explain", "knowledge",
              "targeting", "customer_risk", "segment_profile"]
PLAN_FIELDS = {"status", "operation", "metric", "city", "product", "channel", "family", "group_by",
               "month", "previous_month", "sort", "message"}

BLOCKLIST = ["api key", "apikey", "password", "secret", "token", "ignore previous", "ignore instructions",
             "ignore all", "system prompt", "drop table", "delete", "truncate", "execute", "exec(", "import os",
             "rm -rf", "update set", "insert into", "aadhaar", "credit card",
             # Added after the adversarial suite (copilot/adversarial.py) got 9 of 10
             # attacks through. These are the phrasings that landed: an override
             # smuggled alongside a legitimate request, a role reassignment, and an
             # instruction dressed up as a data row.
             "ignore your", "ignore the tool", "ignore tool", "your instructions", "your restrictions",
             "tool restrictions", "developer mode", "reveal your", "print your", "system message",
             "note to assistant", "for the rest of this conversation", "from your own knowledge",
             "list every tool", "your configuration"]

# Metrics people reasonably ask for that are NOT in the sales lines. Without this,
# the planner silently falls back to net_sales and answers a question about EBITDA
# with a revenue figure.
OUT_OF_SCOPE_METRICS = ["ebitda", "ebit", "cac", "customer acquisition cost", "acquisition cost",
                        "ltv", "cltv", "lifetime value", "roas", "contribution margin", "cm1", "cm2",
                        "cm3", "gmv", "working capital", "break-even", "breakeven", "payback",
                        "net profit", "net margin", "operating profit", "overhead", "burn rate"]

# Entities absent from the simulated data. The failure these prevent is the worst
# one in the suite: asked for a city that does not exist, the planner left `city`
# as None, the filter was dropped, and the all-India total was returned as if it
# were that city's figure.
ABSENT_CITIES = ["chennai", "kolkata", "calcutta", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur",
                 "nagpur", "indore", "bhopal", "patna", "vadodara", "coimbatore", "kochi", "cochin",
                 "chandigarh", "visakhapatnam", "vizag", "thiruvananthapuram", "guwahati", "bhubaneswar",
                 "ludhiana", "agra", "varanasi", "nashik", "rajkot", "madurai", "mysore", "mysuru",
                 "goa", "noida", "gurgaon", "gurugram", "faridabad", "ghaziabad", "thane", "navi mumbai"]

# SKU names that do not exist in the ladder.
ABSENT_SKUS = ["mini", "max", "plus", "air", "ultra", "compact", "nano", "grande", "xl", "slim",
               "premium edition", "classic", "deluxe"]

RECOMMENDATION_WORDS = r"\b(should we|should i|do you recommend|would you recommend|what should we|" \
                       r"recommend that we|is it worth|ought we)\b"

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], start=1)}
PRODUCT_WORDS = [  # order matters: specific variants before the bare family name
    ("heritage", "FlexFold Heritage Edition"), ("lite", "FlexFold Lite"), ("pro", "FlexFold Pro (B2B)"),
    ("liner", "Liner Set"), ("charcoal", "Charcoal Odour Insert (3-pack)"), ("odour insert", "Charcoal Odour Insert (3-pack)"),
    ("wheel kit", "Wheel Kit"), ("flexfold", "FlexFold"),
]
CHANNEL_WORDS = [("amazon", "Amazon"), ("marketplace", "Amazon"), ("d2c", "D2C"), ("website", "D2C"),
                 ("b2b", "B2B"), ("institutional", "B2B"), ("co-living", "B2B")]
GROUP_WORDS = {"city": ["city", "cities", "region", "regions", "market"], "product": ["product", "products", "sku", "skus"],
               "channel": ["channel", "channels"], "family": ["family", "families", "category", "categories"]}
KNOWLEDGE_HINTS = ["tagline", "brand", "warranty", "tone", "voice", "sleeve", "packaging", "phase", "entry",
                   "positioning", "persona", "requirement", "prd", "kano", "policy", "guardrail", "price of",
                   "pricing", "priced", "mission", "vision", "purpose", "values", "race", "playbook", "colour",
                   "color", "palette", "retired name", "roadmap"]


def metric_named_data(q: str, has_entity: bool) -> bool:
    """A metric word plus a scope (city/product/channel) means a data question, not a definition."""
    return has_entity and bool(re.search(r"sales|revenue|profit|margin|units|sold|discount", q))


def blocked(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in BLOCKLIST)


def _month(token_mon: str, available: list[str]) -> str | None:
    num = MONTHS[token_mon[:3]]
    cands = [m for m in available if int(m[5:7]) == num]
    return cands[-1] if cands else None


def offline_plan(question: str, ctx: dict) -> dict:
    q = question.lower().strip()
    months, cities = ctx["months"], ctx["cities"]
    plan = {"status": "ok", "operation": "total", "metric": "net_sales", "city": None, "product": None,
            "channel": None, "family": None, "group_by": None, "month": None, "previous_month": None,
            "sort": "desc", "message": ""}
    if blocked(q):
        return {**plan, "status": "unsupported", "message": "This request is outside what the copilot is allowed to do."}
    if re.search(r"\b(forecast|predict|next year|next month|future sales)\b", q):
        return {**plan, "status": "unsupported",
                "message": "Forecasting is not supported: the copilot only describes recorded sales and model scores."}

    # --- entity and scope guards (added after the adversarial suite) ----------
    for m in OUT_OF_SCOPE_METRICS:
        if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", q):
            return {**plan, "status": "unsupported",
                    "message": (f"'{m}' is not in the loaded sales data, which holds units, net sales, cost, "
                                f"discount and gross margin by date, city, SKU and channel. Planning figures "
                                f"such as CAC, contribution margin and EBITDA live in the economics model "
                                f"(the Unit Economics and Parameters tabs), not here.")}
    for c in ABSENT_CITIES:
        if re.search(rf"\b{re.escape(c)}\b", q):
            return {**plan, "status": "data_error",
                    "message": (f"There is no data for {c.title()}. The simulated launch covers "
                                f"{', '.join(cities)}.")}
    for s in ABSENT_SKUS:
        if re.search(rf"\b(flexfold|sku|edition|model)\b", q) and re.search(rf"\b{re.escape(s)}\b", q):
            return {**plan, "status": "data_error",
                    "message": (f"There is no '{s}' SKU. The ladder is FlexFold Lite, FlexFold, FlexFold "
                                f"Heritage Edition, FlexFold Pro (B2B), Liner Set, Charcoal Odour Insert "
                                f"and Wheel Kit.")}
    years = sorted({m[:4] for m in months})
    for y in re.findall(r"\b(?:19|20)\d\d\b", q):
        if y not in years:
            return {**plan, "status": "data_error",
                    "message": f"There is no data for {y}. The simulated period runs {months[0]} to {months[-1]}."}

    # entities
    for c in cities:
        if c.lower() in q or (c == "Delhi NCR" and ("delhi" in q or "ncr" in q)):
            plan["city"] = c
    for w, p in PRODUCT_WORDS:
        if re.search(rf"\b{w}\b", q):
            plan["product"] = p
            break
    if "accessor" in q:
        plan["family"], plan["product"] = "Accessories", None
    for w, ch in CHANNEL_WORDS:
        if w in q:
            plan["channel"] = ch
            break
    if re.search(r"\bprofit", q):
        plan["metric"] = "gross_profit"
    if "margin" in q:
        plan["metric"] = "gross_margin"
    if re.search(r"\b(units|volume|sold|how many)\b", q):
        plan["metric"] = "units"
    if "discount" in q:
        plan["metric"] = "discount"
    metric_named = bool(re.search(r"profit|margin|units|volume|sold|how many|discount|sales|revenue", q))

    found = re.findall(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                       r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b", q)
    mlist = [x for x in (_month(f, months) for f in found) if x]
    mlist = sorted(set(mlist), key=mlist.index)

    group = None
    for dim, words in GROUP_WORDS.items():
        if re.search(rf"\b(by|per|each|which|across|every)\s+({'|'.join(words)})\b", q) or \
           re.search(rf"\b({'|'.join(words)})\s+(wise|breakdown|split)\b", q):
            group = dim
    ranking = re.search(r"\b(best|top|highest|most|largest|biggest|lowest|least|worst|weakest|smallest)\b", q)

    has_entity = any(plan[d] for d in ("city", "product", "channel", "family"))
    is_def = re.search(r"^(what is|what's|what are|define|meaning of|how is|how are|how do we define)|"
                       r"\b(defined|definition|meaning)\b", q)
    if is_def and not has_entity and not ranking and not group and not mlist and re.search(
            r"\b(defined|definition|meaning|define|how is|how are)\b", q):
        return {**plan, "operation": "knowledge"}

    # model-backed tools
    if re.search(r"\b(target|propensity|prospect|likely to buy|who should we (target|retarget))", q):
        return {**plan, "operation": "targeting"}
    if re.search(r"\b(at risk|lapse|lapsing|churn|retain|retention|save list)\b", q):
        return {**plan, "operation": "customer_risk"}
    if re.search(r"\b(segment|cluster)s?\b", q):
        return {**plan, "operation": "segment_profile"}

    # Recommendation requests. Placed AFTER the model-backed tools so that
    # "who should we target" still routes to targeting - the copilot reports
    # figures and scores, it does not decide things.
    if re.search(RECOMMENDATION_WORDS, q):
        return {**plan, "status": "unsupported",
                "message": ("The copilot reports recorded figures and model scores; it does not make "
                            "recommendations. Ask for the numbers behind the decision instead, for "
                            "example units and gross margin by SKU.")}

    # knowledge questions (definitions, brand, PRD, policy) - only when not a ranking/data request
    if (is_def or any(h in q for h in KNOWLEDGE_HINTS)) and not ranking and not group and not mlist \
            and not re.search(r"\bwhy\b", q) and (not has_entity or any(h in q for h in KNOWLEDGE_HINTS)) \
            and not metric_named_data(q, has_entity):
        return {**plan, "operation": "knowledge"}

    if re.search(r"\bwhy\b", q) and re.search(r"declin|drop|fall|fell|down|dip|decrease|slump", q):
        plan["operation"] = "explain"
        plan["metric"] = "net_sales"
    elif re.search(r"\b(compare|versus|vs\.?)\b", q) or len(mlist) >= 2:
        plan["operation"] = "compare"
    elif re.search(r"\b(trend|monthly|over time|by month|month by month|each month)\b", q):
        plan["operation"] = "trend"
    elif group or ranking:
        if ranking and not metric_named and not group:
            return {**plan, "status": "clarify",
                    "message": "Best by which measure: net sales, gross profit, gross margin or units?"}
        if ranking and not metric_named:
            return {**plan, "status": "clarify",
                    "message": f"Rank {group}s by which measure: net sales, gross profit, gross margin or units?"}
        plan["operation"] = "breakdown"
        plan["group_by"] = group or ("product" if not plan["product"] else "city")
        if ranking and ranking.group(1) in ("lowest", "least", "worst", "weakest", "smallest"):
            plan["sort"] = "asc"
        plan["group_by"] = plan["group_by"]
    # periods
    if plan["operation"] in ("compare", "explain"):
        if len(mlist) >= 2:
            a, b = sorted(mlist[:2])
            plan["previous_month"], plan["month"] = a, b
        elif len(mlist) == 1:
            plan["month"] = mlist[0]
            i = months.index(mlist[0])
            plan["previous_month"] = months[i - 1] if i > 0 else None
        else:
            plan["month"], plan["previous_month"] = months[-1], months[-2]
            plan["message"] = f"No months named, so the latest month ({months[-1]}) is compared with {months[-2]}."
        if plan["operation"] == "compare" and group:
            plan["group_by"] = group
    elif mlist:
        plan["month"] = mlist[0]
    if plan["operation"] == "breakdown" and plan["group_by"] in DIMENSIONS and plan.get(plan["group_by"]):
        plan[plan["group_by"]] = None  # do not filter and group on the same dimension
    return plan


def validate_plan(plan: dict, ctx: dict) -> dict:
    """Reject anything outside the approved schema. Returns the plan or raises ValueError."""
    extra = set(plan) - PLAN_FIELDS
    if extra:
        raise ValueError(f"Unknown plan fields: {sorted(extra)}")
    if plan.get("status") not in ("ok", "clarify", "unsupported", "data_error"):
        raise ValueError("Invalid status")
    if plan["status"] != "ok":
        return plan
    if plan.get("operation") not in OPERATIONS:
        raise ValueError(f"Operation not approved: {plan.get('operation')}")
    if plan.get("metric") not in METRICS:
        raise ValueError(f"Metric not approved: {plan.get('metric')}")
    allowed = {"city": ctx["cities"], "product": ctx["products"], "channel": ctx["channels"], "family": ctx["families"]}
    for dim, vals in allowed.items():
        if plan.get(dim) is not None and plan[dim] not in vals:
            raise ValueError(f"Unknown {dim}: {plan[dim]}")
    if plan.get("group_by") is not None and plan["group_by"] not in DIMENSIONS:
        raise ValueError("Grouping not approved")
    if plan["operation"] == "breakdown" and not plan.get("group_by"):
        raise ValueError("Breakdown needs a grouping")
    for f in ("month", "previous_month"):
        if plan.get(f) is not None and plan[f] not in ctx["months"]:
            raise ValueError(f"{f} {plan[f]} is outside the data ({ctx['months'][0]} to {ctx['months'][-1]})")
    if plan["operation"] in ("compare", "explain"):
        if not plan.get("month") or not plan.get("previous_month"):
            raise ValueError("Compare/explain needs two months")
        if plan["previous_month"] >= plan["month"]:
            raise ValueError("previous_month must be before month")
    if plan.get("sort") not in (None, "asc", "desc"):
        raise ValueError("Invalid sort")
    return plan
