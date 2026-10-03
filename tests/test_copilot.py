from types import SimpleNamespace

import pandas as pd
import pytest

from neatnest.config import DATA_DIR
from neatnest.copilot import Copilot, inr
from neatnest.copilot.evaluate import load_artifacts, run_suite
from neatnest.copilot.planner import offline_plan, validate_plan
from neatnest.copilot.tools import DataError, validate_sales


@pytest.fixture(scope="module")
def cp():
    return Copilot(pd.read_csv(DATA_DIR / "sales.csv"), artifacts=load_artifacts())


def test_inr_grouping():
    assert inr(1234567) == "Rs 12,34,567" and inr(-999) == "-Rs 999"


def test_regression_suite_all_pass():
    out = run_suite()
    assert out["summary"]["total"] == "30/30"


def test_month_words_do_not_match_inside_other_words(cp):
    p = offline_plan("Why did gross margin decline?", cp.ctx)
    assert p["month"] == cp.ctx["months"][-1]  # defaulted, not parsed from 'margin'/'decline'


def test_lowest_sorts_ascending(cp):
    r = cp.ask("Which city has the lowest net sales?")
    t = r.evidence.table
    assert t["net_sales"].is_monotonic_increasing


def test_validator_rejects_unknown_values(cp):
    bad = {**offline_plan("net sales in Mumbai", cp.ctx), "city": "Atlantis"}
    with pytest.raises(ValueError):
        validate_plan(bad, cp.ctx)
    with pytest.raises(ValueError):
        validate_plan({**offline_plan("net sales", cp.ctx), "operation": "run_sql"}, cp.ctx)


def test_missing_month_is_error_not_zero(cp):
    r = cp.ask("Net sales in Delhi NCR in May")
    assert r.status == "data_error" and "not treated as zero" in r.plan["message"]


def test_upload_validation():
    df = pd.read_csv(DATA_DIR / "sales.csv").head(50)
    with pytest.raises(DataError):
        validate_sales(df.drop(columns=["cost"]))
    bad = df.copy(); bad.loc[0, "units"] = -1
    with pytest.raises(DataError):
        validate_sales(bad)
    dup = pd.concat([df, df.head(1)])
    with pytest.raises(DataError):
        validate_sales(dup)


def test_live_mode_requires_consent(cp):
    r = cp.ask("Why did sales decline in Hyderabad?", mode="live", consent=False)
    assert r.status == "blocked"


class FakeLLM:
    """Mimics ClaudeClient; returns canned tool outputs."""
    def __init__(self, plan=None, explanation=None, fail=False):
        self._plan, self._exp, self.fail, self.usage = plan, explanation, fail, []

    def plan(self, q, ctx):
        if self.fail:
            raise TimeoutError
        return self._plan

    def explain(self, q, ev, passages):
        return self._exp


def _cp_with(llm):
    return Copilot(pd.read_csv(DATA_DIR / "sales.csv"), artifacts=load_artifacts(), llm=llm)


def test_live_plan_and_clean_explanation_used():
    plan = {"status": "ok", "operation": "explain", "metric": "net_sales", "city": "Hyderabad", "product": None,
            "channel": None, "family": None, "group_by": None, "month": "2028-03", "previous_month": "2028-02",
            "sort": "desc", "message": ""}
    exp = {"summary": "Heritage Edition drove most of the fall.", "actions": ["Check sleeve stock."],
           "limitation": "Observed, not causal.", "source_ids": ["POLICY_STOCKOUT"]}
    r = _cp_with(FakeLLM(plan, exp)).ask("why the dip in hyd?", mode="live", consent=True)
    assert r.explained_by in ("claude", "template")
    assert r.evidence.facts["top_product"] == "FlexFold Heritage Edition"


def test_live_explanation_with_digits_is_rejected():
    plan = {"status": "ok", "operation": "explain", "metric": "net_sales", "city": "Hyderabad", "product": None,
            "channel": None, "family": None, "group_by": None, "month": "2028-03", "previous_month": "2028-02",
            "sort": "desc", "message": ""}
    exp = {"summary": "Sales fell 34%.", "actions": [], "limitation": "", "source_ids": []}
    r = _cp_with(FakeLLM(plan, exp)).ask("why?", mode="live", consent=True)
    assert r.explained_by == "template" and any("digits" in w for w in r.warnings)


def test_live_plan_failure_falls_back_offline():
    r = _cp_with(FakeLLM(fail=True)).ask("Which city has the highest net sales?", mode="live", consent=True)
    assert r.status == "ok" and any("offline planner" in w for w in r.warnings)


def test_live_plan_with_bad_operation_is_rejected():
    plan = {"status": "ok", "operation": "drop_table", "metric": "net_sales", "city": None, "product": None,
            "channel": None, "family": None, "group_by": None, "month": None, "previous_month": None,
            "sort": "desc", "message": ""}
    r = _cp_with(FakeLLM(plan)).ask("Which city has the highest net sales?", mode="live", consent=True)
    assert r.plan["operation"] == "breakdown" and any("rejected" in w for w in r.warnings)


def test_claude_client_parses_tool_use():
    from neatnest.copilot.llm import ClaudeClient
    block = SimpleNamespace(type="tool_use", input={"summary": "ok", "actions": [], "limitation": "", "source_ids": []})
    resp = SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    c = ClaudeClient(client=fake, model="test-model")
    assert c.explain("q", {}, [])["summary"] == "ok" and c.usage[0]["input_tokens"] == 10
