"""Approved, read-only analytics tools. Python computes every number."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED = ["line_id", "date", "city", "product", "family", "channel", "units", "net_sales", "cost"]
METRICS = ["net_sales", "gross_profit", "gross_margin", "units", "discount"]
DIMENSIONS = ["city", "product", "channel", "family"]
MAX_ROWS, MAX_MB = 200_000, 10


class DataError(ValueError):
    """Raised when uploaded data fails validation or a requested period is missing."""


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {', '.join(missing)}")
    if len(df) > MAX_ROWS:
        raise DataError(f"Too many rows ({len(df):,}); the limit is {MAX_ROWS:,}")
    if df[REQUIRED].isna().any().any():
        raise DataError("Required columns contain blank values")
    if df["line_id"].duplicated().any():
        raise DataError("line_id values must be unique")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise DataError("Some dates could not be parsed (expected YYYY-MM-DD)")
    for c in ["units", "net_sales", "cost"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any() or (out[c] < 0).any() or not np.isfinite(out[c]).all():
            raise DataError(f"{c} must be a finite, non-negative number")
    if (out["units"] % 1 != 0).any():
        raise DataError("units must be whole numbers")
    if "discount_pct" in out and ((out["discount_pct"] < 0) | (out["discount_pct"] > 1)).any():
        raise DataError("discount_pct must be between 0 and 1")
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["gross_profit"] = out["net_sales"] - out["cost"]
    if "discount_pct" not in out:
        out["discount_pct"] = 0.0
    return out


def _agg(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    g = df.groupby(by) if by else df.assign(_all="All").groupby("_all")
    out = g.agg(net_sales=("net_sales", "sum"), gross_profit=("gross_profit", "sum"),
                units=("units", "sum"),
                discount=("discount_pct", lambda s: float(np.average(s, weights=df.loc[s.index, "units"]))))
    out["gross_margin"] = np.where(out["net_sales"] > 0, out["gross_profit"] / out["net_sales"] * 100, np.nan)
    out["discount"] = out["discount"] * 100
    return out


@dataclass
class Evidence:
    title: str
    headline: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None
    chart: dict | None = None
    assumptions: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # structured numbers used by the template explainer


def apply_filters(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    for dim in DIMENSIONS:
        if plan.get(dim):
            df = df[df[dim] == plan[dim]]
    return df


def require_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    sub = df[df["month"] == month]
    if sub.empty:
        raise DataError(f"No records for {month} in the selected scope; it is not treated as zero.")
    return sub


def op_total(df, plan):
    sub = apply_filters(df, plan)
    if plan.get("month"):
        sub = require_month(sub, plan["month"])
    if sub.empty:
        raise DataError("No records match those filters.")
    a = _agg(sub).iloc[0]
    m = plan["metric"]
    return Evidence(title=f"{m.replace('_', ' ').title()}", headline={m: float(a[m])},
                    table=_agg(sub).round(2), facts={"value": float(a[m])})


def op_breakdown(df, plan):
    sub = apply_filters(df, plan)
    if plan.get("month"):
        sub = require_month(sub, plan["month"])
    if sub.empty:
        raise DataError("No records match those filters.")
    t = _agg(sub, [plan["group_by"]])
    asc = plan.get("sort") == "asc"
    t = t.sort_values(plan["metric"], ascending=asc)
    top = t.index[0]
    return Evidence(title=f"{plan['metric'].replace('_', ' ').title()} by {plan['group_by']}",
                    headline={"leader": top, plan["metric"]: float(t.iloc[0][plan["metric"]])},
                    table=t.round(2), chart={"type": "bar", "x": plan["group_by"], "y": plan["metric"]},
                    facts={"leader": top, "value": float(t.iloc[0][plan["metric"]]), "ascending": asc})


def op_trend(df, plan):
    sub = apply_filters(df, plan)
    if sub.empty:
        raise DataError("No records match those filters.")
    t = _agg(sub, ["month"])
    return Evidence(title=f"Monthly {plan['metric'].replace('_', ' ')}", table=t.round(2),
                    chart={"type": "line", "x": "month", "y": plan["metric"]},
                    headline={"first": t.index[0], "last": t.index[-1]},
                    facts={"peak": t[plan["metric"]].idxmax(), "peak_value": float(t[plan["metric"]].max())})


def op_compare(df, plan):
    sub = apply_filters(df, plan)
    cur, prev = require_month(sub, plan["month"]), require_month(sub, plan["previous_month"])
    m = plan["metric"]
    by = [plan["group_by"]] if plan.get("group_by") else None
    a, b = _agg(prev, by), _agg(cur, by)
    t = pd.DataFrame({plan["previous_month"]: a[m], plan["month"]: b[m]}).fillna(0)
    t["change"] = t[plan["month"]] - t[plan["previous_month"]]
    if m in ("gross_margin", "discount"):
        t["change_unit"] = "pp"
    else:
        t["change_pct"] = np.where(t[plan["previous_month"]] > 0, t["change"] / t[plan["previous_month"]] * 100, np.nan)
    tot_prev, tot_cur = float(_agg(prev).iloc[0][m]), float(_agg(cur).iloc[0][m])
    return Evidence(title=f"{m.replace('_', ' ').title()}: {plan['previous_month']} vs {plan['month']}",
                    headline={plan["previous_month"]: tot_prev, plan["month"]: tot_cur,
                              "change_pct": (tot_cur - tot_prev) / tot_prev * 100 if tot_prev and m not in ("gross_margin", "discount") else None},
                    table=t.sort_values("change").round(2),
                    chart={"type": "grouped", "x": plan.get("group_by") or "period", "y": m},
                    facts={"prev": tot_prev, "cur": tot_cur})


def op_explain(df, plan):
    """Decline diagnosis: split the change by product and channel, units vs price."""
    sub = apply_filters(df, plan)
    cur, prev = require_month(sub, plan["month"]), require_month(sub, plan["previous_month"])
    rows = []
    for prod in sorted(set(cur["product"]) | set(prev["product"])):
        p, c = prev[prev["product"] == prod], cur[cur["product"] == prod]
        pu, cu = p["units"].sum(), c["units"].sum()
        rows.append({"product": prod, "prev_sales": p["net_sales"].sum(), "cur_sales": c["net_sales"].sum(),
                     "prev_units": pu, "cur_units": cu,
                     "prev_rev_per_unit": p["net_sales"].sum() / pu if pu else np.nan,
                     "cur_rev_per_unit": c["net_sales"].sum() / cu if cu else np.nan})
    t = pd.DataFrame(rows).set_index("product")
    t["change"] = t["cur_sales"] - t["prev_sales"]
    t["units_change_pct"] = np.where(t["prev_units"] > 0, (t["cur_units"] - t["prev_units"]) / t["prev_units"] * 100, np.nan)
    t = t.sort_values("change")
    ch = pd.DataFrame({"prev": _agg(prev, ["channel"])["net_sales"], "cur": _agg(cur, ["channel"])["net_sales"]}).fillna(0)
    ch["change"] = ch["cur"] - ch["prev"]
    disc_prev, disc_cur = float(_agg(prev).iloc[0]["discount"]), float(_agg(cur).iloc[0]["discount"])
    tp, tc = prev["net_sales"].sum(), cur["net_sales"].sum()
    top = t.index[0]
    scope = ", ".join(str(plan[d]) for d in DIMENSIONS if plan.get(d)) or "All business"
    return Evidence(
        title=f"Why did net sales change in {scope}? ({plan['previous_month']} to {plan['month']})",
        headline={plan["previous_month"]: tp, plan["month"]: tc, "change_pct": (tc - tp) / tp * 100 if tp else None},
        table=t.round(2), chart={"type": "grouped_product"},
        facts={"scope": scope, "prev": tp, "cur": tc, "change": tc - tp, "top_product": top,
               "top_change": float(t.loc[top, "change"]), "top_units_pct": float(t.loc[top, "units_change_pct"]),
               "share_of_decline": float(t.loc[top, "change"] / (tc - tp) * 100) if tc < tp else None,
               "channel_table": ch.round(2), "discount_prev": disc_prev, "discount_cur": disc_cur})


def find_largest_decline(df: pd.DataFrame, month: str, previous: str) -> str | None:
    a = df[df["month"] == previous].groupby("city")["net_sales"].sum()
    b = df[df["month"] == month].groupby("city")["net_sales"].sum()
    chg = ((b - a) / a).dropna()
    chg = chg[chg < 0]
    return None if chg.empty else str(chg.idxmin())
