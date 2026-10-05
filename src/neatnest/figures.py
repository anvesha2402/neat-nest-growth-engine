"""Static figures for the project brief (matplotlib, brand-adjacent validated palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIG_DIR, GRID, INK, INK_MUTED, PALETTE, SERIES  # noqa: E402

BG = "#fcfbf8"


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=170)
    plt.close(fig)


def fig_market(R):
    m = R["market"]
    stages = [("Urban households", m["urban_hh_m"]), ("Own a washing machine", m["urban_wm_hh_m"]),
              ("SAM: reachable online,\npremium-capable", m["sam_hh_m"]),
              ("SAM annual unit demand\n(4-year replacement)", m["sam_units_per_year_m"])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [v for _, v in stages], color=SERIES[0], height=0.62)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v + 1.5, yi, f"{v:,.1f} M", va="center", color=INK, fontweight="bold")
    ax.set_yticks(y, [s for s, _ in stages]); ax.set_xlim(0, 150); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Millions of households / units")
    ax.set_title("126 M urban households narrow to a ~2.1 M-unit annual SAM")
    _save(fig, "01_market_funnel")


def fig_unit_economics(R):
    ue = pd.DataFrame(R["unit_economics"]).T if isinstance(R["unit_economics"], dict) else R["unit_economics"]
    row = ue.loc[("FLEX", "D2C")]
    steps = [("MRP", row["price"]), ("GST (5%)", -row["gst"]), ("COGS", -row["cogs"]), ("Packaging", -row["packaging"]),
             ("Shipping", -row["shipping"]), ("Gateway fee", -row["channel_fees"]), ("Returns", -row["returns"]),
             ("CM2", row["cm2"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    run = 0
    for i, (lab, v) in enumerate(steps):
        if i == 0 or lab == "CM2":
            ax.bar(i, v, color=SERIES[0] if i == 0 else SERIES[2], width=0.62)
            run = v if i == 0 else run
            ax.text(i, v + 25, f"Rs {v:,.0f}", ha="center", color=INK, fontweight="bold", fontsize=9)
        else:
            ax.bar(i, -v, bottom=run + v, color=SERIES[1], width=0.62)
            ax.text(i, run + 25, f"-{-v:,.0f}", ha="center", color=INK_MUTED, fontsize=9)
            run += v
    ax.set_xticks(range(len(steps)), [s for s, _ in steps], fontsize=9)
    ax.set_ylabel("Rs per unit"); ax.grid(axis="x", visible=False)
    ax.set_title(f"FlexFold on D2C keeps Rs {row['cm2']:,.0f} ({row['cm2_pct'] * 100:.0f}% of net revenue) before acquisition cost")
    _save(fig, "02_unit_economics_flexfold_d2c")


def fig_som(R):
    som = pd.DataFrame(R["som_plan"]).T if isinstance(R["som_plan"], dict) else R["som_plan"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(som))
    ax.bar(x - 0.2, som["net_revenue_cr"], 0.38, color=SERIES[0], label="Net revenue")
    ax.bar(x + 0.2, som["cm3_cr"], 0.38, color=SERIES[2], label="Contribution after marketing (CM3)")
    for xi, (a, b) in enumerate(zip(som["net_revenue_cr"], som["cm3_cr"])):
        ax.text(xi - 0.2, a + 0.3, f"{a:.1f}", ha="center", color=INK, fontsize=9)
        ax.text(xi + 0.2, b + 0.3, f"{b:.1f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [f"{y}\n{int(u):,} units" for y, u in zip(som.index, som["core_units"])])
    ax.set_ylabel("Rs crore"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_title("Three-year plan: revenue and contribution (Rs crore)")
    _save(fig, "03_som_plan")


def fig_vw(vw):
    c = vw["curves"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, colr, ls in [("Too cheap", SERIES[2], "--"), ("Cheap", SERIES[2], "-"),
                          ("Expensive", SERIES[1], "-"), ("Too expensive", SERIES[1], "--")]:
        ax.plot(c["price"], c[col] * 100, color=colr, ls=ls, lw=2, label=col)
    ax.axvspan(vw["PMC"], vw["PME"], color=PALETTE["sand"], alpha=0.35, lw=0)
    for k, yy in (("OPP", 93), ("IPP", 84)):
        ax.axvline(vw[k], color=INK_MUTED, lw=1, ls=":")
        ax.text(vw[k] + 15, yy, f"{k} Rs {vw[k]:,.0f}", color=INK, fontsize=8.5)
    ax.axvline(1799, color=SERIES[0], lw=2)
    ax.text(1799 + 15, 70, "FlexFold\nRs 1,799", color=SERIES[0], fontsize=9, fontweight="bold")
    ax.set_xlim(500, 3200); ax.set_ylim(0, 100); ax.set_xlabel("Price (Rs)"); ax.set_ylabel("% of respondents")
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_title(f"Van Westendorp: acceptable range Rs {vw['PMC']:,.0f}-{vw['PME']:,.0f} (shaded), n=600")
    _save(fig, "04_van_westendorp")


def fig_kano(R):
    k = pd.DataFrame(R["kano"]).T if isinstance(R["kano"], dict) else R["kano"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axvline(0.5, color=GRID, lw=1.2); ax.axhline(0.5, color=GRID, lw=1.2)
    for f, r in k.iterrows():
        ax.scatter(-r["worse"], r["better"], s=90, color=SERIES[0], edgecolor=BG, lw=2, zorder=3)
        off = {"three_compartments": (8, -14), "ventilated_antimicrobial": (-60, -18)}.get(f, (6, 5))
        ax.annotate(f.replace("_", " "), (-r["worse"], r["better"]), xytext=off, textcoords="offset points",
                    fontsize=8.5, color=INK)
    for x, y, t in [(0.04, 0.95, "Attractive"), (0.62, 0.95, "Performance"), (0.62, 0.05, "Must-be"),
                    (0.04, 0.05, "Indifferent")]:
        ax.text(x, y, t, color=INK_MUTED, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Dissatisfaction if absent (|worse| coefficient)"); ax.set_ylabel("Satisfaction if present (better coefficient)")
    ax.set_title("Kano: wheels and compartments perform; the sensor app can wait")
    _save(fig, "05_kano")


def fig_factors(R, fa):
    L = fa["loadings"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ev = fa["eigenvalues"]
    axes[0].plot(range(1, len(ev) + 1), ev, marker="o", color=SERIES[0], lw=2, ms=7)
    axes[0].axhline(1, color=SERIES[1], ls="--", lw=1.2)
    axes[0].set_xlabel("Factor"); axes[0].set_ylabel("Eigenvalue"); axes[0].set_title("Scree plot: 3 factors above 1")
    im = axes[1].imshow(L.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(3), ["Digital\nEngagement", "Spending\nPower", "Loyalty &\nPromo"], fontsize=8.5)
    axes[1].set_yticks(range(len(L)), [i.replace("_", " ") for i in L.index], fontsize=8.5)
    axes[1].grid(False)
    for i in range(L.shape[0]):
        for j in range(3):
            v = L.values[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > 0.6 else INK)
    axes[1].set_title(f"Varimax loadings (KMO {fa['kmo']:.2f}, {fa['cumulative_pct']:.1f}% variance)")
    _save(fig, "06_factor_analysis")


def fig_personas(R):
    p = pd.DataFrame(R["personas"]).T if isinstance(R["personas"], dict) else R["personas"]
    cols = ["Digital Engagement", "Spending Power", "Loyalty & Promo Response"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(p)); w = 0.26
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * w, p[c], w - 0.02, color=SERIES[i], label=c)
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xticks(x, [f"{n.split(' (')[0]}\n(n={int(k)})" for n, k in zip(p.index, p["n"])], fontsize=8.5)
    ax.set_ylabel("Mean factor score (SD units)"); ax.legend(fontsize=8.5, ncol=3, loc="upper center",
                                                             bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", visible=False)
    ax.set_title("Students: reachable but cash-poor. Seniors: loyal but offline")
    _save(fig, "07_persona_factors")


def fig_propensity(R, ev):
    t = pd.DataFrame(R["propensity"]["coefficients"]).T if isinstance(R["propensity"]["coefficients"], dict) \
        else R["propensity"]["coefficients"]
    t = t.sort_values("odds_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    y = np.arange(len(t))
    colors = [SERIES[0] if s else "#b9b3a8" for s in t["significant"]]
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colors, lw=2)
    ax.scatter(t["odds_ratio"], y, color=colors, s=50, zorder=3)
    ax.axvline(1, color=INK_MUTED, lw=1, ls="--")
    ax.set_yticks(y, [i.replace("_", " ").replace("source ", "src: ") for i in t.index], fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("Odds ratio (95% CI, log scale)")
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.8, 1, 1.5, 2, 3])); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_title("Purchase drivers (blue = p < 0.05)")
    lift = ev["lift"]
    axes[1].bar(lift.index.astype(int), lift["conv_rate"] * 100, color=SERIES[0], width=0.7)
    axes[1].axhline(ev["base_rate"] * 100, color=SERIES[1], ls="--", lw=1.2)
    axes[1].text(10.4, ev["base_rate"] * 100 + 0.6, "average", color=SERIES[1], ha="right", fontsize=8.5)
    axes[1].set_xticks(range(1, 11)); axes[1].set_xlabel("Propensity decile (1 = highest score)")
    axes[1].set_ylabel("Conversion rate (%)"); axes[1].grid(axis="x", visible=False)
    axes[1].set_title(f"Holdout: top 3 deciles hold {lift['cum_capture'].iloc[2] * 100:.0f}% of buyers")
    _save(fig, "08_propensity")


def fig_lapse(R, lm):
    imp = lm["importance"].head(8)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].barh(range(len(imp)), imp.values, color=SERIES[0], height=0.6)
    axes[0].set_yticks(range(len(imp)), [i.replace("_", " ") for i in imp.index], fontsize=8.5)
    axes[0].set_xlabel("Permutation importance (drop in AUC)"); axes[0].grid(axis="y", visible=False)
    axes[0].set_title("What predicts lapse (Random Forest)")
    for name, (fpr, tpr) in lm["roc"].items():
        auc = lm["metrics"].loc[name, "auc"]
        axes[1].plot(fpr, tpr, lw=2, color=SERIES[0] if "Logistic" in name else SERIES[1],
                     label=f"{name} (AUC {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color=GRID, lw=1)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate"); axes[1].legend(fontsize=8.5)
    axes[1].set_title("The simpler model matches the forest")
    _save(fig, "09_lapse_model")


def fig_segments(R, sg):
    p = sg["profile"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, (name, r) in enumerate(p.iterrows()):
        ax.scatter(r["lapse_rate"] * 100, r["value_12m"], s=r["customers"] / 4, color=SERIES[i], alpha=0.9,
                   edgecolor=BG, lw=2)
        ax.annotate(f"{name}\n{r['share'] * 100:.0f}% of owners", (r["lapse_rate"] * 100, r["value_12m"]),
                    xytext=(10, -4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Lapse rate, months 7-12 (%)"); ax.set_ylabel("Mean 12-month value (Rs)")
    ax.set_xlim(0, 80)
    ax.set_title("Four owner segments: protect Advocates, rescue the Service-scarred")
    _save(fig, "10_segments")


def fig_sales(frames):
    s = frames["sales"].assign(month=lambda d: d["date"].str[:7])
    pv = s.pivot_table(index="month", columns="city", values="net_sales", aggfunc="sum").fillna(0) / 1e5
    fig, ax = plt.subplots(figsize=(9, 3.8))
    order = ["Mumbai", "Pune", "Bengaluru", "Hyderabad", "Delhi NCR"]
    for i, c in enumerate(order):
        ax.plot(range(len(pv)), pv[c], color=SERIES[i], lw=2, marker="o", ms=3.5, label=c)
    ax.set_xticks(range(len(pv)), [m[2:] for m in pv.index], fontsize=8.5)
    ax.set_ylabel("Net sales (Rs lakh)"); ax.legend(ncol=5, fontsize=8.5, loc="upper left")
    ax.annotate("Hyderabad, Mar 2028:\nHeritage sleeve stock-out", xy=(len(pv) - 1, pv["Hyderabad"].iloc[-1]),
                xytext=(len(pv) - 4.6, pv["Hyderabad"].iloc[-1] - 1.4), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK_MUTED})
    ax.set_title("Simulated launch year: monthly net sales by city")
    _save(fig, "11_sales_by_city")


def fig_explain(R):
    import pandas as pd  # local import keeps module import light
    from .config import DATA_DIR
    from .copilot.engine import Copilot
    cp = Copilot(pd.read_csv(DATA_DIR / "sales.csv"))
    r = cp.ask("Why did sales decline in Hyderabad?")
    t = r.evidence.table.sort_values("change")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["prev_sales"] / 1e3, 0.38, color=SERIES[0], label=r.plan["previous_month"])
    ax.barh(y - 0.2, t["cur_sales"] / 1e3, 0.38, color=SERIES[1], label=r.plan["month"])
    ax.set_yticks(y, t.index, fontsize=8.5); ax.invert_yaxis(); ax.legend(fontsize=8.5)
    ax.set_xlabel("Net sales (Rs thousand)"); ax.grid(axis="y", visible=False)
    ax.set_title("Copilot evidence: Heritage Edition leads Hyderabad's March fall")
    _save(fig, "12_copilot_hyderabad")


def box(ax, x, y, w, h, title, sub, color, title_size=9.5, sub_size=8):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="none", alpha=0.14))
    ax.add_patch(plt.Rectangle((x, y + h - 0.08), w, 0.08, fc=color, ec="none"))
    ax.text(x + 0.12, y + h - 0.3, title, fontsize=title_size, fontweight="bold", color=INK, va="top")
    if sub:
        ax.text(x + 0.12, y + h - 0.62, sub, fontsize=sub_size, color=INK_MUTED, va="top", linespacing=1.35)


def pill(ax, x, y, w, h, text, color):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                          fc=color, ec="none", alpha=0.16))
    ax.text(x + w / 2, y + h / 2, text, fontsize=7.6, color=INK, ha="center", va="center",
            linespacing=1.25, fontweight="bold")


def band_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=8, color=INK_MUTED, fontweight="bold", ha="left", va="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": BG, "ec": GRID, "lw": 0.8})


def vline(ax, x, y0, y1, color=INK_MUTED, lw=1.3):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops={"arrowstyle": "->", "color": color, "lw": lw})


def draw_architecture(ax):
    """Draws the full 5-band Neat Nest reference architecture onto `ax`. Shared by the
    reproducible pipeline figure (reports/figures/00_architecture.png) and
    scripts/make_diagrams.py (assets/diagrams/, used by the app, README and brief)."""
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 11.1)
    ax.text(0.1, 10.85, "Neat Nest growth engine: reference architecture", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.1, 10.5, "One simulated launch-year data universe, four analytics layers and a guarded AI agent, "
                       "served through one Streamlit app.", fontsize=9, color=INK_MUTED)

    # Band A - data & knowledge
    band_label(ax, 0.1, 10.0, "DATA & KNOWLEDGE")
    data_boxes = [("survey.csv", "n=600"), ("campaign.csv", "n=5,000"), ("customers.csv", "n=8,000"),
                  ("sales.csv", "8,513 rows"), ("knowledge/*.md", "brand book, PRD,\nKPI rules, GTM playbook")]
    dw, gap = 2.36, 0.12
    for i, (t, s) in enumerate(data_boxes):
        box(ax, 0.1 + i * (dw + gap), 8.75, dw, 1.05, t, s, PALETTE["charcoal"], title_size=8.6, sub_size=7.4)

    for i in range(4):
        vline(ax, 0.1 + i * (dw + gap) + dw / 2, 8.35, 8.75)
    vline(ax, 0.1 + 4 * (dw + gap) + dw / 2, 5.55, 8.75, color=SERIES[3])  # knowledge feeds the agent band directly

    # Band B - analytics & ML (4 columns)
    band_label(ax, 0.1, 8.05, "ANALYTICS & ML")
    cw = (12.6 - 3 * 0.15) / 4
    cols = [
        ("1  Brand & product", "STP · JTBD · Kano\nconcept matrix · CBBE\npackaging · RACE", SERIES[0]),
        ("2  Customer factors", "Exploratory factor analysis\n(KMO, Bartlett, varimax)\nVan Westendorp, Kano", SERIES[2]),
        ("3  Acquisition", "Binary logistic regression\n(statsmodels + sklearn)\ndecile lift, odds ratios", SERIES[1]),
        ("4  Retention & value", "Logistic vs Random Forest\nOLS/Ridge/Lasso, K-Means\nCLV regression", SERIES[4]),
    ]
    for i, (t, s, c) in enumerate(cols):
        box(ax, 0.1 + i * (cw + 0.15), 6.15, cw, 1.7, t, s, c)
        vline(ax, 0.1 + i * (cw + 0.15) + cw / 2, 5.55, 6.15)

    # Band C - AI agent layer (Nest Copilot)
    band_label(ax, 0.1, 5.35, "AI AGENT LAYER")
    box(ax, 0.1, 3.75, 12.6, 1.55, "5  Nest Copilot  (bounded agent + RAG)", None, SERIES[3])
    sub_steps = ["Guardrails\n(blocklist,\nscope check)", "Planner\n(offline keywords\nor Claude, JSON)",
                 "9 approved\nPython tools\n(no raw SQL)", "RAG retrieval\n(TF-IDF +\ntitle bonus)",
                 "Explainer\n(template or Claude,\ncited IDs only)"]
    sw = (12.2 - 4 * 0.1) / 5
    for i, s in enumerate(sub_steps):
        sx = 0.3 + i * (sw + 0.1)
        ax.add_patch(plt.Rectangle((sx, 3.88), sw, 0.95, fc="white", ec=SERIES[3], lw=1.1))
        ax.text(sx + sw / 2, 4.355, s, fontsize=7.3, color=INK, ha="center", va="center", linespacing=1.25)
        if i < 4:
            ax.annotate("", xy=(sx + sw + 0.1, 4.355), xytext=(sx + sw, 4.355),
                        arrowprops={"arrowstyle": "->", "color": SERIES[3], "lw": 1.1})
    vline(ax, 6.4, 3.35, 3.75)

    # Band D - application layer
    band_label(ax, 0.1, 3.05, "APPLICATION LAYER")
    box(ax, 0.1, 1.85, 8.7, 1.1, "Streamlit app (8 pages)", "Home + Brand/Product, Market, Factors, Propensity, "
        "Retention, Copilot, Methods, Architecture", PALETTE["indigo"], title_size=9.5, sub_size=7.6)
    box(ax, 9.0, 1.85, 3.7, 1.1, "Claude API (optional)", "Live planning + explanation;\nfully offline by default",
        PALETTE["olive"], title_size=9.2, sub_size=7.6)
    vline(ax, 4.45, 1.45, 1.85)

    # Band E - business outcomes
    band_label(ax, 0.1, 1.15, "BUSINESS OUTCOMES")
    outcomes = [("SAM\n8.5M households", SERIES[0]), ("Hero price\nRs 1,799", SERIES[2]),
                ("Y3 net revenue\nRs 18.3 cr plan", SERIES[1]), ("Lapse AUC\n0.78", SERIES[4]),
                ("Copilot suite\n30/30", SERIES[3])]
    ow, ogap = 2.36, 0.12
    for i, (t, c) in enumerate(outcomes):
        pill(ax, 0.1 + i * (ow + ogap), 0.15, ow, 0.85, t, c)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 11.1))
    draw_architecture(ax)
    _save(fig, "00_architecture")


def make_all(R, frames, fa, vw, ev, lm, sg):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    fig_architecture(); fig_market(R); fig_unit_economics(R); fig_som(R); fig_vw(vw); fig_kano(R)
    fig_factors(R, fa); fig_personas(R); fig_propensity(R, ev); fig_lapse(R, lm); fig_segments(R, sg)
    fig_sales(frames); fig_explain(R)
