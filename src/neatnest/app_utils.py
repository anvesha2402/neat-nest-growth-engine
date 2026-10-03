"""Shared helpers for the Streamlit pages."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .config import ASSET_DIR, DATA_DIR, GRID, INK, INK_MUTED, MODEL_DIR, REPORT_DIR, SERIES

pio.templates["neatnest"] = go.layout.Template(layout=go.Layout(
    font={"family": "Inter, system-ui, sans-serif", "color": INK, "size": 13},
    colorway=SERIES, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis={"gridcolor": GRID, "zeroline": False, "linecolor": GRID},
    yaxis={"gridcolor": GRID, "zeroline": False, "linecolor": GRID},
    legend={"orientation": "h", "y": -0.2}, margin={"l": 10, "r": 10, "t": 50, "b": 10},
    hoverlabel={"bgcolor": "white", "font": {"color": INK}},
))
pio.templates.default = "plotly_white+neatnest"
TEMPLATE = "plotly_white+neatnest"

CSS = """
<style>
.block-container {padding-top: 3.4rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.nn-kicker {color:#C47A4F; font-weight:600; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem;}
.nn-card {border:1px solid #e6e1d8; border-radius:10px; padding:14px 16px; background:rgba(244,235,221,.35); height:100%;}
.nn-card h4 {margin:0 0 6px 0; font-size:1rem;}
.nn-card p {margin:0; color:#6b665f; font-size:.9rem;}
.nn-note {border-left:3px solid #C47A4F; padding:6px 12px; color:#6b665f; font-size:.88rem; background:rgba(244,235,221,.35);}
</style>
"""


def page_setup(title: str, icon: str = "🧺"):
    st.set_page_config(page_title=f"{title} · Neat Nest", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def header(kicker: str, title: str, lede: str):
    st.markdown(f'<div class="nn-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(lede)


def card(title: str, body: str):
    st.markdown(f'<div class="nn-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="nn-note">{text}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building the Neat Nest data universe and models (first run only)...")
def ensure_built() -> bool:
    needed = [REPORT_DIR / "results.json", MODEL_DIR / "propensity.joblib", MODEL_DIR / "lapse.joblib",
              DATA_DIR / "sales.csv"]
    if not all(p.exists() for p in needed):
        from .pipeline import run
        run(make_figures=False)
    return True


@st.cache_data
def results() -> dict:
    ensure_built()
    return json.loads((REPORT_DIR / "results.json").read_text())


@st.cache_data
def data(name: str) -> pd.DataFrame:
    ensure_built()
    return pd.read_csv(DATA_DIR / f"{name}.csv")


@st.cache_data
def report_csv(name: str, index_col=None) -> pd.DataFrame:
    ensure_built()
    return pd.read_csv(REPORT_DIR / f"{name}.csv", index_col=index_col)


@st.cache_resource
def model(name: str):
    import joblib
    ensure_built()
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def frame(d: dict) -> pd.DataFrame:
    """results.json stores DataFrames as {index: {col: val}}."""
    return pd.DataFrame(d).T


def asset(name: str) -> str:
    return str(ASSET_DIR / name)


def fig_layout(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height,
                      title={"text": title, "x": 0, "font": {"size": 15}} if title else None)
    return fig


def muted() -> str:
    return INK_MUTED
