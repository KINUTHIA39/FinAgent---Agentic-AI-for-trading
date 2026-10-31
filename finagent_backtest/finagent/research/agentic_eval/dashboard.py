"""Interactive adaptation-surface dashboard (localhost).

Reads ``data/agentic_eval/runs.parquet`` and serves an interactive view of the
agent's plan adaptation:

  * 3D adaptation surface  — defensiveness over regime x exposure (rotatable)
  * 2D heatmap             — same, with exact values
  * reproducibility        — defensiveness distribution per cell across K runs
  * posture decomposition  — which risk controls drive the defensiveness
  * reliability & cost      — LLM-success, latency, tokens, $/plan

Run:
    python -m finagent.research.agentic_eval.dashboard
    # then open http://127.0.0.1:8050
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "data" / "agentic_eval" / "runs.parquet"

REGIME_ORDER = ["bull-calm", "bull-stress", "bear-calm", "bear-stress"]   # severity ↑
EXPOSURE_ORDER = ["low", "mid", "near-full"]                             # exposure ↑

# Claude Sonnet pricing ($ per 1M tokens) for the cost panel.
PRICE_IN, PRICE_OUT = 3.0, 15.0


def load() -> pd.DataFrame:
    df = pd.read_parquet(RUNS)
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df["exposure"] = pd.Categorical(df["exposure"], EXPOSURE_ORDER, ordered=True)
    return df


def _matrix(df: pd.DataFrame, value: str, agg: str = "mean") -> np.ndarray:
    """4x3 (regime x exposure) matrix of an aggregated metric."""
    piv = (df.groupby(["regime", "exposure"], observed=True)[value]
             .agg(agg).unstack("exposure"))
    piv = piv.reindex(index=REGIME_ORDER, columns=EXPOSURE_ORDER)
    return piv.to_numpy(dtype=float)


def make_surface(df: pd.DataFrame) -> go.Figure:
    z = _matrix(df, "defensiveness_score", "mean")
    sd = _matrix(df, "defensiveness_score", "std")
    n = _matrix(df, "defensiveness_score", "count")
    hover = np.empty(z.shape, dtype=object)
    for i, r in enumerate(REGIME_ORDER):
        for j, e in enumerate(EXPOSURE_ORDER):
            hover[i, j] = (f"<b>{r} · {e}</b><br>defensiveness={z[i,j]:.3f}"
                           f"<br>SD={sd[i,j]:.3f} · n={0 if np.isnan(n[i,j]) else int(n[i,j])}")
    fig = go.Figure(go.Surface(
        x=list(range(3)), y=list(range(4)), z=z,
        colorscale="RdYlGn_r", cmin=0, cmax=max(0.5, np.nanmax(z) if np.isfinite(np.nanmax(z)) else 0.5),
        text=hover, hoverinfo="text",
        colorbar=dict(title="defensive"),
    ))
    fig.update_layout(
        title="Adaptation surface — defensiveness vs regime × exposure",
        scene=dict(
            xaxis=dict(title="exposure", tickvals=[0, 1, 2], ticktext=EXPOSURE_ORDER),
            yaxis=dict(title="regime (severity →)", tickvals=[0, 1, 2, 3], ticktext=REGIME_ORDER),
            zaxis=dict(title="defensiveness", range=[0, max(0.5, np.nanmax(z) if np.isfinite(np.nanmax(z)) else 0.5)]),
        ),
        margin=dict(l=0, r=0, t=40, b=0), height=520,
    )
    return fig


def make_heatmap(df: pd.DataFrame) -> go.Figure:
    z = _matrix(df, "defensiveness_score", "mean")
    fig = go.Figure(go.Heatmap(
        z=z, x=EXPOSURE_ORDER, y=REGIME_ORDER, colorscale="RdYlGn_r",
        zmin=0, zmax=max(0.5, np.nanmax(z) if np.isfinite(np.nanmax(z)) else 0.5),
        text=np.round(z, 3), texttemplate="%{text}", colorbar=dict(title="defensive"),
    ))
    fig.update_layout(title="Defensiveness (exact)", height=360,
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def make_box(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["cell"] = df["regime"].astype(str) + "\n" + df["exposure"].astype(str)
    order = [f"{r}\n{e}" for r in REGIME_ORDER for e in EXPOSURE_ORDER
             if ((df["regime"].astype(str) == r) & (df["exposure"].astype(str) == e)).any()]
    fig = go.Figure()
    for cell in order:
        sub = df[df["cell"] == cell]
        fig.add_trace(go.Box(y=sub["defensiveness_score"], name=cell, boxpoints="all",
                             jitter=0.4, marker_size=4))
    fig.update_layout(title="Reproducibility — defensiveness spread across K runs per cell",
                      showlegend=False, height=360, margin=dict(l=10, r=10, t=40, b=40),
                      yaxis_title="defensiveness")
    return fig


def make_posture(df: pd.DataFrame) -> go.Figure:
    """Mean posture indicators by regime — what the agent actually changes."""
    g = df.groupby("regime", observed=True).agg(
        max_position_pct=("max_position_pct", "mean"),
        reduce_only=("reduce_only", "mean"),
        circuit_breaker=("circuit_breaker", "mean"),
        flagged_ratio=("n_flagged_symbols", "mean"),
    ).reindex(REGIME_ORDER)
    fig = go.Figure()
    for col, color in [("reduce_only", "#d62728"), ("circuit_breaker", "#ff7f0e"),
                       ("flagged_ratio", "#9467bd"), ("max_position_pct", "#1f77b4")]:
        fig.add_trace(go.Bar(name=col, x=REGIME_ORDER, y=g[col]))
    fig.update_layout(title="Risk-posture decomposition by regime",
                      barmode="group", height=360, margin=dict(l=10, r=10, t=40, b=40),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _card(title: str, value: str, sub: str = "") -> html.Div:
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "26px", "fontWeight": "600"}),
        html.Div(sub, style={"fontSize": "11px", "color": "#aaa"}),
    ], style={"padding": "12px 18px", "background": "#1b1b1b", "borderRadius": "8px",
              "minWidth": "150px", "color": "#eee"})


def build_app() -> Dash:
    full = load()
    recall_opts = [{"label": "All recall", "value": "ALL"}] + [
        {"label": rc, "value": rc} for rc in sorted(full["recall_profile"].unique())]

    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Agentic Plan-Adaptation Surface", style={"color": "#eee"}),
        html.Div("Does the LLM planner adapt its risk posture to market regime, "
                 "portfolio exposure, and recalled memory? Higher = more defensive.",
                 style={"color": "#aaa", "marginBottom": "12px"}),
        html.Div(id="cards", style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                                    "marginBottom": "16px"}),
        html.Div([html.Span("Recall profile: ", style={"color": "#ccc"}),
                  dcc.Dropdown(id="recall", options=recall_opts, value="ALL",
                               clearable=False, style={"width": "240px", "display": "inline-block"})],
                 style={"marginBottom": "12px"}),
        html.Div([
            html.Div(dcc.Graph(id="surface"), style={"flex": "1.4"}),
            html.Div(dcc.Graph(id="heatmap"), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
        html.Div([
            html.Div(dcc.Graph(id="box"), style={"flex": "1"}),
            html.Div(dcc.Graph(id="posture"), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ], style={"background": "#111", "padding": "20px", "fontFamily": "system-ui",
              "minHeight": "100vh"})

    @app.callback(
        Output("surface", "figure"), Output("heatmap", "figure"),
        Output("box", "figure"), Output("posture", "figure"), Output("cards", "children"),
        Input("recall", "value"),
    )
    def update(recall):
        df = full if recall == "ALL" else full[full["recall_profile"] == recall]
        n_plans = len(df)
        llm = df["llm_success"].mean() if n_plans else 0
        lat = df["latency_ms"].mean() if n_plans else 0
        tok_out = df["tokens_out"].mean() if n_plans else 0
        tok_in = df["tokens_in"].mean() if n_plans else 0
        cost = (tok_in * PRICE_IN + tok_out * PRICE_OUT) / 1e6
        cards = [
            _card("plans", f"{n_plans}", f"{df['scenario'].nunique()} cells"),
            _card("LLM-plan success", f"{llm:.0%}", f"{int((~df['llm_success']).sum())} fallbacks"),
            _card("latency / plan", f"{lat/1000:.1f}s", "vs ~0s fallback"),
            _card("output tokens", f"{tok_out:.0f}", f"in {tok_in:.0f}"),
            _card("est. cost / plan", f"${cost:.3f}", f"~${cost*n_plans:.2f} total"),
        ]
        return (make_surface(df), make_heatmap(df), make_box(df), make_posture(df), cards)

    return app


if __name__ == "__main__":
    print(f"Reading {RUNS}")
    build_app().run(debug=False, host="127.0.0.1", port=8050)
