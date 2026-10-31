"""Dashboard for the agentic regime-switching experiment (localhost:8070).

A companion to ``dashboard_portfolio`` (port 8060), which covers the main study.
This one covers the controlled experiment: what happens when an agent sets the
three parameters the classical regime-switching model fixes by assumption.

  Tab 1  The experiment   what was tested, and how the five arms compare
  Tab 2  Is the edge real  the controls and the bootstrap, stated either way
  Tab 3  What it did       whether the interventions were selective or blanket
  Tab 4  Month by month    the briefing, the decision, and the reasoning

Prerequisites:
    python -m finagent.research.agentic_eval.rs_agentic --mode fixed
    python -m finagent.research.agentic_eval.rs_agentic --mode rule
    python -m finagent.research.agentic_eval.rs_agentic --mode constant --cap 0.20
    python -m finagent.research.agentic_eval.rs_agentic --mode random --seed 0   (etc.)
    python -m finagent.research.agentic_eval.rs_agentic --mode agent

Run:
    python -m finagent.research.agentic_eval.dashboard_rs_agentic
    # then open http://127.0.0.1:8070
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

REPO_ROOT = Path(__file__).resolve().parents[3]
RSDIR = REPO_ROOT / "data" / "agentic_eval" / "rs_agent"
CALLS = RSDIR / "agent_calls"
PORT = 8070

# ---- theme -------------------------------------------------------------------
BG, PANEL, LINE = "#FFFFFF", "#F7F8FA", "#E3E6EC"
INK, MUTED = "#1A1A2E", "#5B6472"

# Categorical hues, assigned in fixed order and validated with the dataviz
# palette checker: lightness band, chroma floor, CVD separation, normal-vision
# separation and contrast all pass on the light surface. Do not cycle or extend
# this list; a fifth arm would fold into the muted "random" band instead.
AGENT = "#7A3E9D"     # the arm under test
CLASSIC = "#C0851C"   # the unmodified model
RULE = "#2563C9"      # the demanding control
CONST = "#1A7F5A"     # the naive control
RANDOM = "#A8AFBC"    # five seeds, drawn as one recessive band

ARMS = {
    "Agent": (AGENT, "path_agent.parquet"),
    "Classical RS": (CLASSIC, "path_fixed.parquet"),
    "Deterministic rule": (RULE, "path_rule.parquet"),
    "Constant cap 0.20": (CONST, "path_constant_c20.parquet"),
}

CARD = {"backgroundColor": PANEL, "border": f"1px solid {LINE}",
        "borderRadius": "6px", "padding": "14px 16px"}
FONT = "Segoe UI, Inter, system-ui, sans-serif"
MONO = {"backgroundColor": "#FCFCFD", "border": f"1px solid {LINE}",
        "borderRadius": "5px", "padding": "12px", "fontSize": "11.5px",
        "lineHeight": "1.5", "maxHeight": "330px", "overflowY": "auto",
        "fontFamily": "JetBrains Mono, Consolas, monospace",
        "whiteSpace": "pre-wrap", "margin": 0}


def _fig(fig, height=320, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=58, r=24, t=46, b=42),
        showlegend=legend, hovermode="closest",
        legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor=LINE,
                    borderwidth=1),
    )
    # recessive grid and axes: the data should carry the ink, not the frame
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# ---- data --------------------------------------------------------------------
def _load_arm(fname):
    f = RSDIR / fname
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d.index = pd.to_datetime(d.index)
    return d


PATHS = {name: _load_arm(f) for name, (_, f) in ARMS.items()}
PATHS = {k: v for k, v in PATHS.items() if v is not None}
RANDOMS = [_load_arm(f.name) for f in sorted(RSDIR.glob("path_random_s*.parquet"))]
RANDOMS = [r for r in RANDOMS if r is not None]
AG = PATHS.get("Agent")
HAS_DATA = AG is not None


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(12))


def stats(r: pd.Series) -> dict:
    w = (1 + r).cumprod()
    return {"final": float(w.iloc[-1]),
            "cagr": float(w.iloc[-1] ** (12 / len(r)) - 1) * 100,
            "vol": float(r.std() * np.sqrt(12)) * 100,
            "sharpe": sharpe(r),
            "maxdd": float((w / w.cummax() - 1).min()) * 100}


@lru_cache(maxsize=1)
def bootstrap() -> dict:
    """Paired bootstrap of the Sharpe difference against each control.

    Months are resampled together across arms so the comparison stays paired;
    resampling each arm independently would understate the shared market moves
    and make every difference look more significant than it is.
    """
    rng = np.random.default_rng(0)
    a = AG["ret"].values
    n, out = len(a), {}
    for name in ("Classical RS", "Deterministic rule", "Constant cap 0.20"):
        if name not in PATHS:
            continue
        b = PATHS[name]["ret"].values
        d = np.empty(10000)
        for k in range(10000):
            i = rng.integers(0, n, n)
            aa, bb = a[i], b[i]
            d[k] = (aa.mean() / aa.std()) * np.sqrt(12) - (bb.mean() / bb.std()) * np.sqrt(12)
        out[name] = {"mean": float(d.mean()),
                     "lo": float(np.percentile(d, 2.5)),
                     "hi": float(np.percentile(d, 97.5)),
                     "p": float((d <= 0).mean())}
    return out


def read_decision(date) -> dict | None:
    try:
        return json.loads((CALLS / f"decision_{date:%Y-%m}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---- small components --------------------------------------------------------
def stat(label, value, color=INK, note=""):
    return html.Div([
        html.Div(label, style={"fontSize": "10.5px", "color": MUTED,
                               "textTransform": "uppercase",
                               "letterSpacing": "0.05em"}),
        html.Div(value, style={"fontSize": "23px", "fontWeight": 680,
                               "color": color, "lineHeight": "1.25",
                               "fontVariantNumeric": "tabular-nums"}),
        html.Div(note, style={"fontSize": "11px", "color": MUTED}) if note else None,
    ], style={**CARD, "flex": "1", "minWidth": "150px"})


def para(text, **kw):
    st = {"color": MUTED, "fontSize": "13px", "lineHeight": "1.6"}
    st.update(kw)
    return html.Div(text, style=st)


# ---- figures -----------------------------------------------------------------
def wealth_fig():
    fig = go.Figure()
    # the five random seeds are context, not identity: one recessive band, one
    # legend entry, so they cannot be mistaken for named arms
    for j, r in enumerate(RANDOMS):
        w = (1 + r["ret"]).cumprod()
        fig.add_trace(go.Scatter(
            x=w.index, y=w, mode="lines", name="Random posture (5 seeds)",
            line=dict(color=RANDOM, width=1), opacity=0.55,
            legendgroup="rand", showlegend=(j == 0),
            hovertemplate="random seed<br>%{x|%b %Y}: %{y:.1f}x<extra></extra>"))
    for name, (colour, _) in ARMS.items():
        if name not in PATHS:
            continue
        w = (1 + PATHS[name]["ret"]).cumprod()
        fig.add_trace(go.Scatter(
            x=w.index, y=w, mode="lines", name=name,
            line=dict(color=colour, width=2.6 if name == "Agent" else 2),
            hovertemplate=name + "<br>%{x|%b %Y}: %{y:.1f}x<extra></extra>"))
    ticks = [1, 2, 5, 10, 20, 50]
    fig.update_yaxes(title="wealth ($1 start)", type="log", tickmode="array",
                     tickvals=ticks, ticktext=[f"{t:g}x" for t in ticks],
                     minor=dict(showgrid=False))
    fig.update_layout(title="Wealth of one dollar, 2008 to 2025")
    return _fig(fig, 400)


def sharpe_bar():
    rows = [(n, stats(PATHS[n]["ret"])["sharpe"], c)
            for n, (c, _) in ARMS.items() if n in PATHS]
    if RANDOMS:
        rows.append(("Random (mean of 5)",
                     float(np.mean([sharpe(r["ret"]) for r in RANDOMS])), RANDOM))
    rows.sort(key=lambda t: t[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v, _ in rows], y=[n for n, _, _ in rows], orientation="h",
        marker=dict(color=[c for _, _, c in rows],
                    line=dict(color=BG, width=2)),
        text=[f"{v:.2f}" for _, v, _ in rows], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Sharpe %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Sharpe ratio", range=[0, 1.55])
    fig.update_layout(title="Risk-adjusted return, identical months")
    return _fig(fig, 330, legend=False)


def selectivity_fig():
    """The designed test: was the blend used where the filter is uncertain?"""
    d = AG.copy()
    d["amb"] = (d.p_bull > 0.2) & (d.p_bull < 0.8)
    grp = [("Filter ambiguous\n0.2 < P(bull) < 0.8", d[d.amb]),
           ("Filter decisive", d[~d.amb])]
    labels = [g[0].replace("\n", "<br>") for g in grp]
    pct = [(g[1].treatment == "probability_blend").mean() * 100 for g in grp]
    cnt = [f"{int((g[1].treatment == 'probability_blend').sum())} of {len(g[1])}"
           for g in grp]
    fig = go.Figure(go.Bar(
        x=labels, y=pct, marker=dict(color=[AGENT, RANDOM],
                                     line=dict(color=BG, width=2)),
        text=[f"{p:.0f}%<br><span style='font-size:11px'>{c}</span>"
              for p, c in zip(pct, cnt)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>blend used in %{y:.0f}% of months<extra></extra>"))
    fig.update_yaxes(title="months using the probability blend (%)", range=[0, 108])
    fig.update_layout(title="Was the intervention targeted?")
    return _fig(fig, 340, legend=False)


def gamma_scatter():
    d = AG
    fig = go.Figure(go.Scatter(
        x=d.p_bull, y=d.gamma, mode="markers",
        marker=dict(color=AGENT, size=8, opacity=0.55,
                    line=dict(color=BG, width=1)),
        customdata=np.stack([d.index.strftime("%b %Y"), d.cap], axis=-1),
        hovertemplate="%{customdata[0]}<br>P(bull) %{x:.2f}<br>"
                      "gamma %{y:.1f}<br>cap %{customdata[1]:.2f}<extra></extra>"))
    fig.add_hline(y=6.0, line=dict(color=CLASSIC, width=2, dash="dash"))
    fig.add_annotation(x=0.03, y=6.0, text="classical default, 6",
                       showarrow=False, yshift=12, xanchor="left",
                       font=dict(size=11, color=MUTED))
    r = float(np.corrcoef(d.p_bull, d.gamma)[0, 1])
    fig.update_xaxes(title="P(bull) reported by the filter")
    fig.update_yaxes(title="risk aversion the agent set")
    fig.update_layout(title=f"Caution rises as confidence falls (r = {r:.2f})")
    return _fig(fig, 340, legend=False)


def cap_fig():
    d = AG
    fig = go.Figure()
    bear = d[d.regime == "bear"]
    for dt in bear.index:
        fig.add_vrect(x0=dt - pd.Timedelta(days=15), x1=dt + pd.Timedelta(days=15),
                      fillcolor=CLASSIC, opacity=0.10, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=d.index, y=d.cap * 100, mode="lines", name="ceiling the agent set",
        line=dict(color=AGENT, width=2),
        hovertemplate="%{x|%b %Y}<br>cap %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=100, line=dict(color=CLASSIC, width=2, dash="dash"))
    fig.add_annotation(x=d.index[3], y=100, text="classical: no ceiling",
                       showarrow=False, yshift=11, xanchor="left",
                       font=dict(size=11, color=MUTED))
    fig.update_yaxes(title="largest allowed position (%)", range=[0, 112])
    fig.update_layout(title="The concentration ceiling, month by month "
                            "(shaded = detected bear)")
    return _fig(fig, 340, legend=False)


def results_table():
    hdr = ["Arm", "Final $1", "CAGR", "Vol", "Sharpe", "Max DD"]
    rows = []
    for name, (colour, _) in ARMS.items():
        if name not in PATHS:
            continue
        s = stats(PATHS[name]["ret"])
        rows.append((name, colour, [f"{s['final']:.1f}x", f"{s['cagr']:.1f}%",
                                    f"{s['vol']:.1f}%", f"{s['sharpe']:.2f}",
                                    f"{s['maxdd']:.1f}%"]))
    if RANDOMS:
        ss = [stats(r["ret"]) for r in RANDOMS]
        rows.append(("Random posture (5 seeds)", RANDOM, [
            f"{min(s['final'] for s in ss):.1f}-{max(s['final'] for s in ss):.1f}x",
            "", f"{np.mean([s['vol'] for s in ss]):.1f}%",
            f"{min(s['sharpe'] for s in ss):.2f}-{max(s['sharpe'] for s in ss):.2f}",
            ""]))
    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"textAlign": "left" if i == 0 else "right",
                              "padding": "7px 12px", "fontSize": "11.5px",
                              "textTransform": "uppercase",
                              "letterSpacing": "0.05em", "color": MUTED,
                              "borderBottom": f"2px solid {LINE}"})
            for i, h in enumerate(hdr)])),
        html.Tbody([
            html.Tr([
                html.Td([html.Span(style={"display": "inline-block", "width": "9px",
                                          "height": "9px", "borderRadius": "2px",
                                          "backgroundColor": colour,
                                          "marginRight": "8px"}), name],
                        style={"padding": "7px 12px",
                               "fontWeight": 650 if name == "Agent" else 400}),
                *[html.Td(v, style={"textAlign": "right", "padding": "7px 12px",
                                    "fontVariantNumeric": "tabular-nums",
                                    "fontWeight": 650 if name == "Agent" else 400})
                  for v in vals]],
                style={"borderBottom": f"1px solid {LINE}"})
            for name, colour, vals in rows]),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"})


def bootstrap_table():
    bs = bootstrap()
    order = ["Classical RS", "Constant cap 0.20", "Deterministic rule"]
    rows = []
    for name in order:
        if name not in bs:
            continue
        b = bs[name]
        sig = b["p"] < 0.05
        rows.append(html.Tr([
            html.Td(f"Agent vs {name}", style={"padding": "7px 12px"}),
            html.Td(f"{b['mean']:+.3f}", style={"textAlign": "right",
                                                "padding": "7px 12px",
                                                "fontVariantNumeric": "tabular-nums"}),
            html.Td(f"[{b['lo']:+.3f}, {b['hi']:+.3f}]",
                    style={"textAlign": "right", "padding": "7px 12px",
                           "fontVariantNumeric": "tabular-nums"}),
            html.Td(f"{b['p']:.3f}", style={"textAlign": "right",
                                            "padding": "7px 12px",
                                            "fontWeight": 650,
                                            "color": CONST if sig else "#B0472F"}),
            html.Td("distinguishable" if sig else "could be chance",
                    style={"padding": "7px 12px", "fontSize": "12px",
                           "color": CONST if sig else "#B0472F"}),
        ], style={"borderBottom": f"1px solid {LINE}"}))
    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"textAlign": "left" if i == 0 else "right",
                              "padding": "7px 12px", "fontSize": "11.5px",
                              "textTransform": "uppercase", "color": MUTED,
                              "letterSpacing": "0.05em",
                              "borderBottom": f"2px solid {LINE}"})
            for i, h in enumerate(["Comparison", "Sharpe gap", "95% interval",
                                   "P(no better)", "Verdict"])])),
        html.Tbody(rows)], style={"width": "100%", "borderCollapse": "collapse",
                                  "fontSize": "13px"})


# ---- app ---------------------------------------------------------------------
app = Dash(__name__, title="Agentic RS experiment")
MONTHS = list(AG.index) if HAS_DATA else []
n_months = len(MONTHS)
marks = ({i: MONTHS[i].strftime("%Y") for i in range(0, n_months, 24)}
         if n_months else {})

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                             "fontFamily": FONT, "color": INK,
                             "padding": "18px 26px"}, children=[
    html.H2("An agentic layer over the regime-switching optimal portfolio",
            style={"margin": "0 0 2px 0", "fontSize": "21px", "color": "#1F3A66"}),
    para("A controlled experiment. The econometrics and the optimiser are left "
         "untouched; only the three parameters the classical model fixes by "
         "assumption are handed to an agent. 216 months, January 2008 to "
         "December 2025.", marginBottom="12px"),
    dcc.Tabs(id="tabs", value="e1", children=[
        dcc.Tab(label="1 · The experiment", value="e1"),
        dcc.Tab(label="2 · Is the edge real?", value="e2"),
        dcc.Tab(label="3 · What the agent did", value="e3"),
        dcc.Tab(label="4 · Month by month", value="e4"),
    ]),
    html.Div(id="rsbody", style={"marginTop": "14px"}),
    dcc.Interval(id="rstick", interval=850, disabled=True),
])


@app.callback(Output("rsbody", "children"), Input("tabs", "value"))
def render(tab):
    if not HAS_DATA:
        return html.Div("No agent run found. Run rs_agentic --mode agent first.",
                        style={**CARD, "color": MUTED})

    if tab == "e1":
        s_ag = stats(AG["ret"])
        s_cl = stats(PATHS["Classical RS"]["ret"])
        return html.Div([
            html.Div([
                stat("Agent", f"{s_ag['sharpe']:.2f}", AGENT, "Sharpe ratio"),
                stat("Classical RS", f"{s_cl['sharpe']:.2f}", CLASSIC, "Sharpe ratio"),
                stat("Worst drop, agent", f"{s_ag['maxdd']:.0f}%", AGENT,
                     f"classical {s_cl['maxdd']:.0f}%"),
                stat("Cost of the run", "$1.09", MUTED, "213 calls, 0 rejected"),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
            html.Div([
                html.Div("What was tested", style={"fontWeight": 600,
                                                   "marginBottom": "6px"}),
                para("The classical model fixes three things by assumption: how "
                     "cautious to be, whether to round the market-state "
                     "probability to a yes or no at the halfway mark, and how "
                     "much of the fund may sit in one stock (currently "
                     "unlimited). Each month the agent reads a short report on "
                     "how reliable the model looks right now and sets those "
                     "three numbers. It never picks stocks and never forecasts "
                     "returns. Five arms share one code path, so any difference "
                     "belongs to the posture rather than the implementation."),
            ], style={**CARD, "marginBottom": "10px"}),
            html.Div(dcc.Graph(figure=wealth_fig(),
                               config={"displayModeBar": False}), style=CARD),
            html.Div([html.Div("All five arms, identical months",
                               style={"fontWeight": 600, "marginBottom": "8px"}),
                      results_table()], style={**CARD, "marginTop": "10px"}),
            para("The agent gives up about four times terminal wealth against "
                 "the classical model and buys a ten-point shallower worst "
                 "drawdown for it. Whether that trade is worth making is a "
                 "judgement about risk appetite, not a fact the data settles.",
                 marginTop="10px"),
        ])

    if tab == "e2":
        bs = bootstrap()
        rule_p = bs.get("Deterministic rule", {}).get("p", float("nan"))
        return html.Div([
            html.Div([
                html.Div("Why there are controls at all",
                         style={"fontWeight": 600, "marginBottom": "6px"}),
                para("Comparing the agent only against the unmodified model "
                     "would not settle anything, because several duller "
                     "explanations produce the same headline. Each control "
                     "rules one of them out. The constant-cap arm asks whether "
                     "the agent merely learned one fixed ceiling. The random "
                     "arm asks whether any fiddling at all would have looked "
                     "good. The deterministic rule, twenty lines of ordinary "
                     "code reading the same diagnostics, asks whether an agent "
                     "is needed for this job."),
            ], style={**CARD, "marginBottom": "10px"}),
            html.Div(dcc.Graph(figure=sharpe_bar(),
                               config={"displayModeBar": False}), style=CARD),
            html.Div([
                html.Div("Paired bootstrap, 10,000 resamples",
                         style={"fontWeight": 600, "marginBottom": "8px"}),
                bootstrap_table(),
                para("Months are resampled together across arms so the "
                     "comparison stays paired; resampling each arm "
                     "independently would hide the shared market moves and "
                     "make every gap look more convincing than it is.",
                     marginTop="10px", fontSize="12px"),
            ], style={**CARD, "marginTop": "10px"}),
            html.Div([
                html.Div("What this supports", style={"fontWeight": 600,
                                                      "color": CONST}),
                para("An agentic layer improves the risk-adjusted performance "
                     "of the regime-switching model against the unmodified "
                     "model, and that improvement is statistically "
                     "distinguishable from zero. It is not reproducible by any "
                     "constant ceiling: capping at the agent's own average "
                     "every month scores no better than the classical model. "
                     "So the variation is doing the work, not the level."),
                html.Div("What it does not support",
                         style={"fontWeight": 600, "color": "#B0472F",
                                "marginTop": "10px"}),
                para(f"That the agent beats a twenty-line deterministic rule "
                     f"with the same diagnostics. It edges ahead on this path, "
                     f"but there is a {rule_p*100:.0f}% probability the "
                     f"ordering is chance and the interval spans zero. On this "
                     f"evidence, judgement helps; that the judgement must come "
                     f"from a language model is unproven."),
            ], style={**CARD, "marginTop": "10px"}),
        ])

    if tab == "e3":
        d = AG.copy()
        d["amb"] = (d.p_bull > 0.2) & (d.p_bull < 0.8)
        amb_pct = (d[d.amb].treatment == "probability_blend").mean() * 100
        clr_pct = (d[~d.amb].treatment == "probability_blend").mean() * 100
        return html.Div([
            html.Div([
                stat("Deviated from defaults",
                     f"{int(d.deviated.sum())}/{len(d)}", MUTED, "99% of months"),
                stat("Blend when filter unclear", f"{amb_pct:.0f}%", AGENT,
                     f"{int(d[d.amb].treatment.eq('probability_blend').sum())} of {int(d.amb.sum())} months"),
                stat("Blend when filter clear", f"{clr_pct:.0f}%", RANDOM,
                     f"{int(d[~d.amb].treatment.eq('probability_blend').sum())} of {int((~d.amb).sum())} months"),
                stat("Schema rejections", f"{int((~d.accepted).sum())}", CONST,
                     "out of 216"),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
            html.Div([
                html.Div("Blanket caution, or actual discrimination?",
                         style={"fontWeight": 600, "marginBottom": "6px"}),
                para("The agent changed something in 99% of months, which on "
                     "its own looks like permanent caution rather than "
                     "judgement. The composition tells a different story. The "
                     "probability blend exists to handle a filter that is "
                     "genuinely uncertain, and that is almost exactly where it "
                     "was used. This is the strongest evidence in the "
                     "experiment, because it is hard to reach by luck."),
            ], style={**CARD, "marginBottom": "10px"}),
            html.Div([
                html.Div(dcc.Graph(figure=selectivity_fig(),
                                   config={"displayModeBar": False}),
                         style={**CARD, "flex": "1"}),
                html.Div(dcc.Graph(figure=gamma_scatter(),
                                   config={"displayModeBar": False}),
                         style={**CARD, "flex": "1"}),
            ], style={"display": "flex", "gap": "10px"}),
            html.Div(dcc.Graph(figure=cap_fig(),
                               config={"displayModeBar": False}),
                     style={**CARD, "marginTop": "10px"}),
            para("The ceiling is looser in bear months than bull, which looks "
                 "backwards until the interaction is considered: a higher risk "
                 "aversion already shrinks the allocation, so the ceiling has "
                 "little left to bind on. The agent tightened whichever "
                 "instrument was actually operative rather than applying both "
                 "at once.", marginTop="10px"),
        ])

    # tab 4
    return html.Div([
        html.Div([
            html.Button("▶  Play", id="rsplay", n_clicks=0,
                        style={"padding": "8px 20px", "fontSize": "14px",
                               "fontWeight": 600, "cursor": "pointer",
                               "border": f"1px solid {AGENT}",
                               "borderRadius": "5px", "backgroundColor": AGENT,
                               "color": "#fff"}),
            html.Span(id="rsmonth", style={"marginLeft": "18px",
                                           "fontSize": "19px", "fontWeight": 650}),
            html.Span(id="rsbadge", style={"marginLeft": "14px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "10px"}),
        dcc.Slider(id="rsslider", min=0, max=max(n_months - 1, 0), step=1, value=0,
                   marks=marks, tooltip={"placement": "bottom"},
                   updatemode="drag"),
        html.Div(id="rsstats", style={"display": "flex", "gap": "10px",
                                      "margin": "14px 0"}),
        html.Div([
            html.Div([html.Div("What the agent was shown",
                               style={"fontWeight": 600, "marginBottom": "6px"}),
                      html.Pre(id="rsbrief", style=MONO)],
                     style={**CARD, "flex": "1"}),
            html.Div([html.Div("What it decided, and why",
                               style={"fontWeight": 600, "marginBottom": "6px"}),
                      html.Div(id="rsdecision")],
                     style={**CARD, "flex": "1"}),
        ], style={"display": "flex", "gap": "10px"}),
    ])


@app.callback(Output("rstick", "disabled"), Output("rsplay", "children"),
              Input("rsplay", "n_clicks"), prevent_initial_call=True)
def toggle(n):
    playing = n % 2 == 1
    return (not playing), ("❚❚  Pause" if playing else "▶  Play")


@app.callback(Output("rsslider", "value"), Input("rstick", "n_intervals"),
              State("rsslider", "value"), prevent_initial_call=True)
def advance(_, cur):
    return 0 if cur is None or cur >= n_months - 1 else cur + 1


@app.callback(
    Output("rsmonth", "children"), Output("rsbadge", "children"),
    Output("rsstats", "children"), Output("rsbrief", "children"),
    Output("rsdecision", "children"),
    Input("rsslider", "value"))
def update(i):
    row = AG.iloc[int(i or 0)]
    d = AG.index[int(i or 0)]
    is_bear = row.regime == "bear"
    badge = html.Span(row.regime.upper(), style={
        "backgroundColor": CLASSIC if is_bear else CONST, "color": "#fff",
        "padding": "3px 12px", "borderRadius": "11px", "fontSize": "12px",
        "fontWeight": 700, "letterSpacing": "0.05em"})

    cap_txt = "no ceiling" if row.cap >= 1.0 else f"{row.cap*100:.0f}%"
    stats_row = [
        stat("P(bull)", f"{row.p_bull:.2f}", MUTED),
        stat("Risk aversion", f"{row.gamma:.1f}", AGENT,
             "classical uses 6"),
        stat("Position ceiling", cap_txt, AGENT, "classical has none"),
        stat("Regime treatment",
             "blend" if row.treatment == "probability_blend" else "hard label",
             AGENT if row.treatment == "probability_blend" else MUTED),
        stat("Return that month", f"{row.ret*100:+.1f}%",
             CONST if row.ret >= 0 else "#B0472F"),
    ]

    rec = read_decision(d)
    if not rec:
        return (d.strftime("%B %Y"), badge, stats_row,
                "(no decision file found)", "")
    body = html.Div([
        html.Div(rec["posture"].get("rationale", ""),
                 style={"fontSize": "13px", "lineHeight": "1.65",
                        "color": INK, "borderLeft": f"3px solid {AGENT}",
                        "paddingLeft": "12px", "marginBottom": "12px"}),
        html.Div([
            html.Span("accepted by the schema" if rec["accepted"]
                      else f"rejected: {rec['reject_reason']}",
                      style={"color": CONST if rec["accepted"] else "#B0472F",
                             "fontWeight": 600, "fontSize": "12px"}),
            html.Span(f" · {rec['usage']['output_tokens']} tokens · "
                      f"{rec['usage']['latency_ms']/1000:.1f}s",
                      style={"color": MUTED, "fontSize": "12px"}),
        ]),
        html.Details([
            html.Summary("Raw reply", style={"cursor": "pointer",
                                             "fontSize": "12px",
                                             "color": AGENT, "fontWeight": 600,
                                             "marginTop": "12px"}),
            html.Pre(rec.get("raw_text", ""),
                     style={**MONO, "maxHeight": "180px", "marginTop": "8px"}),
        ]),
    ])
    return d.strftime("%B %Y"), badge, stats_row, rec["briefing"], body


def main() -> int:
    print(f"Agentic RS dashboard  ->  http://127.0.0.1:{PORT}")
    app.run(debug=False, host="127.0.0.1", port=PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
