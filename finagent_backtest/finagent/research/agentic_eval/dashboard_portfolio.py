"""FinAgent portfolio dashboard: the agent in action, month by month (localhost).

A replay of the archived monthly backtest, held to the 225 months both regime
signals cover (2005-05 to 2024-01). Steps through those 225 autonomous
decisions, showing for each month the briefing the model actually
read (regenerated deterministically by ``regen_sessions``) alongside the
portfolio it produced.

  Tab 1  In action    - month scrubber + Play; briefing -> plan the model returned
  Tab 2  Autonomy     - does the plan change with the market state?
  Tab 3  Trust        - reliability, turnover and cost across every cycle
  Tab 4  Performance  - wealth, drawdown and relative strength vs equal-weight
  Tab 5  AI vs trad.  - agentic vs static M-V vs regime-switching optimal
  Tab 6  Planner      - what the LLM actually varies across the captured plans
  Tab 7  Regime swap  - Hamilton vs the frozen agentic detector (negative result)

Prerequisites:
    python -m finagent.research.agentic_eval.regen_sessions
    python -m finagent.research.agentic_eval.capture_plans   # optional; enables
                                                             # the plan pane + tab 6

Run:
    python -m finagent.research.agentic_eval.dashboard_portfolio
    # then open http://127.0.0.1:8060
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from finagent.research.agentic_eval import dashboard_regime_swap as SWAP

REPO_ROOT = Path(__file__).resolve().parents[3]
SESSIONS = REPO_ROOT / "data" / "agentic_eval" / "sessions"
PLANS = REPO_ROOT / "data" / "agentic_eval" / "plans"
BACKTEST = REPO_ROOT / "data" / "agentic_eval" / "backtest" / "path0.parquet"

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]

# Every tab is held to the window both regime signals cover. The frozen agentic
# detector ends 2023-12-29, so the swap comparison in tab 7 cannot run past a
# 2024-01 realised return; the archived run itself does continue to 2025-12.
# Truncating here rather than only in tab 7 is what keeps the dashboard
# self-consistent: a wealth multiple quoted from tab 4 now means the same thing
# as the one quoted from tab 7, instead of silently carrying two extra years of
# compounding that the other arm never had the chance to earn.
WINDOW_END = pd.Timestamp("2024-01-31")

# ---- white theme -------------------------------------------------------------
BG, PANEL, LINE = "#FFFFFF", "#F7F8FA", "#E3E6EC"
INK, MUTED = "#1A1A2E", "#5B6472"
NAVY, BULL, BEAR, AMBER, TEAL = "#1F3A66", "#2E7D4F", "#C23A2B", "#C0851C", "#2C7A7B"
PORT = 8060

CARD = {"backgroundColor": PANEL, "border": f"1px solid {LINE}", "borderRadius": "6px",
        "padding": "14px 16px"}
FONT = "Segoe UI, Inter, system-ui, sans-serif"
MONO = {"backgroundColor": "#FCFCFD", "border": f"1px solid {LINE}",
        "borderRadius": "5px", "padding": "12px", "fontSize": "11.5px",
        "lineHeight": "1.45", "maxHeight": "340px", "overflowY": "auto",
        "fontFamily": "JetBrains Mono, Consolas, monospace",
        "whiteSpace": "pre-wrap", "margin": 0}


def _log_wealth_axis(fig, ticks):
    """Label only the given multiples on a log wealth axis.

    Plotly labels minor gridlines on a log scale by default, so a range of
    0.4 to 60 comes out as 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4 ..., unreadable.
    """
    fig.update_yaxes(title="wealth ($1 start)", type="log",
                     tickmode="array", tickvals=ticks,
                     ticktext=[(f"{t:g}×") for t in ticks],
                     showgrid=True, minor=dict(showgrid=False))
    return fig


def _fig(fig, height=300, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=56, r=24, t=44, b=40),
        showlegend=legend,
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=LINE, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# ---- data --------------------------------------------------------------------
def load():
    idx = pd.read_parquet(SESSIONS / "index.parquet")
    idx["decision_date"] = pd.to_datetime(idx["decision_date"])
    idx["realised_date"] = pd.to_datetime(idx["realised_date"])
    idx = idx.sort_values("decision_date").reset_index(drop=True)
    # cut on the realised month, not the decision month: a row is stamped with
    # the month its return landed, which is the quantity the wealth curves and
    # the tab-7 comparison are both built from.
    idx = idx[idx["realised_date"] <= WINDOW_END].reset_index(drop=True)

    bt = pd.read_parquet(BACKTEST)
    bt.index = pd.to_datetime(bt.index)
    bt = bt[bt.index <= WINDOW_END]

    # equal-weight benchmark on the identical realised returns
    ew_ret = bt[[f"r_{t}" for t in TICKERS]].mean(axis=1)
    perf = pd.DataFrame({
        "agent": bt["nav"] / bt["nav"].iloc[0] * (1 + bt["net_ret"].iloc[0]),
        "equal_weight": (1 + ew_ret).cumprod(),
    }, index=bt.index)
    perf["agent"] = (1 + bt["net_ret"]).cumprod()
    return idx, bt, perf


def load_plans():
    """Planner capture, if ``capture_plans`` has been run. Optional by design:
    the dashboard degrades to its pre-capture behaviour when absent."""
    f = PLANS / "index.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f)
    p["decision_date"] = pd.to_datetime(p["decision_date"])
    p = p.sort_values("decision_date").reset_index(drop=True)
    # plans are stamped with the decision month, one month ahead of the realised
    # return, so the last plan in window is the one that produced WINDOW_END
    return p[p["decision_date"] < WINDOW_END].reset_index(drop=True)


def read_plan(date) -> dict | None:
    """Lazily read one month's captured plan (~10 KB each)."""
    try:
        return json.loads(
            (PLANS / f"plan_{date.strftime('%Y-%m')}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


IDX, BT, PERF = load()
PLAN_IDX = load_plans()
HAS_PLANS = PLAN_IDX is not None
BEARS = IDX[IDX.regime == "bear"]["decision_date"]

# ---- regime-switching optimal benchmark (Markose & Yang, EC907 Note 4) -------
RS_FILE = REPO_ROOT / "data" / "agentic_eval" / "rs_optimal.parquet"
RS = None
if RS_FILE.exists():
    RS = pd.read_parquet(RS_FILE)
    RS.index = pd.to_datetime(RS.index)
    # align all four strategies on the RS window (RS needs 36m of CAPM history)
    common = RS.index.intersection(BT.index)
    RS = RS.loc[common]
    CMP = pd.DataFrame({
        "FinAgent (agentic AI)": (1 + BT.loc[common, "net_ret"]).cumprod(),
        "RS dynamic optimal": (1 + RS["ret_rs"]).cumprod(),
        "Traditional static M-V": (1 + RS["ret_static"]).cumprod(),
        "Equal-weight 8": (1 + RS["ret_equal"]).cumprod(),
    })
    CMP_RET = pd.DataFrame({
        "FinAgent (agentic AI)": BT.loc[common, "net_ret"],
        "RS dynamic optimal": RS["ret_rs"],
        "Traditional static M-V": RS["ret_static"],
        "Equal-weight 8": RS["ret_equal"],
    })
    AGENT_GROSS = (1 - BT.loc[common, "w_cash"])

CMP_COLS = {"FinAgent (agentic AI)": NAVY, "RS dynamic optimal": AMBER,
            "Traditional static M-V": "#8A6BAF", "Equal-weight 8": TEAL}


def bear_bands(fig):
    """Shade contiguous detected-bear stretches."""
    d = IDX.set_index("decision_date")["regime"].eq("bear")
    grp = (d != d.shift()).cumsum()
    for _, seg in d[d].groupby(grp[d]):
        fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor=BEAR,
                      opacity=0.09, line_width=0, layer="below")
    return fig


# ---- tab 1: in action --------------------------------------------------------
def weights_fig(row):
    w = [row[f"w_{t}"] * 100 for t in TICKERS]
    colors = [MUTED if v <= 1e-6 else NAVY for v in w]
    fig = go.Figure(go.Bar(x=TICKERS, y=w, marker_color=colors,
                           text=[f"{v:.1f}%" if v > 1e-6 else "dropped" for v in w],
                           textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=["CASH"], y=[row["w_cash"] * 100], marker_color=AMBER,
                         text=[f"{row['w_cash']*100:.1f}%"], textposition="outside",
                         cliponaxis=False))
    fig.update_yaxes(title="weight (%)", range=[0, max(28, max(w) * 1.35)])
    return _fig(fig, 330, legend=False)


def gauge_fig(p_bull):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_bull * 100,
        number={"suffix": "%", "font": {"size": 26, "color": INK}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED},
               "bar": {"color": BULL if p_bull > 0.5 else BEAR},
               "bgcolor": BG, "borderwidth": 1, "bordercolor": LINE,
               "steps": [{"range": [0, 50], "color": "#FBEAE7"},
                         {"range": [50, 100], "color": "#E8F3EC"}]}))
    fig.update_layout(height=190, paper_bgcolor=BG,
                      font=dict(family=FONT, color=INK),
                      margin=dict(l=24, r=24, t=10, b=8))
    return fig


def stat(label, value, color=INK):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": MUTED,
                               "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        html.Div(value, style={"fontSize": "23px", "fontWeight": 650, "color": color}),
    ], style={**CARD, "flex": "1", "minWidth": "128px"})


# ---- tab 2/3/4 static figures ------------------------------------------------
def exposure_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=IDX.decision_date, y=IDX.gross_exposure * 100,
                             mode="lines", line=dict(color=NAVY, width=1.6),
                             name="gross exposure", fill="tozeroy",
                             fillcolor="rgba(31,58,102,0.10)"))
    bear_bands(fig)
    fig.update_yaxes(title="invested (%)", range=[0, 105])
    fig.update_layout(title="Gross exposure over time (shaded = detected bear)")
    return _fig(fig, 360)


def exposure_box():
    fig = go.Figure()
    for lab, col in [("bull", BULL), ("bear", BEAR)]:
        s = IDX[IDX.regime == lab]["gross_exposure"] * 100
        fig.add_trace(go.Box(y=s, name=f"{lab} (n={len(s)})", marker_color=col,
                             boxmean=True))
    fig.update_yaxes(title="invested (%)")
    fig.update_layout(title="Exposure by detected regime")
    return _fig(fig, 360, legend=False)


def turnover_fig():
    fig = go.Figure(go.Bar(x=IDX.decision_date, y=IDX.turnover,
                           marker_color=NAVY, name="turnover"))
    bear_bands(fig)
    fig.update_yaxes(title="turnover (sum |Δw|)")
    fig.update_layout(title="Monthly turnover: how much the book was rewritten")
    return _fig(fig, 320, legend=False)


def reliability_fig():
    ok = int((~IDX.used_fallback).sum())
    fig = go.Figure(go.Bar(
        x=["LLM plan accepted", "Safe-default fallback"],
        y=[ok, int(IDX.used_fallback.sum())],
        marker_color=[BULL, BEAR],
        text=[f"{ok}", f"{int(IDX.used_fallback.sum())}"], textposition="outside",
        cliponaxis=False))
    fig.update_yaxes(title="cycles", range=[0, len(IDX) * 1.18])
    fig.update_layout(title=f"Planner reliability across {len(IDX)} decisions")
    return _fig(fig, 320, legend=False)


def wealth_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=PERF.index, y=PERF.agent, mode="lines", name="FinAgent",
                             line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(x=PERF.index, y=PERF.equal_weight, mode="lines",
                             name="Equal-weight 8", line=dict(color=TEAL, width=1.5)))
    bear_bands(fig)
    _log_wealth_axis(fig, [1, 2, 5, 10, 20, 50, 100])
    fig.update_layout(
        title=f"Wealth, {PERF.index[0]:%Y}–{PERF.index[-1]:%Y} (log scale)")
    return _fig(fig, 340)


def drawdown_fig():
    fig = go.Figure()
    for name, col, c in [("FinAgent", "agent", NAVY), ("Equal-weight 8", "equal_weight", TEAL)]:
        dd = (PERF[col] / PERF[col].cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=PERF.index, y=dd, mode="lines", name=name,
                                 line=dict(color=c, width=1.3), fill="tozeroy",
                                 fillcolor=("rgba(31,58,102,0.16)" if c == NAVY
                                            else "rgba(44,122,123,0.13)")))
    fig.update_yaxes(title="drawdown (%)")
    fig.update_layout(title="Drawdown: the agent's edge is a shallower hole")
    return _fig(fig, 320)


def cmp_wealth_fig():
    fig = go.Figure()
    for name, col in CMP_COLS.items():
        fig.add_trace(go.Scatter(x=CMP.index, y=CMP[name], mode="lines", name=name,
                                 line=dict(color=col,
                                           width=2.2 if "FinAgent" in name else 1.5)))
    bear_bands(fig)
    # Explicit decade-ish ticks: plotly's default log axis labels the minor
    # gridlines too (0.6, 0.7, 3, 4, 6 ...), which reads as noise.
    _log_wealth_axis(fig, [0.5, 1, 2, 5, 10, 20, 50])
    fig.update_layout(title="Agentic AI vs regime-switching optimal vs traditional")
    return _fig(fig, 380)


def cmp_exposure_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=AGENT_GROSS.index, y=AGENT_GROSS * 100, mode="lines",
                             name="FinAgent", line=dict(color=NAVY, width=1.8)))
    fig.add_trace(go.Scatter(x=RS.index, y=RS["gross_rs"] * 100, mode="lines",
                             name="RS dynamic optimal", line=dict(color=AMBER, width=1.5)))
    fig.add_trace(go.Scatter(x=RS.index, y=RS["gross_static"] * 100, mode="lines",
                             name="Traditional static M-V",
                             line=dict(color="#8A6BAF", width=1.2, dash="dot")))
    bear_bands(fig)
    fig.update_yaxes(title="invested (%)", range=[0, 105])
    fig.update_layout(title="How each method sizes risk (shaded = detected bear)")
    return _fig(fig, 340)


def cmp_table():
    rows = []
    for name in CMP_COLS:
        r = CMP_RET[name]
        w = CMP[name]
        n = len(r)
        cagr = (w.iloc[-1] ** (12 / n) - 1) * 100
        vol = r.std() * np.sqrt(12) * 100
        shp = (r.mean() / r.std()) * np.sqrt(12)
        dd = (w / w.cummax() - 1).min() * 100
        rows.append((name, f"{w.iloc[-1]:.1f}×", f"{cagr:.1f}%", f"{vol:.1f}%",
                     f"{shp:.2f}", f"{dd:.1f}%"))
    hdr = ["Strategy", "Final $1", "CAGR", "Vol", "Sharpe", "Max DD"]
    return html.Table([
        html.Thead(html.Tr([html.Th(h, style={
            "textAlign": "left" if i == 0 else "right", "padding": "7px 12px",
            "borderBottom": f"2px solid {NAVY}", "fontSize": "12px",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "color": NAVY}) for i, h in enumerate(hdr)])),
        html.Tbody([html.Tr([html.Td(c, style={
            "textAlign": "left" if i == 0 else "right", "padding": "7px 12px",
            "borderBottom": f"1px solid {LINE}",
            "fontWeight": 650 if "FinAgent" in r[0] else 400,
            "color": CMP_COLS[r[0]] if i == 0 else INK}) for i, c in enumerate(r)])
            for r in rows]),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"})


def style_cards():
    """What each horse in the race actually is.

    Written from the implementations (``rs_optimal.py`` / ``backtest.py``) rather
    than from the textbook definitions, so the reader can tie a line in the chart
    to the code that produced it.
    """
    styles = [
        ("FinAgent (agentic AI)",
         "The system under test. Each month a briefing (detected regime, the eight "
         "Alpha158 scores, and the agent's own realised P&L history) is compressed into "
         "SESSION.md and sent to Claude, which returns an execution graph whose nodes "
         "carry a risk posture: a per-position cap, a sizing rule and any names to "
         "exclude. A deterministic mapping turns that posture into long-only weights; "
         "whatever is not invested sits in cash. No optimiser and no forecast of "
         "returns: the judgement is the language model's, the arithmetic is fixed."),
        ("RS dynamic optimal",
         "(Markose & Yang, EC907 Note 4) method. A Hamilton two-state "
         "model is fitted to market excess returns in-sample to April 2005. Each month "
         "it takes the prevailing state and forms *forward-looking* market moments from "
         "the transition probabilities, not the current state's mean but the "
         "probability-weighted mean of where the market is heading next (Eqs. 15–16). "
         "Individual stocks enter only through their CAPM beta (Eq. 17), so a stock's "
         "expected return shifts with the regime purely via its market sensitivity. "
         "Those conditional moments feed a long-only mean-variance programme re-solved "
         "every month, with betas re-estimated point-in-time on an expanding window."),
        ("Traditional static M-V",
         "The control, and the textbook baseline. Identical optimiser, identical "
         "constraints and identical risk aversion to the RS method. The single "
         "difference is that its inputs are *unconditional*: the sample mean and "
         "covariance of the eight stocks over all history up to the decision month, with "
         "no notion of a market state. It is included to isolate what regime-conditioning "
         "adds, since any gap between this row and the RS row is attributable to that "
         "one change and nothing else."),
        ("Equal-weight 8",
         "One-eighth in each name, every month, always fully invested. No estimation, no "
         "optimisation, no parameters, and therefore nothing to overfit. It is here "
         "because naive diversification is a notoriously hard benchmark to beat once "
         "estimation error is accounted for: a strategy that cannot clear this bar has "
         "not demonstrated that its machinery earns its keep."),
    ]
    return html.Div([
        html.Div([
            html.Div(name, style={"fontWeight": 650, "fontSize": "13px",
                                  "color": CMP_COLS[name], "marginBottom": "5px",
                                  "borderLeft": f"3px solid {CMP_COLS[name]}",
                                  "paddingLeft": "8px"}),
            html.Div(body, style={"color": MUTED, "fontSize": "12.5px",
                                  "lineHeight": "1.55"}),
        ], style={"flex": "1 1 46%", "minWidth": "280px"})
        for name, body in styles
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px 24px"})


def rs_params() -> dict:
    """The Hamilton parameters the RS benchmark actually used.

    Read from the parquet's ``attrs`` rather than refitted here, so the panel can
    never drift from the moments that produced the weights. The nested ``e_m`` /
    ``S_m`` keys come back as strings after the parquet round-trip.
    """
    p = dict(RS.attrs["params"])
    p["e_m"] = {int(k): v for k, v in p["e_m"].items()}
    p["S_m"] = {int(k): v for k, v in p["S_m"].items()}
    return p


def rs_param_table():
    p = rs_params()
    P, Q = p["P"], p["Q"]
    rows = [
        ("Bull mean return, mu_1", f"{p['mu1']*100:+.3f} %/mo"),
        ("Bear mean return, mu_2", f"{p['mu2']*100:+.3f} %/mo"),
        ("Bull volatility, sigma_1",
         f"{p['s1']*100:.3f} %/mo  ({p['s1']*100*np.sqrt(12):.1f} %/yr, constrained)"),
        ("Bear volatility, sigma_2",
         f"{p['s2']*100:.3f} %/mo  ({p['s2']*100*np.sqrt(12):.1f} %/yr)"),
        ("P, probability bull persists", f"{P:.4f}"),
        ("Q, probability bear persists", f"{Q:.4f}"),
        ("Expected bull duration", f"{1/(1-P):.1f} months"),
        ("Expected bear duration", f"{1/(1-Q):.1f} months"),
        ("Ergodic P(bull)", f"{(1-Q)/(2-P-Q):.3f}"),
        ("Risk aversion, gamma", f"{p['gamma']:.1f}"),
    ]
    return html.Table([html.Tbody([
        html.Tr([
            html.Td(k, style={"padding": "4px 0", "color": MUTED, "fontSize": "12.5px"}),
            html.Td(v, style={"padding": "4px 0 4px 18px", "textAlign": "right",
                              "fontWeight": 600, "fontSize": "12.5px",
                              "fontVariantNumeric": "tabular-nums",
                              "whiteSpace": "nowrap"}),
        ]) for k, v in rows
    ])], style={"width": "100%", "borderCollapse": "collapse"})


def rs_moments_fig():
    """State parameters against the Eq. 15-16 moments actually optimised against.

    The point of the figure is the gap between the two: because both states are
    persistent, the expected *next-month* moment is a probability-weighted blend,
    so the means pull toward each other and the variances pick up an extra
    regime-uncertainty term. That blend is what makes the method forward-looking
    rather than merely regime-labelled.
    """
    p = rs_params()
    raw = [p["mu1"] * 100, p["mu2"] * 100, p["s1"] * 100, p["s2"] * 100]
    fwd = [p["e_m"][1] * 100, p["e_m"][2] * 100,
           np.sqrt(p["S_m"][1]) * 100, np.sqrt(p["S_m"][2]) * 100]
    x = ["Mean, bull", "Mean, bear", "SD, bull", "SD, bear"]
    fig = go.Figure()
    fig.add_bar(x=x, y=raw, name="State parameter", marker_color=MUTED,
                text=[f"{v:+.2f}" if i < 2 else f"{v:.2f}" for i, v in enumerate(raw)],
                textposition="outside", cliponaxis=False)
    fig.add_bar(x=x, y=fwd, name="Eq. 15-16, forward-looking", marker_color=AMBER,
                text=[f"{v:+.2f}" if i < 2 else f"{v:.2f}" for i, v in enumerate(fwd)],
                textposition="outside", cliponaxis=False)
    fig.add_hline(y=0, line=dict(color=LINE, width=1))
    fig.update_yaxes(title="market excess return (%/mo)")
    fig.update_layout(title="What the optimiser is actually given each month",
                      barmode="group",
                      legend=dict(orientation="h", y=1.04, x=0))
    return _fig(fig, 330)


_RS_EQUATIONS = r"""
**The model.** The market excess return $r_t$ is drawn from one of two states,
$s_t \in \{1=\text{bull},\, 2=\text{bear}\}$, each with its own mean and variance:

$$r_t \mid s_t = i \;\sim\; \mathcal{N}\!\left(\mu_i,\; \sigma_i^2\right)$$

This is what $\mu_1, \mu_2, \sigma_1, \sigma_2$ in the table are. The state itself is
never observed; it is inferred, and $\sigma_1$ is pinned at 14 percent a year so the
low-volatility state is always the one labelled bull.

**The state evolves as a Markov chain**, which is what $P$ and $Q$ measure:

$$\Pi =
\begin{pmatrix} P & 1-P \\ 1-Q & Q \end{pmatrix},
\qquad
P = \Pr(s_t = 1 \mid s_{t-1} = 1), \quad Q = \Pr(s_t = 2 \mid s_{t-1} = 2)$$

Both being close to one means states are persistent. The expected time spent in a
state, and the long-run share of months in the bull state, follow directly:

$$E[\text{duration}_1] = \frac{1}{1-P}, \qquad
E[\text{duration}_2] = \frac{1}{1-Q}, \qquad
\pi_1 = \frac{1-Q}{2-P-Q}$$

**Eqs. 15 and 16, the forward-looking moments.** Because the state persists only
probabilistically, next month's expected market moment is a blend of both states,
weighted by the chance of staying or switching. This is what the optimiser receives:

$$e^m(1) = P\mu_1 + (1-P)\mu_2, \qquad e^m(2) = (1-Q)\mu_1 + Q\mu_2$$

$$\Sigma^m(1) = P\sigma_1^2 + (1-P)\sigma_2^2 + P(1-P)(\mu_1-\mu_2)^2$$

$$\Sigma^m(2) = (1-Q)\sigma_1^2 + Q\sigma_2^2 + Q(1-Q)(\mu_1-\mu_2)^2$$

The final term in each variance is the regime-uncertainty premium: risk rises with
the chance of switching and with how far apart the two state means are.

**Eq. 17, from market to securities.** Individual stocks inherit the regime only
through their CAPM beta, with $\alpha_i$, $\beta_i$ and the idiosyncratic variances
$D$ re-estimated point-in-time each month:

$$\mu_i(s) = \alpha_i + \beta_i\, e^m(s), \qquad
\Sigma(s) = \beta\,\beta^{\top} \Sigma^m(s) + D$$

Here $\beta$ is the vector of the eight betas, so $\beta\beta^{\top}$ is the matrix of
their pairwise products, and $D$ is the diagonal matrix of idiosyncratic variances.

**The portfolio choice**, solved afresh every month, is where $\gamma$ enters:

$$\max_{w} \; w^{\top}\mu(s)
-\frac{\gamma}{2}\, w^{\top}\Sigma(s)\,w
\qquad \text{s.t.} \quad w \ge 0, \;\; \sum_i w_i \le 1$$

A higher $\gamma$ penalises variance more heavily. Any weight not allocated is held
in cash at the risk-free rate.
"""


def rs_equations():
    return dcc.Markdown(_RS_EQUATIONS, mathjax=True,
                        style={"color": MUTED, "fontSize": "12.5px",
                               "lineHeight": "1.6"})


def rs_transition_note():
    p = rs_params()
    P, Q = p["P"], p["Q"]
    return html.Div([
        html.Div("Transition matrix", style={"fontWeight": 600, "fontSize": "12.5px",
                                            "marginBottom": "6px"}),
        html.Table([html.Tbody([
            html.Tr([html.Td(""), html.Td("to bull", style={"padding": "2px 10px",
                                                            "color": BULL}),
                     html.Td("to bear", style={"padding": "2px 10px", "color": BEAR})]),
            html.Tr([html.Td("from bull", style={"color": MUTED, "paddingRight": "10px"}),
                     html.Td(f"{P:.4f}", style={"padding": "2px 10px",
                                                "fontWeight": 600}),
                     html.Td(f"{1-P:.4f}", style={"padding": "2px 10px"})]),
            html.Tr([html.Td("from bear", style={"color": MUTED, "paddingRight": "10px"}),
                     html.Td(f"{1-Q:.4f}", style={"padding": "2px 10px"}),
                     html.Td(f"{Q:.4f}", style={"padding": "2px 10px",
                                                "fontWeight": 600})]),
        ])], style={"fontSize": "12.5px", "fontVariantNumeric": "tabular-nums",
                    "borderCollapse": "collapse"}),
    ])


def cmp_regime_bar():
    ag_b = AGENT_GROSS[RS.regime == "bull"].mean() * 100
    ag_r = AGENT_GROSS[RS.regime == "bear"].mean() * 100
    rs_b = RS.loc[RS.regime == "bull", "gross_rs"].mean() * 100
    rs_r = RS.loc[RS.regime == "bear", "gross_rs"].mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Bull", "Bear"], y=[ag_b, ag_r], name="FinAgent",
                         marker_color=NAVY, text=[f"{ag_b:.0f}%", f"{ag_r:.0f}%"],
                         textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=["Bull", "Bear"], y=[rs_b, rs_r], name="RS optimal",
                         marker_color=AMBER, text=[f"{rs_b:.0f}%", f"{rs_r:.0f}%"],
                         textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="avg invested (%)", range=[0, 118])
    fig.update_layout(title="Risk-off behaviour by regime", barmode="group")
    return _fig(fig, 340)


# ---- the planned DAG ---------------------------------------------------------
# One colour per node type, so the invariant seven-stage chain is readable at a
# glance and the risk node, the only one whose params actually move, stands out.
NODE_COLOR = {
    "data_fetch": "#8A94A6", "feature": "#8A94A6", "signal": TEAL,
    "risk": AMBER, "portfolio": NAVY, "execution": "#8A94A6", "audit": "#8A94A6",
}
_KEY_PARAMS = ("max_position_pct", "max_position_size_pct", "sizing_mode",
               "reduce_only", "halt_new_longs", "flagged_symbols",
               "circuit_breaker", "slippage_guard_bps")


def dag_chain(plan: dict | None):
    """Render the returned DAG as a left-to-right chain of typed nodes."""
    if not plan:
        return html.Div("No captured plan for this month. Run capture_plans.",
                        style={"color": MUTED, "fontSize": "12px"})
    nodes = plan["dag"]["nodes"]
    chips = []
    for j, n in enumerate(nodes):
        if j:
            chips.append(html.Span("→", style={"color": MUTED, "margin": "0 6px",
                                               "fontSize": "15px"}))
        colour = NODE_COLOR.get(n["type"], MUTED)
        interesting = {k: v for k, v in (n.get("params") or {}).items()
                       if k in _KEY_PARAMS}
        chips.append(html.Span(
            n["agent"].replace("Agent", ""),
            title=json.dumps(interesting or n.get("params") or {}, indent=2),
            style={"backgroundColor": colour, "color": "#fff", "padding": "4px 10px",
                   "borderRadius": "4px", "fontSize": "11.5px", "fontWeight": 600,
                   "whiteSpace": "nowrap", "cursor": "help"}))
    return html.Div(chips, style={"display": "flex", "alignItems": "center",
                                  "flexWrap": "wrap", "rowGap": "6px"})


def posture_chips(plan: dict | None):
    """The decision layer the planner actually varies, as labelled values."""
    if not plan:
        return html.Div()
    p = plan.get("posture") or {}
    cap, flagged = p.get("cap_pos"), p.get("flagged") or []
    items = [
        ("Position cap", "default 25%" if cap is None else f"{float(cap) * 100:.0f}%",
         MUTED if cap is None else NAVY),
        ("Sizing", str(p.get("sizing") or "n/a"), TEAL),
        ("Flagged", ", ".join(flagged) if flagged else "none",
         AMBER if flagged else MUTED),
        ("Reduce-only", "yes" if p.get("reduce_only") else "no",
         BEAR if p.get("reduce_only") else MUTED),
    ]
    return html.Div([
        html.Div([html.Div(k, style={"fontSize": "10px", "color": MUTED,
                                     "textTransform": "uppercase",
                                     "letterSpacing": "0.05em"}),
                  html.Div(v, style={"fontSize": "13px", "fontWeight": 650, "color": c})],
                 style={"marginRight": "22px"})
        for k, v, c in items
    ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"})


# ---- tab 6: planner anatomy --------------------------------------------------
def cap_hist_fig():
    c = PLAN_IDX.cap_pos.dropna()
    c = c[c <= 1.0] * 100
    fig = go.Figure(go.Histogram(x=c, xbins=dict(start=0, end=26, size=2),
                                 marker_color=NAVY, name="requested cap"))
    fig.add_vline(x=25, line=dict(color=BEAR, dash="dash"),
                  annotation_text="fallback default 25%", annotation_position="top left")
    fig.update_layout(title="Per-position cap the planner asked for",
                      xaxis_title="cap (%)", yaxis_title="months", bargap=0.06)
    return _fig(fig, 300, legend=False)


def sizing_mix_fig():
    top = PLAN_IDX.sizing.fillna("n/a").value_counts().head(6).index.tolist()
    d = PLAN_IDX.assign(s=PLAN_IDX.sizing.fillna("n/a"))
    d = d[d.s.isin(top)]
    ct = pd.crosstab(d.s, d.regime, normalize="columns") * 100
    fig = go.Figure()
    for reg, col in (("bull", BULL), ("bear", BEAR)):
        if reg in ct.columns:
            fig.add_bar(y=ct.index, x=ct[reg], name=reg, orientation="h",
                        marker_color=col)
    fig.update_layout(title="Sizing rule chosen, by regime (% of months)",
                      barmode="group", xaxis_title="% of months in regime")
    return _fig(fig, 300)


def planner_table():
    g = PLAN_IDX.groupby("regime")
    rows = [("Months", {r: f"{int(v)}" for r, v in g.size().items()}),
            ("Mean position cap requested",
             {r: ("n/a" if pd.isna(v) else f"{v * 100:.0f}%")
              for r, v in g.cap_pos.apply(lambda s: s[s <= 1].mean()).items()}),
            ("Mean names flagged", {r: f"{v:.2f}" for r, v in g.n_flagged.mean().items()}),
            ("Most common sizing",
             {r: (s.mode().iat[0] if not s.mode().empty else "n/a")
              for r, s in g.sizing}),
            ("Distinct DAG topologies", {r: "1" for r in g.groups}),
            ("Planner fallbacks", {r: f"{int(v)}" for r, v in g.used_fallback.sum().items()})]
    regs = [r for r in ("bear", "bull") if r in PLAN_IDX.regime.unique()]
    head = html.Tr([html.Th("", style={"textAlign": "left"})] +
                   [html.Th(r.upper(), style={"textAlign": "right", "padding": "0 14px",
                                              "color": BEAR if r == "bear" else BULL})
                    for r in regs])
    body = [html.Tr([html.Td(label, style={"padding": "5px 0", "color": MUTED})] +
                    [html.Td(vals.get(r, "n/a"),
                             style={"textAlign": "right", "padding": "5px 14px",
                                    "fontWeight": 600, "fontVariantNumeric": "tabular-nums"})
                     for r in regs])
            for label, vals in rows]
    return html.Table([html.Thead(head), html.Tbody(body)],
                      style={"width": "100%", "fontSize": "13px",
                             "borderCollapse": "collapse"})


# ---- the memory channel ------------------------------------------------------
_CONF_RE = re.compile(r"Conf: ([0-9.]+)x")
_WARN_RE = re.compile(r"!! LOW CONFIDENCE \(consider skipping\): (.+)")


@lru_cache(maxsize=1)
def memory_series() -> pd.DataFrame:
    """Per-month spread of the recall block's confidence figure, plus the
    ``LOW CONFIDENCE`` warning. Parsed from the briefings themselves, so it
    reflects exactly what the model was shown."""
    rows = []
    for f in sorted(SESSIONS.glob("session_*.md")):
        txt = f.read_text(encoding="utf-8")
        conf = [float(x) for x in _CONF_RE.findall(txt)]
        if not conf:
            continue
        warn = _WARN_RE.search(txt)
        rows.append({"date": pd.Timestamp(f.stem.replace("session_", "") + "-01"),
                     "lo": min(conf), "hi": max(conf),
                     "warned": bool(warn),
                     "warn_symbols": warn.group(1).strip() if warn else ""})
    return pd.DataFrame(rows)


def conf_band_fig():
    m = memory_series()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.date, y=m.hi, line=dict(color=NAVY, width=1),
                             name="highest-confidence name"))
    fig.add_trace(go.Scatter(x=m.date, y=m.lo, line=dict(color=NAVY, width=1),
                             fill="tonexty", fillcolor="rgba(31,58,102,0.13)",
                             name="lowest-confidence name"))
    fig.add_hline(y=0.4 + 0.8 * 0.4, line=dict(color=BEAR, dash="dot"),
                  annotation_text="skip threshold (win-rate 40%)",
                  annotation_position="bottom right")
    w = m[m.warned]
    if not w.empty:
        fig.add_vrect(x0=w.date.min(), x1=w.date.max(), fillcolor=AMBER, opacity=0.12,
                      line_width=0, annotation_text="the only window the flag fires",
                      annotation_position="top left")
    fig.update_layout(title="Endogenous memory: confidence spread across the 8 names",
                      yaxis_title="Conf ×")
    return _fig(fig, 310)


def memory_panel():
    m = memory_series()
    n, warned = len(m), m[m.warned]
    last = warned.date.max() if not warned.empty else None
    late = m[m.date >= "2015-01-01"]
    excl = PLAN_IDX[PLAN_IDX.n_flagged > 0] if HAS_PLANS else pd.DataFrame()
    hit = 0
    if not excl.empty and not warned.empty:
        wset = set(warned.date.dt.strftime("%Y-%m"))
        hit = int(excl.decision_date.dt.strftime("%Y-%m").isin(wset).sum())

    bullets = [
        f"The recall block's confidence is not independent evidence: it is "
        f"0.4 + 0.8 × win-rate, the same statistic rescaled. It is never read by the "
        f"weight mapper; the “×” suffix multiplies nothing.",
        f"Because win-rate is cumulative over every month ever held, the spread "
        f"collapses as the ledger fills: {late.lo.min():.2f}–{late.hi.max():.2f} across "
        f"all eight names from 2015 on. It stops discriminating.",
        f"The one actionable channel, the “consider skipping” warning, fired in "
        f"{len(warned)} of {n} months, "
        + (f"none after {last:%B %Y}. " if last is not None else "")
        + "A cumulative win-rate above 40% cannot be dragged back under it by any "
          "single bad month, so the trigger becomes unreachable in year three of a "
          "twenty-one-year run.",
    ]
    if not excl.empty:
        bullets.append(
            f"The model excluded names in {len(excl)} months, but only {hit} of those "
            f"coincided with the warning. It largely ignores the flag and reads the "
            f"win-rate and average P&L directly. The exclusions cluster in "
            f"{', '.join(excl.decision_date.dt.strftime('%Y-%m').head(5))} and later "
            f"around XOM's decline.")
    return html.Ul([html.Li(b) for b in bullets],
                   style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.6",
                          "margin": 0, "paddingLeft": "18px"})


# ---- app ---------------------------------------------------------------------
app = Dash(__name__, title="FinAgent portfolio in action")

n_months = len(IDX)
marks = {i: IDX.decision_date.iloc[i].strftime("%Y")
         for i in range(0, n_months, 24)}

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                             "fontFamily": FONT, "color": INK,
                             "padding": "18px 26px"}, children=[

    html.H2("FinAgent: an agentic portfolio, month by month",
            style={"margin": "0 0 2px 0", "color": NAVY}),
    html.Div(f"Replay of {len(IDX)} autonomous monthly decisions, "
             f"{IDX.decision_date.iloc[0]:%Y-%m} to {IDX.decision_date.iloc[-1]:%Y-%m}. "
             "Each month the agent read the briefing shown below and returned a plan "
             "that a deterministic mapping turned into these weights. The run is held "
             "to the window both regime signals cover, so every figure here is "
             "comparable with the regime swap on tab 7.",
             style={"color": MUTED, "fontSize": "13px", "marginBottom": "14px"}),

    dcc.Tabs(id="tabs", value="t1", children=[
        dcc.Tab(label="1 · In action", value="t1"),
        dcc.Tab(label="2 · Autonomy", value="t2"),
        dcc.Tab(label="3 · Trust", value="t3"),
        dcc.Tab(label="4 · Performance", value="t4"),
        dcc.Tab(label="5 · AI vs traditional & RS", value="t5"),
        dcc.Tab(label="6 · Planner anatomy", value="t6"),
        dcc.Tab(label="7 · Regime swap", value="t7"),
    ]),
    html.Div(id="body", style={"marginTop": "16px"}),

    dcc.Interval(id="tick", interval=700, disabled=True),
    dcc.Store(id="month", data=0),
])


SWAP_THEME = {
    "CARD": CARD, "LINE": LINE, "INK": INK, "MUTED": MUTED, "NAVY": NAVY,
    "BEAR": BEAR, "AMBER": AMBER, "TEAL": TEAL,
    "fig": _fig, "log_axis": _log_wealth_axis, "stat": stat,
}


@app.callback(Output("body", "children"), Input("tabs", "value"))
def render(tab):
    if tab == "t7":
        return SWAP.panel(SWAP_THEME)

    if tab == "t1":
        return html.Div([
            html.Div([
                html.Button("▶  Play", id="play", n_clicks=0,
                            style={"padding": "8px 20px", "fontSize": "14px",
                                   "fontWeight": 600, "cursor": "pointer",
                                   "border": f"1px solid {NAVY}", "borderRadius": "5px",
                                   "backgroundColor": NAVY, "color": "#fff"}),
                html.Span(id="month-label",
                          style={"marginLeft": "18px", "fontSize": "19px",
                                 "fontWeight": 650}),
                html.Span(id="regime-badge", style={"marginLeft": "14px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

            dcc.Slider(id="slider", min=0, max=n_months - 1, step=1, value=0,
                       marks=marks, tooltip={"placement": "bottom"},
                       updatemode="drag"),

            html.Div(id="stats", style={"display": "flex", "gap": "10px",
                                        "margin": "14px 0"}),

            html.Div([
                html.Div([html.Div("Portfolio the agent built",
                                   style={"fontWeight": 600, "marginBottom": "4px"}),
                          dcc.Graph(id="g-weights", config={"displayModeBar": False})],
                         style={**CARD, "flex": "3"}),
                html.Div([html.Div("P(bull) from the regime filter",
                                   style={"fontWeight": 600, "marginBottom": "4px"}),
                          dcc.Graph(id="g-gauge", config={"displayModeBar": False}),
                          html.Div(id="dropped-note",
                                   style={"fontSize": "12px", "color": MUTED,
                                          "marginTop": "6px"})],
                         style={**CARD, "flex": "1"}),
            ], style={"display": "flex", "gap": "10px"}),

            html.Div([
                html.Div([
                    html.Div("SESSION.md: exactly what the agent read this month",
                             style={"fontWeight": 600, "marginBottom": "6px"}),
                    html.Pre(id="session-md", style=MONO),
                ], style={**CARD, "flex": "1"}),

                html.Div([
                    html.Div([
                        html.Span("The plan it returned",
                                  style={"fontWeight": 600}),
                        html.Span(id="plan-meta",
                                  style={"marginLeft": "10px", "fontSize": "11px",
                                         "color": MUTED}),
                    ], style={"marginBottom": "8px"}),
                    html.Div(id="dag-chain"),
                    html.Div(id="posture-row"),
                    html.Details([
                        html.Summary("Raw model response",
                                     style={"cursor": "pointer", "fontSize": "12px",
                                            "color": NAVY, "marginTop": "12px",
                                            "fontWeight": 600}),
                        html.Pre(id="plan-raw", style={**MONO, "maxHeight": "220px",
                                                       "marginTop": "8px"}),
                    ]),
                ], style={**CARD, "flex": "1"}),
            ], style={"display": "flex", "gap": "10px", "marginTop": "10px"}),

            html.Div("Hover a node to see the parameters the model set on it. The seven-stage "
                     f"chain is identical in all {len(PLAN_IDX)} months. What the planner "
                     "varies is the "
                     "risk posture inside the nodes, not the shape of the pipeline.",
                     style={"color": MUTED, "fontSize": "12px", "marginTop": "8px"}),
        ])

    if tab == "t2":
        bull_m = IDX[IDX.regime == "bull"]["gross_exposure"].mean() * 100
        bear_m = IDX[IDX.regime == "bear"]["gross_exposure"].mean() * 100
        bear = IDX[IDX.regime == "bear"]
        cut = bear[bear.gross_exposure < 0.99]
        deepest = bear.loc[bear.gross_exposure.idxmin()]
        return html.Div([
            html.Div([
                stat("Avg exposure · BULL", f"{bull_m:.1f}%", BULL),
                stat("Avg exposure · BEAR", f"{bear_m:.1f}%", BEAR),
                stat("Bear months de-risked", f"{len(cut)} / {len(bear)}", AMBER),
                stat("Deepest cut", f"{deepest.gross_exposure*100:.0f}%  "
                                    f"({deepest.decision_date:%b %Y})", NAVY),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
            html.Div([
                html.Div("The posture is not fixed, but the response is episodic "
                         "rather than mechanical.",
                         style={"fontWeight": 600, "marginBottom": "4px"}),
                html.Div(
                    f"In {len(bear) - len(cut)} of {len(bear)} detected bear months the "
                    f"agent stayed fully invested; the bear median is 100%. What moves the "
                    f"average down to {bear_m:.1f}% is a minority of months where it cut "
                    f"hard, down to {deepest.gross_exposure*100:.0f}% invested, and those "
                    "cluster in the genuinely severe episodes: five consecutive months "
                    "through the 2008–09 crisis, and again in August 2022. Read the box "
                    "plot on the right for this: the bear box sits at 100% with a long "
                    "lower tail, not a shifted centre. A language-model decoration would "
                    "show a flat line; a mechanical regime rule would shift the whole box. "
                    "This does neither.",
                    style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.55"}),
            ], style={**CARD, "marginBottom": "10px"}),
            html.Div([html.Div(dcc.Graph(figure=exposure_fig()), style={**CARD, "flex": "2"}),
                      html.Div(dcc.Graph(figure=exposure_box()), style={**CARD, "flex": "1"})],
                     style={"display": "flex", "gap": "10px"}),
        ])

    if tab == "t3":
        return html.Div([
            html.Div([
                stat("Decisions", f"{len(IDX)}"),
                stat("Fallbacks used", f"{int(IDX.used_fallback.sum())}", BULL),
                stat("Mean turnover", f"{IDX.turnover.mean():.2f}"),
                stat("Cost charged", "10 bps", MUTED),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
            html.Div(f"Every one of the {len(PLAN_IDX)} plans was well-formed, so the "
                     "deterministic "
                     "safe-default DAG was never needed. Turnover stays moderate, so the "
                     "edge is not an artefact of frantic trading.",
                     style={"color": MUTED, "fontSize": "13px", "marginBottom": "10px"}),
            html.Div([html.Div(dcc.Graph(figure=reliability_fig()), style={**CARD, "flex": "1"}),
                      html.Div(dcc.Graph(figure=turnover_fig()), style={**CARD, "flex": "2"})],
                     style={"display": "flex", "gap": "10px"}),
        ])

    if tab == "t5":
        if RS is None:
            return html.Div("Run  python -m finagent.research.agentic_eval.rs_optimal  "
                            "to build the regime-switching benchmark.",
                            style={**CARD, "color": BEAR})
        ag_r = AGENT_GROSS[RS.regime == "bear"].mean() * 100
        rs_r = RS.loc[RS.regime == "bear", "gross_rs"].mean() * 100
        return html.Div([
            html.Div("Following Markose & Yang (EC907, Note 4): the market follows a "
                     "two-state Hamilton process, each stock is regime-dependent only "
                     "through its CAPM beta, and the one-step-ahead conditional moments "
                     "(Eqs. 15–17) feed a long-only mean-variance programme re-solved "
                     "every month. All four strategies run on the identical eight stocks "
                     "and identical months, point-in-time.",
                     style={"color": MUTED, "fontSize": "13px", "marginBottom": "12px"}),
            html.Div([
                stat("Window", f"{CMP.index[0]:%Y} – {CMP.index[-1]:%Y} · {len(CMP)} mo",
                     MUTED),
                stat("FinAgent · bear exposure", f"{ag_r:.0f}%", NAVY),
                stat("RS optimal · bear exposure", f"{rs_r:.0f}%", AMBER),
                stat("Regime conditioning gain",
                     f"{CMP['RS dynamic optimal'].iloc[-1]/CMP['Traditional static M-V'].iloc[-1]:.1f}× "
                     f"vs static", AMBER),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
            html.Div([
                html.Span(f"Why {len(CMP)} months and not {len(BT)}: ",
                          style={"fontWeight": 600}),
                f"the RS benchmark needs {36} months of CAPM history before it will "
                f"trust a beta, so it cannot start until {CMP.index[0]:%Y}. All four "
                "strategies are therefore truncated to that common window, and the "
                "figures here are not comparable with the numbers on the Performance "
                "tab: same agent, shorter run.",
            ], style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.55",
                      "marginBottom": "12px"}),
            html.Div([
                html.Span("Reading the result: ", style={"fontWeight": 600}),
                "the regime-switching optimal portfolio reproduces its published claim: "
                "conditioning on the regime beats the same optimiser run on unconditional "
                "moments by a wide margin, which is the benchmark working as intended "
                "rather than being strawmanned. Against the agent it earns more in "
                "absolute terms, but at higher volatility, landing both at effectively "
                "the same Sharpe. The distinction worth drawing is not performance but "
                "model dependence: RS earns its return by going almost entirely to cash "
                f"in the {int((RS.regime == 'bear').sum())} months the Hamilton filter "
                "calls bear, so it is a concentrated bet on that filter being right.",
            ], style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.55",
                      "marginBottom": "12px"}),
            html.Div([html.Div("What each strategy actually does",
                               style={"fontWeight": 600, "marginBottom": "10px"}),
                      style_cards()], style={**CARD, "marginBottom": "10px"}),
            html.Div(dcc.Graph(figure=cmp_wealth_fig()), style=CARD),
            html.Div([html.Div(dcc.Graph(figure=cmp_exposure_fig()), style={**CARD, "flex": "2"}),
                      html.Div(dcc.Graph(figure=cmp_regime_bar()), style={**CARD, "flex": "1"})],
                     style={"display": "flex", "gap": "10px", "marginTop": "10px"}),
            html.Div([html.Div("Same universe, same months, same point-in-time discipline",
                               style={"fontWeight": 600, "marginBottom": "8px"}),
                      cmp_table()], style={**CARD, "marginTop": "10px"}),

            html.Div([
                html.Div("Regime-switching parameters behind the RS benchmark",
                         style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Div(
                    "The Hamilton model is estimated once, on market excess returns up "
                    f"to {pd.Timestamp('2005-04-30'):%B %Y}, and then applied causally; "
                    "it is not refitted each month. The bull-state volatility is "
                    "constrained to 14 percent a year, which is what identifies which "
                    "state is which rather than leaving the labels to chance. The "
                    "coefficients below are read from the benchmark's own output file, "
                    "so they are the values that produced the weights, not a "
                    "re-estimation.",
                    style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.55",
                           "marginBottom": "12px"}),
                html.Div([
                    html.Div([rs_param_table(),
                              html.Div(style={"height": "14px"}),
                              rs_transition_note()],
                             style={"flex": "1 1 40%", "minWidth": "300px"}),
                    html.Div(dcc.Graph(figure=rs_moments_fig(),
                                       config={"displayModeBar": False}),
                             style={"flex": "1 1 55%", "minWidth": "320px"}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px"}),
                html.Div(
                    "Read the two bars together. The state parameters are what the "
                    "filter estimates; the Eq. 15-16 values are what the optimiser is "
                    "handed. They differ because both states are persistent, so next "
                    "month's expected moment is a probability-weighted blend of the two "
                    "states rather than the current one. The means pull toward each "
                    "other and the variances gain a regime-uncertainty term. That blend "
                    "is the mechanism by which the portfolio is dynamic rather than "
                    "merely regime-labelled.",
                    style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.55",
                           "marginTop": "10px"}),
                html.Details([
                    html.Summary("The estimating equations, for reference",
                                 style={"cursor": "pointer", "fontWeight": 600,
                                        "fontSize": "12.5px", "color": AMBER,
                                        "marginTop": "14px"}),
                    html.Div(rs_equations(),
                             style={"borderLeft": f"3px solid {AMBER}",
                                    "paddingLeft": "14px", "marginTop": "10px"}),
                ], open=True),
            ], style={**CARD, "marginTop": "10px"}),
        ])

    if tab == "t6":
        if not HAS_PLANS:
            return html.Div("No planner capture on disk. Run "
                            "`python -m finagent.research.agentic_eval.capture_plans` "
                            "to populate this tab.",
                            style={**CARD, "color": MUTED})
        n = len(PLAN_IDX)
        caps = PLAN_IDX.cap_pos
        return html.Div([
            html.Div([
                stat("Plans captured", f"{n}", NAVY),
                stat("Planner fallbacks", f"{int(PLAN_IDX.used_fallback.sum())}", BULL),
                stat("Distinct DAG topologies", f"{PLAN_IDX.n_nodes.nunique()}", BEAR),
                stat("Distinct sizing rules", f"{PLAN_IDX.sizing.nunique()}", TEAL),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),

            html.Div([
                html.Div("What the model varies, and what it doesn't",
                         style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Div(
                    f"Across all {n} months the planner returned the same seven-stage "
                    "chain every single time: one distinct node sequence, no branching, "
                    "no parallel stages. On that evidence the claim to make is not that "
                    "the agent rewrites its pipeline. What it does vary, and varies with "
                    f"the regime, is risk posture: it names {PLAN_IDX.sizing.nunique()} "
                    "different sizing rules and moves the position cap and exclusion list "
                    "in step with the detected state.",
                    style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.55"}),
            ], style={**CARD, "marginBottom": "10px"}),

            html.Div([html.Div(dcc.Graph(figure=cap_hist_fig()), style={**CARD, "flex": "1"}),
                      html.Div(dcc.Graph(figure=sizing_mix_fig()), style={**CARD, "flex": "1"})],
                     style={"display": "flex", "gap": "10px"}),

            html.Div([html.Div("Posture by regime", style={"fontWeight": 600,
                                                           "marginBottom": "8px"}),
                      planner_table()], style={**CARD, "marginTop": "10px"}),

            html.Div([
                html.Div("What the briefing actually tells the model",
                         style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Div("Of the three numbers per candidate, only one carries "
                         "month-to-month information. Score is an Alpha158 six-factor "
                         "composite, cross-sectionally rescaled so the best name each "
                         "month is pinned at 1.00 and the worst at 0.05, a within-month "
                         "ranking, not a forecast. Win-rate and confidence are the same "
                         "endogenous statistic printed twice.",
                         style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.55",
                                "marginBottom": "10px"}),
                dcc.Graph(figure=conf_band_fig(), config={"displayModeBar": False}),
                memory_panel(),
            ], style={**CARD, "marginTop": "10px"}),

            html.Div([
                html.Div("Two caveats", style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Ul([
                    html.Li("These plans were captured by replaying each archived briefing "
                            "through the planner. They are new, representative responses to "
                            "the same inputs, not the specific DAGs that produced the "
                            "recorded NAV path."),
                    html.Li(f"The planner named no recognised position cap in "
                            f"{int(caps.isna().sum())} of {n} months, so those fell back to "
                            "the hard-coded 25%, looser than almost every cap it does ask "
                            f"for. A further {int((caps > 1).sum())} emitted the cap in "
                            "percent form (15 rather than 0.15), which the weight mapper "
                            "reads as effectively uncapped."),
                ], style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.6",
                          "margin": 0, "paddingLeft": "18px"}),
            ], style={**CARD, "marginTop": "10px"}),
        ])

    ag, ew = PERF.agent, PERF.equal_weight
    ag_dd = (ag / ag.cummax() - 1).min() * 100
    ew_dd = (ew / ew.cummax() - 1).min() * 100
    return html.Div([
        html.Div([
            stat("FinAgent final", f"{ag.iloc[-1]:.0f}×", NAVY),
            stat("Equal-weight final", f"{ew.iloc[-1]:.0f}×", TEAL),
            stat("FinAgent max DD", f"{ag_dd:.1f}%", NAVY),
            stat("Equal-weight max DD", f"{ew_dd:.1f}%", TEAL),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),
        html.Div(dcc.Graph(figure=wealth_fig()), style=CARD),
        html.Div(dcc.Graph(figure=drawdown_fig()), style={**CARD, "marginTop": "10px"}),
    ])


@app.callback(Output("tick", "disabled"), Output("play", "children"),
              Input("play", "n_clicks"), prevent_initial_call=True)
def toggle(n):
    playing = n % 2 == 1
    return (not playing), ("❚❚  Pause" if playing else "▶  Play")


@app.callback(Output("slider", "value"), Input("tick", "n_intervals"),
              State("slider", "value"), prevent_initial_call=True)
def advance(_, cur):
    return 0 if cur is None or cur >= n_months - 1 else cur + 1


@app.callback(
    Output("month-label", "children"), Output("regime-badge", "children"),
    Output("stats", "children"), Output("g-weights", "figure"),
    Output("g-gauge", "figure"), Output("dropped-note", "children"),
    Output("session-md", "children"), Output("dag-chain", "children"),
    Output("posture-row", "children"), Output("plan-raw", "children"),
    Output("plan-meta", "children"),
    Input("slider", "value"))
def update(i):
    row = IDX.iloc[int(i or 0)]
    d = row.decision_date
    is_bear = row.regime == "bear"
    badge = html.Span(
        row.regime.upper(),
        style={"backgroundColor": BEAR if is_bear else BULL, "color": "#fff",
               "padding": "3px 12px", "borderRadius": "11px", "fontSize": "12px",
               "fontWeight": 700, "letterSpacing": "0.05em"})

    stats = [
        stat("NAV (×$1)", f"{row.nav_at_decision:.1f}×", NAVY),
        stat("Invested", f"{row.gross_exposure*100:.0f}%",
             BEAR if row.gross_exposure < 0.9 else NAVY),
        stat("Cash", f"{row.w_cash*100:.0f}%", AMBER),
        stat("Turnover", f"{row.turnover:.2f}"),
        stat("Next-month return", f"{row.net_ret_next*100:+.1f}%",
             BULL if row.net_ret_next >= 0 else BEAR),
        stat("Decision", f"{int(i)+1} / {n_months}", MUTED),
    ]

    note = (f"Dropped this month: {row.dropped}" if row.n_dropped
            else "No name excluded this month.")

    try:
        md = (SESSIONS / row.session_file).read_text(encoding="utf-8")
    except OSError:
        md = "(briefing not found; run regen_sessions)"

    plan = read_plan(d) if HAS_PLANS else None
    if plan:
        u = plan.get("usage") or {}
        meta = (f"{plan.get('model', '')} · {u.get('output_tokens', 0)} output tokens · "
                f"{(u.get('latency_ms') or 0) / 1000:.1f}s"
                + ("  · FALLBACK" if plan.get("used_fallback") else ""))
        raw = plan.get("raw_response") or "(no raw response recorded)"
    else:
        meta, raw = "", "(no captured plan; run capture_plans)"

    return (d.strftime("%B %Y"), badge, stats, weights_fig(row),
            gauge_fig(row.p_bull), note, md, dag_chain(plan),
            posture_chips(plan), raw, meta)


def main() -> int:
    print(f"FinAgent portfolio dashboard  ->  http://127.0.0.1:{PORT}")
    app.run(debug=False, host="127.0.0.1", port=PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
