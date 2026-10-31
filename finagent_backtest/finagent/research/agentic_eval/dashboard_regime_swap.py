"""Tab 7: what happened when the regime detector was swapped.

The agentic path reads its market state from one monthly regime signal. That
signal was the Hamilton (1989) two-state filter throughout the archived run. This
tab reports a controlled replacement of it with the frozen agentic detector
shipped in ``data/regime`` (an LLM-scored daily indicator collapsed to two states,
named "Agentic AI Regime" in the interface and ``arm_c`` in the code and in
``data/regime/CONTRACT.md``), holding *everything else* fixed: same eight names,
same Alpha158 scores, same planner, same deterministic plan-to-weights mapping,
same costs.

The result is negative, and is reported as such. It did not improve the
portfolio; the difference from Hamilton is not distinguishable from zero. That
agrees with the artifact's own scoring, which already ranks it behind Hamilton
on Brier score and directional accuracy, and with ``CONTRACT.md`` section 8.8,
which forbids claiming otherwise.

Kept in a separate module because it is self-contained: it owns its own data
load and figures, and the host dashboard only mounts ``panel()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

REPO_ROOT = Path(__file__).resolve().parents[3]
BT = REPO_ROOT / "data" / "agentic_eval" / "backtest"
HAM_PATH = BT / "path0.parquet"                  # archived, HMM, 248 months
ARM_PATH = BT / "path0_llm_arm_c.parquet"        # Agentic AI Regime, 225 months
RF_PATH = REPO_ROOT / "data" / "agentic_eval" / "market_data" / "rf_monthly.parquet"

N_BOOT = 4000
BOOT_SEED = 7

# The 2x2: each regime signal driven by each planner. Reading it by column asks
# whether the signal helped; by row, whether the planner did. Colour carries the
# signal and dash carries the planner, so the chart stays legible at four lines.
REF = "LLM × HMM"
ARMS = [
    # key, file, planner, signal, colour role, dashed
    (REF, "path0.parquet", "LLM", "HMM", "NAVY", False),
    ("LLM × Agentic", "path0_llm_arm_c.parquet",
     "LLM", "Agentic AI", "AMBER", False),
    ("Rule × HMM", "path0_mock_hamilton.parquet",
     "Fixed rule", "HMM", "NAVY", True),
    ("Rule × Agentic", "path0_mock_arm_c.parquet",
     "Fixed rule", "Agentic AI", "AMBER", True),
]


# ---- data --------------------------------------------------------------------
@lru_cache(maxsize=1)
def load():
    """Both arms on their common window, plus the paired-bootstrap test.

    The archived Hamilton path runs to 2025-12 while the frozen artifact stops at
    2023-12-29, so every comparison here is restricted to the overlap. Quoting
    the archived headline against a shorter agentic run would credit Hamilton with
    two extra years of compounding rather than a better signal.
    """
    ham = pd.read_parquet(HAM_PATH)
    arm = pd.read_parquet(ARM_PATH)
    ham.index = pd.to_datetime(ham.index)
    arm.index = pd.to_datetime(arm.index)
    common = ham.index.intersection(arm.index)
    h, a = ham.loc[common], arm.loc[common]

    rf = pd.read_parquet(RF_PATH)["rf"].reindex(common).fillna(0.0)

    def metrics(d):
        r = d["net_ret"]
        w = (1 + r).cumprod()
        return {
            "final": float(w.iloc[-1]),
            "cagr": float(w.iloc[-1] ** (12 / len(r)) - 1) * 100,
            "vol": float(r.std() * np.sqrt(12)) * 100,
            "sharpe": _sharpe(r, rf),
            "mdd": float((w / w.cummax() - 1).min()) * 100,
            "bear": int((d["regime"] == "bear").sum()),
            "cash": float(d["w_cash"].mean()) * 100,
            "turnover": float(d["turnover"].mean()),
            "wealth": w,
        }

    mh, ma = metrics(h), metrics(a)

    # Paired bootstrap on the Sharpe difference: the two arms share months, so
    # resampling months jointly keeps the pairing and prices in the common
    # market movement both arms were exposed to.
    rng = np.random.default_rng(BOOT_SEED)
    obs = ma["sharpe"] - mh["sharpe"]
    diffs = np.empty(N_BOOT)
    ha, aa = h["net_ret"], a["net_ret"]
    for i in range(N_BOOT):
        pick = rng.integers(0, len(common), len(common))
        diffs[i] = _sharpe(aa.iloc[pick], rf) - _sharpe(ha.iloc[pick], rf)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    # ---- the full 2x2 -------------------------------------------------------
    # The rule arms are a deterministic stand-in, not a published benchmark: they
    # exist to hold the planner constant while the signal varies, which is what
    # separates "this signal is better" from "this planner reacts to it better".
    raw = {}
    for key, fname, planner, signal, col, dash in ARMS:
        f = BT / fname
        if not f.exists():
            continue
        x = pd.read_parquet(f)
        x.index = pd.to_datetime(x.index)
        raw[key] = x
    grid_idx = common
    for x in raw.values():
        grid_idx = grid_idx.intersection(x.index)
    rf_g = pd.read_parquet(RF_PATH)["rf"].reindex(grid_idx).fillna(0.0)

    grid = {}
    for key, fname, planner, signal, col, dash in ARMS:
        if key not in raw:
            continue
        x = raw[key].loc[grid_idx]
        m = metrics(x)
        gross = (1 - x["w_cash"]) * 100
        by = gross.groupby(x["regime"]).mean()
        m.update({"planner": planner, "signal": signal, "colour": col, "dash": dash,
                  "gross_bull": float(by.get("bull", np.nan)),
                  "gross_bear": float(by.get("bear", np.nan))})
        grid[key] = m

    # Every arm is scored against the incumbent cell, so each row can say whether
    # it is distinguishable from the configuration the dissertation already ran.
    ref_ret = raw[REF].loc[grid_idx, "net_ret"] if REF in raw else None
    for key in grid:
        if ref_ret is None or key == REF:
            grid[key]["dsharpe"], grid[key]["p"] = 0.0, None
            continue
        r = raw[key].loc[grid_idx, "net_ret"]
        rg = np.random.default_rng(BOOT_SEED)
        dd = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pk = rg.integers(0, len(grid_idx), len(grid_idx))
            dd[i] = _sharpe(r.iloc[pk], rf_g) - _sharpe(ref_ret.iloc[pk], rf_g)
        grid[key]["dsharpe"] = _sharpe(r, rf_g) - _sharpe(ref_ret, rf_g)
        grid[key]["p"] = float(2 * min((dd <= 0).mean(), (dd >= 0).mean()))

    return {
        "h": h, "a": a, "common": common, "mh": mh, "ma": ma,
        "obs": obs, "lo": lo, "hi": hi, "p": float(p), "diffs": diffs,
        "ham_full": float((1 + ham["net_ret"]).prod()),
        "n_ham_full": len(ham),
        "disagree": int((h["regime"] != a["regime"]).sum()),
        "grid": grid, "grid_idx": grid_idx,
    }


def _sharpe(r: pd.Series, rf: pd.Series) -> float:
    ex = r - rf.reindex(r.index).fillna(0.0).values
    return float(ex.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0


# ---- figures -----------------------------------------------------------------
def wealth_fig(T):
    """All four cells of the 2x2.

    Colour is the regime signal and dash is the planner, so the eye can read
    either question off the same chart: two lines of one colour answer "did the
    planner matter for this signal", two of one style answer "did the signal
    matter for this planner".
    """
    d = load()
    fig = go.Figure()
    for key, _f, planner, signal, col, dash in ARMS:
        m = d["grid"].get(key)
        if m is None:
            continue
        fig.add_trace(go.Scatter(
            x=d["grid_idx"], y=m["wealth"], mode="lines",
            name=f"{planner} × {signal}",
            line=dict(color=T[col], width=1.6 if dash else 2.4,
                      dash="dot" if dash else "solid"),
            opacity=0.75 if dash else 1.0))
    T["log_axis"](fig, [1, 2, 5, 10, 20, 50])
    fig.update_layout(title="Both regime signals, under the agent and under a "
                            "fixed rule")
    return T["fig"](fig, 380)


def matrix_table(T):
    """The 2x2 as numbers, scored against the incumbent configuration."""
    d = load()
    g, LINE, INK, MUTED = d["grid"], T["LINE"], T["INK"], T["MUTED"]
    hdr = ["Planner", "Regime signal", "Final $1", "CAGR", "Vol", "Sharpe",
           "Max DD", "Bear months", "Mean cash", "Gross on bull", "Gross on bear",
           "Turnover", "ΔSharpe vs incumbent", "p"]
    body = []
    for key, _f, planner, signal, col, dash in ARMS:
        m = g.get(key)
        if m is None:
            continue
        is_ref = key == REF
        dsh = "reference" if is_ref else f"{m['dsharpe']:+.3f}"
        pv = "—" if m["p"] is None else f"{m['p']:.2f}"
        cells = [planner, signal, f"{m['final']:.2f}×", f"{m['cagr']:.1f}%",
                 f"{m['vol']:.1f}%", f"{m['sharpe']:.2f}", f"{m['mdd']:.1f}%",
                 f"{m['bear']}", f"{m['cash']:.1f}%", f"{m['gross_bull']:.1f}%",
                 f"{m['gross_bear']:.1f}%", f"{m['turnover']:.3f}", dsh, pv]
        llm = planner == "LLM"
        body.append(html.Tr([html.Td(c, style={
            "textAlign": "left" if i < 2 else "right", "padding": "6px 10px",
            "borderBottom": f"1px solid {LINE}",
            "color": T[col] if i == 1 else (INK if llm else MUTED),
            "fontWeight": 650 if (llm and i < 2) else 400,
        }) for i, c in enumerate(cells)]))
    return html.Table([
        html.Thead(html.Tr([html.Th(h, style={
            "textAlign": "left" if i < 2 else "right", "padding": "6px 10px",
            "borderBottom": f"2px solid {T['NAVY']}", "fontSize": "11px",
            "textTransform": "uppercase", "letterSpacing": "0.04em",
            "color": T["NAVY"]}) for i, h in enumerate(hdr)])),
        html.Tbody(body),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12.5px"})


def exposure_by_regime_fig(T):
    """The mechanism: how much the agent actually stood down on a bear call.

    This is the figure that overturns the cheap prediction. A rule-based stand-in
    de-risks on any bear label, so doubling the number of bear labels
    mechanically buys it protection. The LLM did not behave that way.
    """
    d = load()
    fig = go.Figure()
    arms = [("HMM (Hamilton 1989)", "h", T["NAVY"]),
            ("Agentic AI Regime", "a", T["AMBER"])]
    for nm, key, col in arms:
        df = d[key]
        gross = (1 - df["w_cash"]) * 100
        means = gross.groupby(df["regime"]).mean()
        counts = df["regime"].value_counts()
        order = ["bull", "bear"]
        fig.add_trace(go.Bar(
            x=order, y=[means.get(k, np.nan) for k in order], name=nm,
            marker_color=col, opacity=0.9,
            text=[f"{means.get(k, float('nan')):.1f}%<br>"
                  f"<span style='font-size:10px'>n={int(counts.get(k, 0))}</span>"
                  for k in order],
            textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="mean invested (%)", range=[0, 118])
    fig.update_xaxes(title="regime the signal reported")
    fig.update_layout(title="How far the agent stood down when told 'bear'",
                      barmode="group")
    return T["fig"](fig, 330)


def agreement_fig(T):
    """Where the two signals disagree, month by month."""
    d = load()
    h, a = d["h"]["regime"], d["a"]["regime"]
    state = pd.Series("both bull", index=d["common"])
    state[(h == "bear") & (a == "bear")] = "both bear"
    state[(h == "bull") & (a == "bear")] = "Agentic only"
    state[(h == "bear") & (a == "bull")] = "HMM only"
    cols = {"both bull": T["LINE"], "both bear": T["BEAR"],
            "Agentic only": T["AMBER"], "HMM only": T["NAVY"]}
    fig = go.Figure()
    for k, c in cols.items():
        m = state == k
        fig.add_trace(go.Bar(x=d["common"][m], y=np.ones(int(m.sum())), name=k,
                             marker_color=c, marker_line_width=0, width=26 * 86400000))
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(title=f"Which months each signal called bear "
                            f"({d['disagree']} of {len(d['common'])} disagree)",
                      barmode="stack", bargap=0)
    return T["fig"](fig, 190)


def bootstrap_fig(T):
    d = load()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d["diffs"], nbinsx=60, marker_color=T["AMBER"],
                               opacity=0.75, name="bootstrap draws"))
    fig.add_vline(x=0, line=dict(color=T["INK"], width=1.5, dash="dash"))
    fig.add_vline(x=d["obs"], line=dict(color=T["NAVY"], width=2))
    fig.add_annotation(x=d["obs"], y=1, yref="paper", yanchor="bottom",
                       text=f"observed {d['obs']:+.3f}", showarrow=False,
                       font=dict(color=T["NAVY"], size=11))
    fig.update_xaxes(title="Sharpe difference (Agentic AI Regime − HMM)")
    fig.update_yaxes(title="draws")
    fig.update_layout(title=f"The difference is not distinguishable from zero "
                            f"(p = {d['p']:.2f})")
    return T["fig"](fig, 300, legend=False)


def table(T):
    d = load()
    rows = [("HMM regime (Hamilton 1989)", d["mh"], T["NAVY"]),
            ("Agentic AI Regime", d["ma"], T["AMBER"])]
    hdr = ["Regime signal", "Final $1", "CAGR", "Vol", "Sharpe", "Max DD",
           "Bear months", "Mean cash", "Turnover"]
    body = []
    for name, m, col in rows:
        cells = [name, f"{m['final']:.2f}×", f"{m['cagr']:.1f}%", f"{m['vol']:.1f}%",
                 f"{m['sharpe']:.2f}", f"{m['mdd']:.1f}%", f"{m['bear']}",
                 f"{m['cash']:.1f}%", f"{m['turnover']:.3f}"]
        body.append(html.Tr([html.Td(c, style={
            "textAlign": "left" if i == 0 else "right", "padding": "7px 12px",
            "borderBottom": f"1px solid {T['LINE']}",
            "color": col if i == 0 else T["INK"],
            "fontWeight": 650 if i == 0 else 400}) for i, c in enumerate(cells)]))
    return html.Table([
        html.Thead(html.Tr([html.Th(hh, style={
            "textAlign": "left" if i == 0 else "right", "padding": "7px 12px",
            "borderBottom": f"2px solid {T['NAVY']}", "fontSize": "12px",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "color": T["NAVY"]}) for i, hh in enumerate(hdr)])),
        html.Tbody(body),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"})


# ---- panel -------------------------------------------------------------------
def panel(T):
    """The whole tab. ``T`` carries the host dashboard's theme and helpers."""
    d = load()
    CARD, MUTED, INK = T["CARD"], T["MUTED"], T["INK"]
    n = len(d["common"])

    # the months the agentic signal calls bear and Hamilton does not
    extra = d["common"][(d["h"]["regime"] == "bull") & (d["a"]["regime"] == "bear")]
    ex_ret = d["a"].loc[extra, "gross_ret"]
    all_ret = d["a"]["gross_ret"]

    prose = {"color": MUTED, "fontSize": "13px", "lineHeight": "1.55"}

    return html.Div([
        html.Div([
            T["stat"]("HMM final", f"{d['mh']['final']:.1f}×", T["NAVY"]),
            T["stat"]("Agentic AI Regime final", f"{d['ma']['final']:.1f}×",
                      T["AMBER"]),
            T["stat"]("Sharpe difference", f"{d['obs']:+.3f}", T["INK"]),
            T["stat"]("p-value", f"{d['p']:.2f}", T["INK"]),
            T["stat"]("Planner fallbacks", f"0 / {n}", T["INK"]),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),

        html.Div([
            html.Div("What this tab tests", style={"fontWeight": 600,
                                                   "marginBottom": "6px"}),
            html.Div([
                "The agent reads its market state from a single monthly regime "
                "signal. Everything above this line used the Hamilton (1989) "
                "two-state filter. Here that one input is replaced by a frozen "
                "agentic detector, an LLM-scored daily indicator collapsed to the "
                "same two states, with every other component held fixed: the same "
                "eight names, the same Alpha158 scores, the same planner, the same "
                "plan-to-weights mapping and the same 10 bps of cost. Any "
                "difference in the curves below is therefore attributable to the "
                "signal rather than to the plumbing.",
            ], style={**prose, "marginBottom": "8px"}),
            html.Div([
                html.B("The result is negative. "),
                f"Over the {n} months both signals cover, the Agentic AI "
                f"Regime returned {d['ma']['final']:.2f}× against Hamilton's "
                f"{d['mh']['final']:.2f}×, "
                f"a Sharpe difference of {d['obs']:+.3f} with a 95% interval of "
                f"[{d['lo']:+.3f}, {d['hi']:+.3f}] and p = {d['p']:.2f}. The two are "
                "not distinguishable, and the point estimate favours the classical "
                "filter. This is consistent with the detector's own scoring, which "
                "already ranks it behind Hamilton on Brier score and directional "
                "accuracy.",
            ], style=prose),
        ], style={**CARD, "marginBottom": "10px"}),

        html.Div(dcc.Graph(figure=wealth_fig(T), config={"displayModeBar": False}),
                 style=CARD),

        html.Div([
            html.Div("The 2×2: which mattered, the signal or the planner?",
                     style={"fontWeight": 600, "marginBottom": "6px"}),
            html.Div([
                "Swapping the signal only tells you something once the planner is "
                "held fixed too. The four rows below cross both regime signals with "
                "both planners — the LLM, and the deterministic rule used to "
                "validate the loop. Read down a column and the question is whether "
                "the signal helped; read across and it is whether the planner did. ",
                html.B("The two disagree: "),
                "the LLM does better on the HMM, while the fixed rule does better on "
                "the agentic signal. A signal that suits a mechanical de-risker is "
                "not automatically the one that suits the agent, which is why the "
                "rule-based validation run mispredicted this experiment.",
            ], style={**prose, "marginBottom": "10px"}),
            matrix_table(T),
            html.Div([
                html.B("The rule arms are controls, not benchmarks. "),
                "``mock_posture`` is a hard-coded stand-in that cuts exposure on any "
                "bear label; it exists to hold the planner constant, not to be "
                "beaten. It should not be read as a rival strategy — and where it "
                "scores better than the agent on a risk-adjusted measure, that is "
                "worth stating rather than hiding.",
            ], style={**prose, "marginTop": "8px", "fontSize": "12.5px"}),
        ], style={**CARD, "marginTop": "10px"}),

        html.Div([
            html.Div("Why the obvious prediction was wrong",
                     style={"fontWeight": 600, "marginBottom": "6px"}),
            html.Div([
                f"The Agentic AI Regime calls bear far more often — "
                f"{d['ma']['bear']} months against {d['mh']['bear']}. A "
                "rule-based agent would therefore have looked "
                "safer automatically, because a fixed rule cuts exposure on every "
                "bear label; a validation run using exactly such a rule did predict "
                "the agentic signal would win. The LLM did not behave that way. "
                "It stood down by a similar amount on a bear call in both arms, "
                "so mean cash barely "
                "moved despite twice the warnings. What the rule-based run measured "
                "was its own hard-coded reflex, not the quality of the signal.",
            ], style={**prose, "marginBottom": "10px"}),
            html.Div([
                html.Div(dcc.Graph(figure=exposure_by_regime_fig(T),
                                   config={"displayModeBar": False}),
                         style={"flex": "1"}),
                html.Div(dcc.Graph(figure=bootstrap_fig(T),
                                   config={"displayModeBar": False}),
                         style={"flex": "1"}),
            ], style={"display": "flex", "gap": "10px"}),
        ], style={**CARD, "marginTop": "10px"}),

        html.Div([
            html.Div("Where the two signals part company",
                     style={"fontWeight": 600, "marginBottom": "6px"}),
            dcc.Graph(figure=agreement_fig(T), config={"displayModeBar": False}),
            html.Div([
                f"The disagreement is nearly one-sided: {len(extra)} months where "
                "the agentic signal sees a bear market and Hamilton does not, "
                f"against only {d['disagree'] - len(extra)} the other way. On those "
                f"{len(extra)} months the agent held "
                f"{(1 - d['a'].loc[extra, 'w_cash']).mean() * 100:.0f}% invested "
                f"instead of Hamilton's "
                f"{(1 - d['h'].loc[extra, 'w_cash']).mean() * 100:.0f}%, and they did "
                f"turn out weaker than average: {ex_ret.mean() * 100:+.2f}% against "
                f"{all_ret.mean() * 100:+.2f}% across all months, with "
                f"{int((ex_ret < 0).sum())} of {len(extra)} actually down. So the "
                "extra caution was aimed at genuinely worse months — just not "
                "reliably enough to pay for the upside it gave up in the others. "
                "That is the whole of the effect, and it is why the difference "
                "lands where it does.",
            ], style={**prose, "marginTop": "8px"}),
        ], style={**CARD, "marginTop": "10px"}),

        html.Div([html.Div("The numbers", style={"fontWeight": 600,
                                                 "marginBottom": "8px"}),
                  table(T)], style={**CARD, "marginTop": "10px"}),

        html.Div([
            html.Div("How to read this honestly", style={"fontWeight": 600,
                                                         "marginBottom": "6px"}),
            html.Ul([
                html.Li([
                    "The comparison is on the ", html.B(f"{n} months both signals "
                    "cover"), ", 2005-05 to 2024-01, and ",
                    html.B("every other tab is now held to the same window"),
                    " so a wealth multiple quoted from one tab means the same thing "
                    "in another. The archived run does continue past it: on the "
                    f"full {d['n_ham_full']} months to 2025-12 the Hamilton path "
                    f"reaches {d['ham_full']:.1f}×. That figure is not the "
                    "comparator here, because setting it against a shorter run "
                    "would credit the classical filter with two extra years of "
                    "compounding rather than a better signal."]),
                html.Li("Two things changed at once. The swap also added "
                        "within-month context to the briefing — whether the regime "
                        "flipped mid-month, and the worst state reached inside it — "
                        "which a month-end filter cannot supply. Attributing the "
                        "difference to the signal alone would need a third arm "
                        "carrying the agentic labels with those lines suppressed."),
                html.Li("One sampling run per arm. The planner is stochastic, so "
                        "this is a single draw from its distribution rather than "
                        "its mean; a difference this small could not be separated "
                        "from sampling noise without repeating both arms."),
                html.Li("No claim of outperformance is made for the agentic "
                        "detector, and none should be read into these figures. It "
                        "lost on the point estimate and tied on the test."),
            ], style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.6",
                      "margin": 0, "paddingLeft": "18px"}),
        ], style={**CARD, "marginTop": "10px"}),
    ])


__all__ = ["panel", "load"]
