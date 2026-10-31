"""Phase 2/3: the regime-switching optimal portfolio with an agent in the loop.

The classical model (``rs_optimal.py``) fixes three things by assumption: risk
aversion at 6, a hard 0.5 threshold on the regime probability, and no limit on
concentration. This module leaves the estimation and the optimiser exactly as
they are, and lets a decision-maker set those three parameters each month given
diagnostics about how far the model looks trustworthy.

Four arms share one code path so they are guaranteed comparable:

    fixed    gamma 6, hard label, no cap. Reproduces rs_optimal exactly.
    rule     a deterministic function of the same diagnostics the agent sees.
    random   sampled from the same parameter space, seeded.
    agent    Claude, constrained by the schema in rs_agent_schema.

``fixed`` is the correctness check: if it does not reproduce the archived RS
path, something here is wrong and the other three arms cannot be trusted.

``rs_optimal.py`` is deliberately left untouched because it produces the numbers
reported in the dissertation, so the CAPM helper below is a duplicate of its
nested one rather than a refactor out from under a frozen file.

Run:
    python -m finagent.research.agentic_eval.rs_agentic --mode fixed
    python -m finagent.research.agentic_eval.rs_agentic --mode rule
    python -m finagent.research.agentic_eval.rs_agentic --mode random --seed 0
    python -m finagent.research.agentic_eval.rs_agentic --mode agent --limit 3
"""

from __future__ import annotations

import argparse
import json
import random as _random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from finagent.research.agentic_eval import regime_hamilton as RH
from finagent.research.agentic_eval.rs_agent_schema import (
    CAP_RANGE, DEFAULT, GAMMA_RANGE, JSON_SCHEMA, RSPosture, TREATMENTS, validate)
from finagent.research.agentic_eval.rs_session import memory_block, render

REPO = Path(__file__).resolve().parents[3]
MD = REPO / "data" / "agentic_eval" / "market_data"
OUT = REPO / "data" / "agentic_eval" / "rs_agent"

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]
IN_SAMPLE_END = "2005-04-30"
MIN_OBS = 36
MODEL = "claude-sonnet-4-6"

_SYSTEM = (
    "You set the decision parameters for a regime-switching optimal portfolio. "
    "The econometrics and the optimiser are fixed and correct; what you control "
    "is how far to trust them this month. Be willing to leave the classical "
    "defaults alone (gamma 6, hard_label, no cap) when the diagnostics look "
    "healthy. Deviate when they do not: an unstable filter, a weak or shifting "
    "CAPM fit, two states that barely differ, or an optimiser demanding a very "
    "large single position are all reasons for caution. Reply with the JSON "
    "object only."
)


def mv_optimise(mu, cov, gamma: float, cap: float) -> np.ndarray:
    """Long-only mean-variance, sum(w) <= 1, with a per-name ceiling.

    The cap enters as a bound rather than a post-hoc clip, so the result stays
    the optimal portfolio subject to the constraint rather than a distorted
    version of an unconstrained one.
    """
    n = len(mu)
    cov = cov + np.eye(n) * 1e-10

    def neg_util(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)

    def grad(w):
        return -(mu - gamma * cov @ w)

    res = minimize(neg_util, np.full(n, min(1.0 / n, cap)), jac=grad,
                   method="SLSQP", bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}],
                   options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, cap)
    if w.sum() > 1.0:
        w = w / w.sum()
    return w


def _load():
    stk = pd.read_parquet(MD / "stocks_monthly.parquet")[TICKERS]
    spx = pd.read_parquet(MD / "spx_monthly.parquet")
    rf = pd.read_parquet(MD / "rf_monthly.parquet")["rf"]
    reg = pd.read_parquet(MD / "regime_hamilton.parquet")
    mkt = spx["ret"].astype(float)
    rfm = rf.astype(float).reindex(mkt.index).fillna(0.0)
    return stk, (mkt - rfm).dropna(), rfm, reg


def _capm(stk, mkt_ex, rfm, upto):
    """Point-in-time OLS per stock, plus fit health. Mirrors rs_optimal.capm."""
    win = stk.index[stk.index <= upto]
    a, b, d, r2 = [], [], [], []
    for t in TICKERS:
        y = (stk.loc[win, t].astype(float) - rfm.reindex(win).fillna(0.0)).dropna()
        x = mkt_ex.reindex(y.index)
        ok = x.notna() & y.notna()
        x, y = x[ok].values, y[ok].values
        bb, aa = np.polyfit(x, y, 1)
        resid = y - (aa + bb * x)
        a.append(float(aa))
        b.append(float(bb))
        d.append(float(resid.var(ddof=1)))
        ss = float(((y - y.mean()) ** 2).sum())
        r2.append(1.0 - float((resid ** 2).sum()) / ss if ss > 0 else 0.0)
    return np.array(a), np.array(b), np.diag(d), np.array(r2)


def _hamilton_moments(mkt_ex):
    ins = mkt_ex[mkt_ex.index <= IN_SAMPLE_END]
    par = RH.fit(ins * 100.0, bull_var=(14.0 ** 2) / 12)
    mu1, mu2 = par["mu"][0] / 100.0, par["mu"][1] / 100.0
    s1, s2 = np.sqrt(par["var"][0]) / 100.0, np.sqrt(par["var"][1]) / 100.0
    P, Q = float(par["P"]), float(par["Q"])
    e_m = {1: P * mu1 + (1 - P) * mu2, 2: (1 - Q) * mu1 + Q * mu2}
    S_m = {1: P * s1 ** 2 + (1 - P) * s2 ** 2 + P * (1 - P) * (mu1 - mu2) ** 2,
           2: (1 - Q) * s1 ** 2 + Q * s2 ** 2 + Q * (1 - Q) * (mu1 - mu2) ** 2}
    return e_m, S_m


def _blend(e_m, S_m, p_bull):
    """Weight the two states by P(bull) rather than committing to one.

    The variance carries the same between-state term as Eq. 16, since mixing two
    distributions adds the dispersion of their means.
    """
    p = float(np.clip(p_bull, 0.0, 1.0))
    e = p * e_m[1] + (1 - p) * e_m[2]
    S = p * S_m[1] + (1 - p) * S_m[2] + p * (1 - p) * (e_m[1] - e_m[2]) ** 2
    return e, S


# ---- the four arms ----------------------------------------------------------
def _posture_fixed(ctx, rng=None):
    return DEFAULT, ""


def _posture_rule(ctx, rng=None):
    """Deterministic strawman using exactly the diagnostics the agent sees."""
    certainty = max(ctx["p_bull_path"][-1], 1 - ctx["p_bull_path"][-1])
    unsure = certainty < 0.80
    return RSPosture(
        gamma=10.0 if unsure else 6.0,
        regime_treatment="probability_blend" if unsure else "hard_label",
        max_position_pct=0.25 if ctx["top_weight"] > 0.40 else 1.00,
        rationale="deterministic rule"), ""


_CONST = {"gamma": 6.0, "cap": 0.32}


def _posture_constant(ctx, rng=None):
    """A fixed posture applied every month, no judgement at all.

    This is the control that matters most once the agent turns out to deviate
    almost always: if a constant cap reproduces the agent's Sharpe, then the
    per-month reasoning is decorative and the gain belongs to the constraint,
    not to the agent.
    """
    return RSPosture(gamma=_CONST["gamma"], regime_treatment="hard_label",
                     max_position_pct=_CONST["cap"],
                     rationale="constant posture"), ""


def _posture_random(ctx, rng):
    return RSPosture(
        gamma=rng.uniform(*GAMMA_RANGE),
        regime_treatment=rng.choice(list(TREATMENTS)),
        max_position_pct=rng.uniform(*CAP_RANGE),
        rationale="random"), ""


def _posture_agent(client, ctx, cache_dir, overwrite=False):
    """One agent decision, cached on disk so an interrupted run is not re-paid."""
    stamp = f"{ctx['date']:%Y-%m}"
    f = cache_dir / f"decision_{stamp}.json"
    if f.exists() and not overwrite:
        rec = json.loads(f.read_text(encoding="utf-8"))
        post, _ = validate(rec.get("raw_parsed"))
        return (post or DEFAULT), rec.get("reject_reason", ""), rec, True

    briefing = render(ctx)
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=MODEL, max_tokens=600, system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
        messages=[{"role": "user", "content": briefing}])
    ms = (time.perf_counter() - t0) * 1000.0
    txt = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        parsed = json.loads(txt)
    except json.JSONDecodeError:
        parsed = None
    post, reason = validate(parsed)

    rec = {
        "date": stamp, "model": MODEL, "briefing": briefing,
        "raw_text": txt, "raw_parsed": parsed,
        "accepted": post is not None, "reject_reason": reason,
        "posture": (post or DEFAULT).as_dict(),
        "usage": {"input_tokens": resp.usage.input_tokens,
                  "output_tokens": resp.usage.output_tokens,
                  "latency_ms": round(ms, 1)},
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return (post or DEFAULT), reason, rec, False


def run(mode: str = "fixed", seed: int = 0, limit: int | None = None,
        overwrite: bool = False) -> pd.DataFrame:
    stk, mkt_ex, rfm, reg = _load()
    e_m, S_m = _hamilton_moments(mkt_ex)

    all_m = sorted(d for d in stk.index if d in reg.index)
    months = [d for d in all_m if all_m.index(d) >= MIN_OBS]

    client, cache_dir = None, OUT / "agent_calls"
    if mode == "agent":
        import anthropic
        from finagent.research.agentic_eval.backtest import _load_env
        _load_env()
        client = anthropic.Anthropic()

    rng = _random.Random(seed)
    rows: list[dict] = []
    past_returns: list[float] = []
    prev_beta = None
    calls, spend = 0, 0.0

    for d in months:
        i = all_m.index(d)
        prev = all_m[i - 1]
        a_vec, b_vec, d_idio, r2 = _capm(stk, mkt_ex, rfm, prev)
        label = str(reg.loc[d, "regime"])
        p_bull = float(reg.loc[d, "p_bull_filtered"])
        s = 1 if label == "bull" else 2

        # What the untouched optimiser would do, used purely as a diagnostic.
        mu_un = a_vec + b_vec * e_m[s]
        cov_un = np.outer(b_vec, b_vec) * S_m[s] + d_idio
        w_un = mv_optimise(mu_un, cov_un, 6.0, 1.0)

        past = [all_m[j] for j in range(max(0, i - 5), i + 1)]
        moved = (np.abs(b_vec - prev_beta) if prev_beta is not None
                 else np.zeros(len(TICKERS)))
        k = int(np.argmax(moved))
        ctx = {
            "date": d, "label": label,
            "p_bull_path": [float(reg.loc[m, "p_bull_filtered"]) for m in past],
            "label_path": [str(reg.loc[m, "regime"]) for m in past],
            "e_bull": e_m[1], "e_bear": e_m[2],
            "sd_bull": float(np.sqrt(S_m[1])), "sd_bear": float(np.sqrt(S_m[2])),
            "n_obs": int((stk.index <= prev).sum()),
            "mean_r2": float(r2.mean()),
            "beta_min": float(b_vec.min()), "beta_max": float(b_vec.max()),
            "beta_move_name": TICKERS[k] if prev_beta is not None else "n/a",
            "beta_move": float(b_vec[k] - prev_beta[k]) if prev_beta is not None else 0.0,
            "top_name": TICKERS[int(np.argmax(w_un))],
            "top_weight": float(w_un.max()),
            "uncon_gross": float(w_un.sum()),
            "n_above_20": int((w_un > 0.20).sum()),
            "memory": memory_block(past_returns),
        }

        reason = ""
        if mode == "agent":
            already = (cache_dir / f"decision_{d:%Y-%m}.json").exists()
            if limit is not None and calls >= limit and not already:
                break
            post, reason, rec, cached = _posture_agent(client, ctx, cache_dir,
                                                       overwrite)
            if not cached:
                calls += 1
                u = rec["usage"]
                spend += (u["input_tokens"] * 3 + u["output_tokens"] * 15) / 1e6
                if calls % 20 == 0 or calls == 1:
                    print(f"  [{calls}] {d:%Y-%m} gamma={post.gamma:.1f} "
                          f"cap={post.max_position_pct:.2f} ${spend:.2f}",
                          flush=True)
        else:
            post, reason = {"fixed": _posture_fixed, "rule": _posture_rule,
                            "random": _posture_random,
                            "constant": _posture_constant}[mode](ctx, rng)

        if post.regime_treatment == "probability_blend":
            e_use, S_use = _blend(e_m, S_m, p_bull)
        else:
            e_use, S_use = e_m[s], S_m[s]
        mu = a_vec + b_vec * e_use
        cov = np.outer(b_vec, b_vec) * S_use + d_idio
        w = mv_optimise(mu, cov, post.gamma, post.max_position_pct)

        r = stk.loc[d, TICKERS].astype(float).values
        rf_d = float(rfm.get(d, 0.0))
        ret = float(w @ r + (1 - w.sum()) * rf_d)
        past_returns.append(ret)
        prev_beta = b_vec

        rows.append({
            "date": d, "regime": label, "p_bull": p_bull, "ret": ret,
            "gross": float(w.sum()), "top_weight": float(w.max()),
            "gamma": post.gamma, "treatment": post.regime_treatment,
            "cap": post.max_position_pct,
            "accepted": reason == "", "reject_reason": reason,
            "deviated": bool(post.gamma != 6.0
                             or post.regime_treatment != "hard_label"
                             or post.max_position_pct < 1.0),
            **{f"w_{t}": w[j] for j, t in enumerate(TICKERS)},
        })

    df = pd.DataFrame(rows).set_index("date")
    df.attrs.update(mode=mode, seed=seed, calls=calls, spend=spend)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = mode + (f"_s{seed}" if mode == "random" else "")
    if mode == "constant":
        tag = f"constant_c{int(_CONST['cap'] * 100)}"
    df.to_parquet(OUT / f"path_{tag}.parquet")
    return df


def summarise(df: pd.DataFrame) -> dict:
    r = df["ret"]
    w = (1 + r).cumprod()
    return {"months": len(r), "final": float(w.iloc[-1]),
            "cagr": float(w.iloc[-1] ** (12 / len(r)) - 1) * 100,
            "vol": float(r.std() * np.sqrt(12)) * 100,
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
            "maxdd": float((w / w.cummax() - 1).min()) * 100,
            "deviated": int(df["deviated"].sum()),
            "rejected": int((~df["accepted"]).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["fixed", "rule", "random", "agent", "constant"],
                    default="fixed")
    ap.add_argument("--cap", type=float, default=0.32,
                    help="constant mode: the fixed per-name ceiling")
    ap.add_argument("--gamma", type=float, default=6.0,
                    help="constant mode: the fixed risk aversion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="agent mode only: cap new API calls (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    _CONST["cap"], _CONST["gamma"] = a.cap, a.gamma

    df = run(mode=a.mode, seed=a.seed, limit=a.limit, overwrite=a.overwrite)
    s = summarise(df)
    print(f"\n[{a.mode}] {s['months']} months")
    print(f"  final {s['final']:.1f}x  CAGR {s['cagr']:.1f}%  vol {s['vol']:.1f}%  "
          f"Sharpe {s['sharpe']:.2f}  maxDD {s['maxdd']:.1f}%")
    print(f"  deviated from defaults: {s['deviated']}/{s['months']}   "
          f"rejected: {s['rejected']}")
    if df.attrs.get("calls"):
        print(f"  API calls {df.attrs['calls']}  spend ${df.attrs['spend']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
