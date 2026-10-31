"""Phase 2: Hamilton 2-state Markov regime-switching model on S&P 500 monthly returns.

Replaces the fixed-threshold SPY/VIX 2x2 classifier with the Markov-switching model of
Markose Note 4 (Hamilton 1989), Eqs. 4-14:

    r_mt = mu_m(s_t) + sigma_m(s_t) eps_mt ,   s_t in {bull, bear}

Estimated by the Hamilton filter + Kim smoother + EM (implemented directly here, faithful
to the lecture). The unconstrained MLE on monthly S&P collapses the bull state to an
unrealistic ~8%/yr volatility, which mislabels calm bull years (2021/2023/2024) as bear.
We therefore **constrain the bull-state volatility to BULL_VOL_ANNUAL (14%/yr)** so the
regimes align economically with the calendar; the bear state is left free and recovers a
negative-mean, transient crisis regime (Markose's structure).

Labels: BULL = low-volatility state, BEAR = high-volatility state.
Output (market_data/regime_hamilton.parquet): per-month *filtered* bull probability
(causal -> used by the backtest) + *smoothed* (display) + 0.5-rule label.

Read-only over market_data/; never touches data/dissertation/.

Run:
    python -m finagent.research.agentic_eval.regime_hamilton
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MD = REPO / "data" / "agentic_eval" / "market_data"

BULL_VOL_ANNUAL = 14.0   # %/yr — economic constraint on the bull-state volatility
_MONTHS = 12


def _gaussian(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def hamilton_filter(r, mu, var, P, Q):
    """Causal forward filter. Returns (filtered, predicted, loglik).

    State 0 = bull, state 1 = bear. P = bull persistence, Q = bear persistence.
    """
    n = len(r)
    T = np.array([[P, 1 - P], [1 - Q, Q]])      # T[i,j] = P(s_t=j | s_{t-1}=i)
    pi = np.array([1 - Q, 1 - P], float); pi /= pi.sum()  # ergodic start
    eta = np.clip(np.column_stack([_gaussian(r, mu[0], var[0]),
                                   _gaussian(r, mu[1], var[1])]), 1e-300, None)
    xp = np.zeros((n, 2)); xf = np.zeros((n, 2)); ll = 0.0
    for t in range(n):
        pred = pi if t == 0 else xf[t - 1] @ T
        xp[t] = pred
        num = pred * eta[t]; s = num.sum()
        xf[t] = num / s
        ll += np.log(s)
    return xf, xp, ll, eta, T


def _kim_smoother(xf, xp, T):
    n = len(xf)
    xs = np.zeros((n, 2)); xs[-1] = xf[-1]
    for t in range(n - 2, -1, -1):
        xs[t] = xf[t] * (T @ (xs[t + 1] / np.clip(xp[t + 1], 1e-300, None)))
        xs[t] /= xs[t].sum()
    return xs


def fit(r, bull_var, n_iter: int = 500, tol: float = 1e-8):
    """Constrained EM. Bull variance held at ``bull_var``; everything else free."""
    r = np.asarray(r, float)
    mu = np.array([1.0, -1.0]); var = np.array([bull_var, np.var(r) * 3.0])
    P, Q = 0.95, 0.85; prev = -np.inf
    xf = xp = T = None
    for _ in range(n_iter):
        xf, xp, ll, eta, T = hamilton_filter(r, mu, var, P, Q)
        xs = _kim_smoother(xf, xp, T)
        mu = (xs * r[:, None]).sum(0) / xs.sum(0)
        var = (xs * (r[:, None] - mu) ** 2).sum(0) / xs.sum(0)
        var[0] = bull_var                                   # <-- the constraint
        nij = np.zeros((2, 2))
        for t in range(1, len(r)):
            nij += (xf[t - 1][:, None] * T) * (xs[t] / np.clip(xp[t], 1e-300, None))[None, :]
        Tn = nij / nij.sum(1, keepdims=True)
        P, Q = float(Tn[0, 0]), float(Tn[1, 1])
        if abs(ll - prev) < tol:
            break
        prev = ll
    return {"mu": mu, "var": var, "P": P, "Q": Q}


def classify(returns_pct: pd.Series, params: dict | None = None,
             bull_vol_annual: float = BULL_VOL_ANNUAL) -> pd.DataFrame:
    """Fit (if needed) and return per-month bull probabilities + label.

    ``filtered`` is causal (uses data up to t only) -> the column the backtest reads.
    Pass pre-fit ``params`` (e.g. estimated on an in-sample window) to apply
    point-in-time without re-fitting.
    """
    r = np.asarray(returns_pct, float)
    bull_var = (bull_vol_annual / np.sqrt(_MONTHS)) ** 2
    if params is None:
        params = fit(r, bull_var)
    xf, xp, _, _, T = hamilton_filter(r, params["mu"], params["var"], params["P"], params["Q"])
    xs = _kim_smoother(xf, xp, T)
    out = pd.DataFrame({
        "p_bull_filtered": xf[:, 0],
        "p_bull_smoothed": xs[:, 0],
        "regime": np.where(xf[:, 0] > 0.5, "bull", "bear"),
    }, index=returns_pct.index)
    out.index.name = "date"
    return out, params


def main() -> int:
    ret = (pd.read_parquet(MD / "spx_monthly.parquet")["ret"].dropna() * 100.0)
    out, p = classify(ret)
    out.to_parquet(MD / "regime_hamilton.parquet")

    mu, var, P, Q = p["mu"], p["var"], p["P"], p["Q"]
    sd = np.sqrt(var)
    print("=" * 70)
    print(f"Hamilton 2-state model  (S&P 500 monthly 1990-2025; bull vol constrained to "
          f"{BULL_VOL_ANNUAL:.0f}%/yr)")
    print("=" * 70)
    print(f"{'':6s}{'mu %/mo':>9s}{'sig %/mo':>9s}{'mu %/yr':>9s}{'sig %/yr':>10s}{'persist':>9s}")
    print(f"{'BULL':6s}{mu[0]:9.2f}{sd[0]:9.2f}{mu[0]*12:9.2f}{sd[0]*np.sqrt(12):10.2f}{P:9.3f}")
    print(f"{'BEAR':6s}{mu[1]:9.2f}{sd[1]:9.2f}{mu[1]*12:9.2f}{sd[1]*np.sqrt(12):10.2f}{Q:9.3f}")
    lab = out["regime"]
    print(f"\nFiltered classification: {100*(lab=='bull').mean():.0f}% bull-months,  "
          f"expected durations bull {1/(1-P):.1f} mo / bear {1/(1-Q):.1f} mo")
    print("\nBear episodes (filtered, causal) since 1998:")
    g = (lab != lab.shift()).cumsum()
    for _, idx in lab.groupby(g).groups.items():
        if lab.loc[idx[0]] == "bear" and idx[-1].year >= 1998:
            print(f"  {idx[0].strftime('%Y-%m')} -> {idx[-1].strftime('%Y-%m')}  ({len(idx)} mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
