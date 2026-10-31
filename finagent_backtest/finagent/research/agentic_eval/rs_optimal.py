"""Regime-switching dynamic optimal portfolio weights (Markose & Yang, EC907 Note 4).

Builds the classical benchmark the agent should be compared against, following
the note's Section 3.2 recipe exactly:

  Step 1  in-sample OLS of each stock's excess return on the market excess
          return -> alpha_i, beta_i, idiosyncratic variance
  Step 2  in-sample moments for the market excess return
  Step 3  Hamilton RS parameters {mu_m1, mu_m2, sigma_m1, sigma_m2, P, Q}
  Step 4  at each month, from the realised regime, form the *forward-looking*
          conditional market moments, Eq. (15) and (16):
              e^m(1)      = P mu1 + (1-P) mu2
              e^m(2)      = (1-Q) mu1 + Q mu2
              Sigma^m(1)  = P s1^2 + (1-P) s2^2 + P(1-P)(mu1-mu2)^2
              Sigma^m(2)  = (1-Q) s1^2 + Q s2^2 + Q(1-Q)(mu1-mu2)^2
          then, Eq. (17), securities are regime-dependent only through beta:
              mu_i(s)     = alpha_i + beta_i e^m(s)
              Sigma(s)    = beta beta' Sigma^m(s) + diag(idio variance)
  Step 5  mean-variance quadratic programme each month, long-only, sum <= 1
  Step 6  cumulate wealth and compare against the non-RS (static) portfolio

Three weight paths are produced for comparison with the agent:
  * ``rs``        regime-switching dynamic optimal (the note's method)
  * ``static``    traditional non-RS mean-variance on unconditional moments
  * ``equal``     equal-weight, the naive benchmark

Run:
    python -m finagent.research.agentic_eval.rs_optimal
    # -> data/agentic_eval/rs_optimal.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
MD = REPO / "data" / "agentic_eval" / "market_data"
OUT = REPO / "data" / "agentic_eval" / "rs_optimal.parquet"

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]
IN_SAMPLE_END = "2005-04-30"     # Hamilton params fit on market data up to here
MIN_OBS = 36                      # min months of stock history before the CAPM is trusted
GAMMA = 6.0                       # risk aversion in the M-V objective


def mv_optimise(mu: np.ndarray, cov: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Long-only mean-variance: max w'mu - (gamma/2) w'Sigma w, sum(w) <= 1."""
    n = len(mu)
    cov = cov + np.eye(n) * 1e-10

    def neg_util(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)

    def grad(w):
        return -(mu - gamma * cov @ w)

    res = minimize(neg_util, np.full(n, 1.0 / n), jac=grad, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}],
                   options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() > 1.0:
        w = w / w.sum()
    return w


def build() -> pd.DataFrame:
    stk = pd.read_parquet(MD / "stocks_monthly.parquet")
    spx = pd.read_parquet(MD / "spx_monthly.parquet")
    rf = pd.read_parquet(MD / "rf_monthly.parquet")
    reg = pd.read_parquet(MD / "regime_hamilton.parquet")

    mkt = spx["ret"].astype(float)
    rfm = rf["rf"].astype(float).reindex(mkt.index).fillna(0.0)
    mkt_ex = (mkt - rfm).dropna()

    def capm(upto):
        """Point-in-time OLS of each stock's excess return on the market's, to `upto`."""
        win = stk.index[stk.index <= upto]
        a_v, b_v, d_v = [], [], []
        for t in TICKERS:
            y = (stk.loc[win, t].astype(float) - rfm.reindex(win).fillna(0.0)).dropna()
            x = mkt_ex.reindex(y.index)
            ok = x.notna() & y.notna()
            x, y = x[ok].values, y[ok].values
            b, a = np.polyfit(x, y, 1)
            resid = y - (a + b * x)
            a_v.append(float(a)); b_v.append(float(b)); d_v.append(float(resid.var(ddof=1)))
        return np.array(a_v), np.array(b_v), np.diag(d_v)

    # ---- Step 3: Hamilton RS parameters, estimated in-sample on the market
    from finagent.research.agentic_eval import regime_hamilton as RH
    ins_mkt = mkt_ex[mkt_ex.index <= IN_SAMPLE_END]
    par = RH.fit(ins_mkt * 100.0, bull_var=(14.0 ** 2) / 12.0)
    mu1, mu2 = par["mu"][0] / 100.0, par["mu"][1] / 100.0
    s1, s2 = np.sqrt(par["var"][0]) / 100.0, np.sqrt(par["var"][1]) / 100.0
    P, Q = float(par["P"]), float(par["Q"])

    # ---- Step 4: forward-looking conditional market moments, Eq. (15)/(16)
    e_m = {1: P * mu1 + (1 - P) * mu2,
           2: (1 - Q) * mu1 + Q * mu2}
    S_m = {1: P * s1 ** 2 + (1 - P) * s2 ** 2 + P * (1 - P) * (mu1 - mu2) ** 2,
           2: (1 - Q) * s1 ** 2 + Q * s2 ** 2 + Q * (1 - Q) * (mu1 - mu2) ** 2}

    # ---- Step 5/6: walk out of sample, re-estimating point-in-time each month
    all_m = sorted(d for d in stk.index if d in reg.index)
    months = [d for d in all_m if all_m.index(d) >= MIN_OBS]

    rows = []
    for d in months:
        prev = all_m[all_m.index(d) - 1]            # information available at decision
        a_vec, b_vec, d_idio = capm(prev)
        s = 1 if str(reg.loc[d, "regime"]) == "bull" else 2

        # Eq. (17): securities regime-dependent only through beta
        mu_i = a_vec + b_vec * e_m[s]
        cov = np.outer(b_vec, b_vec) * S_m[s] + d_idio
        w_rs = mv_optimise(mu_i, cov)

        # traditional non-RS: unconditional moments on the same information set
        win = stk.index[stk.index <= prev]
        w_static = mv_optimise(
            stk.loc[win, TICKERS].astype(float).mean().values,
            stk.loc[win, TICKERS].astype(float).cov().values)

        r = stk.loc[d, TICKERS].astype(float).values
        rf_d = float(rfm.get(d, 0.0))

        rows.append({
            "date": d,
            "regime": str(reg.loc[d, "regime"]),
            "ret_rs": float(w_rs @ r + (1 - w_rs.sum()) * rf_d),
            "ret_static": float(w_static @ r + (1 - w_static.sum()) * rf_d),
            "ret_equal": float(r.mean()),
            "gross_rs": float(w_rs.sum()),
            "gross_static": float(w_static.sum()),
            **{f"wrs_{t}": w_rs[i] for i, t in enumerate(TICKERS)},
            **{f"wst_{t}": w_static[i] for i, t in enumerate(TICKERS)},
        })

    df = pd.DataFrame(rows).set_index("date")
    df.attrs["params"] = {"mu1": mu1, "mu2": mu2, "s1": s1, "s2": s2, "P": P, "Q": Q,
                          "e_m": e_m, "S_m": S_m, "gamma": GAMMA}
    df.to_parquet(OUT)

    print(f"RS-optimal built -> {OUT}  ({len(df)} months)")
    print(f"  Hamilton (in-sample): mu1={mu1*100:+.2f}%/mo  mu2={mu2*100:+.2f}%/mo  "
          f"s1={s1*100:.2f}%  s2={s2*100:.2f}%  P={P:.3f} Q={Q:.3f}")
    print(f"  Eq15 conditional mean : bull {e_m[1]*100:+.3f}%/mo | bear {e_m[2]*100:+.3f}%/mo")
    print(f"  Eq16 conditional sd   : bull {np.sqrt(S_m[1])*100:.2f}%  | bear {np.sqrt(S_m[2])*100:.2f}%")
    for lab, mask in (("bull", df.regime == "bull"), ("bear", df.regime == "bear")):
        print(f"  RS gross exposure {lab}: {df.loc[mask,'gross_rs'].mean():.2f}  "
              f"(n={int(mask.sum())})")
    cum = lambda c: float((1 + df[c]).prod())
    print(f"  cumulative x: RS {cum('ret_rs'):.1f}  static {cum('ret_static'):.1f}  "
          f"equal {cum('ret_equal'):.1f}")
    return df


if __name__ == "__main__":
    build()
