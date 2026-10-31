"""Feature blocks for a single equity index.

All features expanding-window standardized. Full-sample z-scoring is the most
common silent look-ahead bug in this study design -- see tests/test_lookahead.py.

Unit conventions, fixed here so nothing downstream has to guess:
  - returns          : daily log returns, decimal
  - realized vol     : annualized PERCENT (sqrt(252) * std * 100), so it is
                       directly comparable with VIX
  - drawdown         : POSITIVE magnitude below the running peak, 0.15 = 15% off
  - variance risk premium : VIX minus 20-day realized vol, in vol points

SENTIMENT IS ABSENT FROM THIS MODULE BY DESIGN. News and survey features feed
the agents only (plan section 4.4); letting them into the HMM vector would make
C - B unattributable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------------------------------------- standardization
def expanding_standardize(s: pd.Series, min_obs: int = 250) -> pd.Series:
    """Z-score using only data up to and including each point.

    The mean and std at t come from observations 0..t, so the value at t never
    depends on anything after t. Full-sample z-scoring would leak the whole
    distribution backwards -- including the crises the detector must discover.
    """
    mean = s.expanding(min_periods=min_obs).mean()
    std = s.expanding(min_periods=min_obs).std()
    return ((s - mean) / std.replace(0.0, np.nan)).rename(s.name)


def standardize_frame(df: pd.DataFrame, min_obs: int = 250) -> pd.DataFrame:
    return pd.DataFrame({c: expanding_standardize(df[c], min_obs)
                         for c in df.columns})


# ------------------------------------------------------------------- primitives
def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().rename("log_return")


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility in percent, comparable with VIX."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100.0


def trend_efficiency(returns: pd.Series, window: int) -> pd.Series:
    """|cumulative return| / sum(|daily returns|) over `window` days.

    1.0 is a straight line, 0.0 is pure chop. It separates a quiet market that
    is going somewhere from one that is merely quiet.
    """
    num = returns.rolling(window).sum().abs()
    den = returns.abs().rolling(window).sum()
    return (num / den.replace(0.0, np.nan)).rename("trend_efficiency")


def drawdown_from_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Positive magnitude below the running `window`-day peak. 0 at a new high.

    Signed positive so "higher is worse" reads naturally and matches the state
    mapping rule -- the highest mean drawdown becomes bear_stress. The plan's
    thresholds (5% and 10% off the high) are magnitudes, so this is the
    orientation they assume.

    THE most informative single equity state variable, and the one the
    build-step-2 checkpoint is built around.
    """
    peak = close.rolling(window, min_periods=1).max()
    return ((peak - close) / peak).rename("drawdown_from_252d_high")


def variance_risk_premium(vix: pd.Series, rv20: pd.Series) -> pd.Series:
    """Implied minus realized, in vol points.

    Compresses when the market is complacent and blows out in stress. It is
    also carrying the job the VIX term-structure ratio would have done -- that
    feature was dropped because no free source covers the scoring window.
    """
    return (vix - rv20).rename("variance_risk_premium")


# ----------------------------------------------------------------- agent blocks
def volatility_block(close: pd.Series, vol: pd.DataFrame,
                     cfg: dict) -> pd.DataFrame:
    """Realized and implied volatility, and the gap between them."""
    r = log_returns(close)
    out = {}
    for w in cfg["features"]["realized_windows"]:
        out[f"realized_vol_{w}d"] = realized_vol(r, w)
    out["vol_of_vol"] = realized_vol(r, 5).rolling(20).std()
    out["vix"] = vol["VIX"]
    out["vix_chg5"] = vol["VIX"].diff(5)
    out["variance_risk_premium"] = variance_risk_premium(
        vol["VIX"], realized_vol(r, 20))
    for c in ("VVIX", "SKEW"):
        if c in vol.columns:
            out[c.lower()] = vol[c]
    return pd.DataFrame(out)


def trend_block(close: pd.Series, cfg: dict) -> pd.DataFrame:
    """Direction, persistence and distance from the peak."""
    r = log_returns(close)
    out = {"log_return": r,
           "drawdown_from_252d_high": drawdown_from_high(
               close, int(cfg["features"]["drawdown_window"])),
           "trend_efficiency_20d": trend_efficiency(r, 20)}
    for w in (5, 20, 60):
        out[f"momentum_{w}d"] = r.rolling(w).sum()
    return pd.DataFrame(out)


def market_sentiment_block(blocks: dict, cfg: dict) -> pd.DataFrame:
    """Positioning and breadth. Agent input only, never the HMM."""
    br = blocks["breadth"]["breadth_ratio"]
    out = {"put_call_5d": blocks["put_call"]["put_call"].rolling(5).mean(),
           "breadth_ratio": br,
           "breadth_chg20": br.pct_change(20)}
    return pd.DataFrame(out)


def credit_block(blocks: dict, cfg: dict) -> pd.DataFrame:
    """Credit and financial conditions. Agent input only, never the HMM."""
    c = blocks["conditions"]
    return pd.DataFrame({
        "credit_spread": c["credit_spread"],
        "credit_chg20": c["credit_spread"].diff(20),
        "term_spread": c["term_spread"],
        "nfci": c["nfci"],
    })


# ------------------------------------------------------------------ HMM vector
def build_hmm_features(blocks: dict, cfg: dict) -> pd.DataFrame:
    """The detector input, in cfg['features']['hmm_features'] order.

    Four features, not the plan's five: the VIX term-structure ratio is dropped
    because no free source spans the scoring window. Kept deliberately small --
    3 states x 4 features with full covariance is already ~40 parameters, and
    bear_stress is a small share of the sample.

    Every column is expanding-standardized, so "deep drawdown" means deep
    relative to everything seen so far, not relative to a distribution that
    includes the future.
    """
    fcfg = cfg["features"]
    close = blocks["index"]["close"]
    r = log_returns(close)
    rv20 = realized_vol(r, 20)

    raw = pd.DataFrame({
        "log_return": r,
        "realized_vol_20d": rv20,
        "drawdown_from_252d_high": drawdown_from_high(
            close, int(fcfg["drawdown_window"])),
        "variance_risk_premium": variance_risk_premium(
            blocks["volatility"]["VIX"], rv20),
    })
    wanted = fcfg["hmm_features"]
    assert list(raw.columns) == wanted, \
        f"HMM feature order drift: {list(raw.columns)} != {wanted}"
    return standardize_frame(raw, int(fcfg["min_standardize_obs"]))


def build_features(blocks: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    """Run every block. `blocks` is the output of ingest.build_panel()."""
    close = blocks["index"]["close"]
    out = {
        "trend": trend_block(close, cfg),
        "volatility": volatility_block(close, blocks["volatility"], cfg),
        "market_sentiment": market_sentiment_block(blocks, cfg),
        "credit": credit_block(blocks, cfg),
    }
    out["hmm"] = build_hmm_features(blocks, cfg)
    return out


# -------------------------------------------------------------------- checkpoint
def report(blocks: dict, feats: dict, cfg: dict, save_plot: bool = True) -> dict:
    """Build-step-2 checkpoint.

    THE GATE: 2008 and 2020 must be unmistakable in the drawdown series. If a
    crash is not obvious there, the most informative equity state variable is
    broken and everything downstream is measuring a bug.
    """
    from . import config as _cfg

    res = {}
    dd = feats["trend"]["drawdown_from_252d_high"]
    close = blocks["index"]["close"]

    print("=" * 88)
    print("BUILD STEP 2 CHECKPOINT - features")
    print("=" * 88)
    print(f"\nindex {close.index[0].date()} -> {close.index[-1].date()}  "
          f"n={len(close)}")

    print("\n-- THE GATE: peak drawdown inside each hand-labelled episode --")
    ok = True
    for ep in cfg["ground_truth_episodes"]:
        seg = dd.loc[ep["start"]:ep["end"]]
        if not len(seg):
            continue
        worst = seg.max()
        when = seg.idxmax()
        big = ep["name"] in ("dotcom", "gfc", "covid", "bear_2022")
        flag = ""
        if big and worst < 0.15:
            flag = "   <-- EXPECTED DEEPER"
            ok = False
        print(f"   {ep['name']:14s} {worst * 100:6.1f}%  on {when.date()}{flag}")
    res["gate_pass"] = ok
    print(f"\n   GATE: {'PASS' if ok else 'FAIL'} "
          f"(the four major bears must each exceed 15% off the high)")

    print("\n-- drawdown distribution over the whole sample --")
    for q in (0.5, 0.75, 0.9, 0.99):
        print(f"   p{int(q * 100):02d}  {dd.quantile(q) * 100:5.1f}%")
    print(f"   max  {dd.max() * 100:5.1f}%  on {dd.idxmax().date()}")
    print(f"   days more than 10% off the high: {(dd > .10).mean() * 100:.1f}%")

    print("\n-- variance risk premium (implied minus realized, vol points) --")
    vrp = feats["volatility"]["variance_risk_premium"].dropna()
    print(f"   mean {vrp.mean():+.2f}   median {vrp.median():+.2f}   "
          f"negative on {(vrp < 0).mean() * 100:.1f}% of days")
    print("   (it should be POSITIVE most of the time -- selling variance earns "
          "a premium;\n    it goes negative when realized overtakes implied, "
          "which is stress)")
    for name, a, b in (("2008 GFC", "2008-09-01", "2008-12-31"),
                       ("2020 COVID", "2020-02-19", "2020-03-23"),
                       ("calm 2017", "2017-01-01", "2017-12-31")):
        seg = vrp.loc[a:b]
        if len(seg):
            print(f"   {name:12s} mean {seg.mean():+6.2f}")

    print("\n-- HMM feature vector --")
    hmm = feats["hmm"].dropna()
    res["hmm_rows"] = len(hmm)
    print(f"   usable rows: {len(hmm)}  {hmm.index[0].date()} -> "
          f"{hmm.index[-1].date()}")
    print(hmm.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    print("\n   correlation between HMM features "
          "(high values waste parameters):")
    corr = hmm.corr()
    print(corr.round(3).to_string())
    off = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(off, 0.0)
    res["max_feature_corr"] = float(off.max())
    print(f"   worst |rho| = {off.max():.3f}")

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax[0].semilogy(close.index, close, lw=.9, color="#0F141A")
        ax[0].set_title("S&P 500 (log scale)")
        ax[1].fill_between(dd.index, 0, -dd * 100, lw=0, color="#C05A16",
                           alpha=.85)
        ax[1].set_title("Drawdown from the 252-day high (%)")
        for ep in cfg["ground_truth_episodes"]:
            for a_ in ax:
                a_.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                           color="k", alpha=.07, lw=0)
        fig.tight_layout()
        path = _cfg.paths()["results"] / "step2_drawdown.png"
        fig.savefig(path, dpi=120, facecolor="white")
        plt.close(fig)
        print(f"\nplot: {path}")

    return res


if __name__ == "__main__":
    from . import config, ingest

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    report(blocks, build_features(blocks, cfg), cfg)
