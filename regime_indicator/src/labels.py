"""The ex-post targets. Without one, Brier, log loss and kappa are undefined.

Two mechanical proxies, both computed from price alone so neither favours an
arm. This is the main advantage equities have over FX: the FX study had to
argue for its ground truth, and here it is close to mechanical.

PRIMARY -- forward 20-day realized-volatility terciles.
    lowest tercile  -> bull_quiet
    middle          -> bull_volatile
    highest         -> bear_stress
Forward-looking by construction, which is the point: it is the thing the
detectors are trying to anticipate. Realized volatility over the *next* month
is unknowable today, so no arm can read it off its own inputs.

SECONDARY -- contemporaneous drawdown state, reported as robustness only.
    < 5% off the 252-day high -> bull_quiet
    5-10%                     -> bull_volatile
    > 10%                     -> bear_stress

A WARNING ABOUT THE SECONDARY. Drawdown from the 252-day high is *itself one of
the four HMM features*. Scoring Arm B against a drawdown-defined target is
therefore close to circular -- the detector is being asked to reproduce one of
its own inputs, and it will look good doing it. It is kept because the plan asks
for two proxies and disagreement between them is informative, but the headline
number must come from the forward-vol target. Say which one produced it; if you
do not, that is the first question you will get.

Neither target may enter a feature, a fit window or a prompt. scoring.py is
their only consumer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import STATES
from .features import TRADING_DAYS, drawdown_from_high, log_returns


def forward_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized vol over the NEXT `window` trading days, in percent.

    rolling(w).std() at t+w covers [t+1, t+w]; shifting back by w puts that
    value at t. The final `window` observations are necessarily undefined.
    """
    fwd = returns.rolling(window).std().shift(-window)
    return (fwd * np.sqrt(TRADING_DAYS) * 100.0).rename("fwd_vol")


def build_target(close: pd.Series, cfg: dict) -> pd.Series:
    """Primary target: forward realized-vol terciles, in {bull_quiet, ...}."""
    t = cfg["target"]
    w = int(t["forward_window"])
    min_obs = int(cfg["features"]["min_standardize_obs"])
    vol = forward_realized_vol(log_returns(close), w)

    if t.get("tercile_basis", "expanding") == "expanding":
        # Cuts derived only from data available at label time, so the class
        # boundaries themselves carry no full-sample information.
        lo = vol.expanding(min_periods=min_obs).quantile(1 / 3)
        hi = vol.expanding(min_periods=min_obs).quantile(2 / 3)
    else:
        lo = pd.Series(vol.quantile(1 / 3), index=vol.index)
        hi = pd.Series(vol.quantile(2 / 3), index=vol.index)

    out = pd.Series(pd.NA, index=close.index, dtype="object")
    ok = vol.notna() & lo.notna() & hi.notna()
    out[ok & (vol <= lo)] = "bull_quiet"
    out[ok & (vol > lo) & (vol <= hi)] = "bull_volatile"
    out[ok & (vol > hi)] = "bear_stress"
    return out.rename("target")


def build_secondary_target(close: pd.Series, cfg: dict) -> pd.Series:
    """Secondary: contemporaneous drawdown state. Robustness only.

    Near-circular for Arm B, which has drawdown as an input feature. See the
    module docstring.
    """
    t = cfg["target"]["secondary"]["thresholds"]
    dd = drawdown_from_high(close, int(cfg["features"]["drawdown_window"]))
    out = pd.Series("bull_quiet", index=close.index, dtype="object")
    out[dd > float(t["bull_quiet"])] = "bull_volatile"
    out[dd > float(t["bull_volatile"])] = "bear_stress"
    return out.rename("target_drawdown")


# ------------------------------------------------------------------- baselines
def persistence_forecast(target: pd.Series, cfg: dict) -> pd.DataFrame:
    """The most recent target value actually KNOWABLE at t, carried forward.

    The plan asks for "yesterday's label", which is right for a contemporaneous
    label and wrong here. This target is forward-looking over `forward_window`
    days, so target[t-1] covers [t, t+w-1] and shares w-1 of its w days with
    target[t]. A forecaster standing at t cannot know it -- it is not determined
    until t+w-1. Scored that way, persistence reached 95.2% accuracy and kappa
    0.93, which is not a strong baseline but a leak: it is reading the answer.

    The feasible version lags by the full forward window. target[t-w] is the
    most recent value determined by time t, so this is a forecast someone could
    actually have made.
    """
    t = target.dropna().astype(str)
    w = int(cfg["target"]["forward_window"])
    prev = t.shift(w)
    min_obs = int(cfg["features"]["min_standardize_obs"])
    # Expanding hit rate of the persistence rule, so its confidence is earned
    # from history rather than assumed.
    hit = (prev == t).astype(float).expanding(min_periods=min_obs).mean()

    out = pd.DataFrame(index=t.index, columns=[f"p_{s}" for s in STATES],
                       dtype=float)
    for s in STATES:
        # the remaining mass is split evenly across the other two states
        out[f"p_{s}"] = np.where(prev == s, hit, (1 - hit) / (len(STATES) - 1))
    out = out.div(out.sum(axis=1), axis=0)
    out["label"] = prev
    out["confidence"] = out[[f"p_{s}" for s in STATES]].max(axis=1)
    return out.dropna()


def climatology_forecast(target: pd.Series, cfg: dict) -> pd.DataFrame:
    """Expanding class frequencies. The floor every arm must clear.

    Accuracy has an obvious majority-class floor; Brier and log loss have this
    one, and it is far less obvious. On the FX study it was the baseline neither
    detector beat, which was the single most important finding there.
    """
    t = target.dropna().astype(str)
    min_obs = int(cfg["features"]["min_standardize_obs"])
    freq = pd.DataFrame({s: (t == s).astype(float) for s in STATES}, index=t.index)
    base = freq.expanding(min_periods=min_obs).mean().dropna(how="all")
    base = base.div(base.sum(axis=1), axis=0)

    out = pd.DataFrame(index=base.index)
    for s in STATES:
        out[f"p_{s}"] = base[s]
    out["label"] = base.idxmax(axis=1)
    out["confidence"] = base.max(axis=1)
    return out


# ------------------------------------------------------------------ diagnostics
def report(close: pd.Series, cfg: dict) -> dict:
    """Build-step-5a checkpoint on the targets."""
    res = {}
    tgt = build_target(close, cfg)
    sec = build_secondary_target(close, cfg)
    t = tgt.dropna()

    print("=" * 88)
    print("BUILD STEP 5a - ground truth")
    print("=" * 88)
    print(f"\nprimary   forward {cfg['target']['forward_window']}-day realized-vol "
          f"terciles ({cfg['target']['tercile_basis']} cuts)")
    print(f"          defined {t.index[0].date()} -> {t.index[-1].date()}  n={len(t)}")

    print("\n-- class balance (a degenerate target is trivially gamed) --")
    vc = t.value_counts(normalize=True)
    for s in STATES:
        print(f"   {s:14s} {vc.get(s, 0) * 100:5.1f}%")
    res["balance"] = {s: float(vc.get(s, 0)) for s in STATES}
    res["balanced"] = bool(vc.min() > 0.15)

    print("\n-- what any arm must beat --")
    print(f"   always-{vc.index[0]:14s} accuracy {vc.iloc[0] * 100:5.1f}%")
    clim = float(1 - (vc ** 2).sum())
    print(f"   climatology Brier (base rates every day) = {clim:.4f}")
    res["climatology_brier"] = clim

    print("\n-- episode structure of the target --")
    grp = (t != t.shift()).cumsum()
    dur = t.groupby(grp).size()
    w = int(cfg["target"]["forward_window"])
    print(f"   episodes: {len(dur)}   median duration: {dur.median():.0f} days")
    print(f"   consecutive values share {w - 1} of {w} forward days, so daily")
    print(f"   observations are ~{w}x overcounted. Every significance test")
    print(f"   blocks at >= {w} days; n={len(t)} is not n independent.")
    res["min_block_length"] = w

    print("\n-- SANITY: share of days labelled bear_stress --")
    gate = {}
    for ep in cfg["ground_truth_episodes"]:
        seg = t.loc[ep["start"]:ep["end"]]
        if not len(seg):
            continue
        gate[ep["name"]] = float((seg == "bear_stress").mean())
        print(f"   {ep['name']:14s} {gate[ep['name']] * 100:5.1f}%")
    for name, a, b in (("calm 2017", "2017-01-01", "2017-12-31"),
                       ("calm 2021", "2021-01-01", "2021-12-31")):
        seg = t.loc[a:b]
        if len(seg):
            print(f"   {name:14s} {(seg == 'bear_stress').mean() * 100:5.1f}%")
    res["gate"] = gate
    ok = gate.get("gfc", 0) > .5 and gate.get("covid", 0) > .5
    res["gate_pass"] = ok
    print(f"\n   GATE: {'PASS' if ok else 'FAIL'} (GFC and COVID above 50%)")

    print("\n-- primary vs secondary (drawdown state) --")
    idx = t.index.intersection(sec.dropna().index)
    agree = float((t.loc[idx] == sec.loc[idx]).mean())
    res["primary_secondary_agreement"] = agree
    print(f"   the two targets agree on {agree * 100:.1f}% of days")
    print(f"   secondary balance: "
          + "  ".join(f"{s} {(sec.loc[idx] == s).mean() * 100:.0f}%" for s in STATES))
    print("\n   NOTE: the drawdown target is near-CIRCULAR for Arm B, which has")
    print("   drawdown_from_252d_high as an input feature. Headline numbers come")
    print("   from the forward-vol target; the drawdown target is robustness only.")
    return res


if __name__ == "__main__":
    from . import config, ingest

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    report(blocks["index"]["close"], cfg)
