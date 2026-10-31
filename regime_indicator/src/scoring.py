"""Metrics. Accepts any arm's output contract so all arms share one path.

TWO REPORTING SCOPES, always:
  - full   : every scored day
  - gated  : only days the agent panel was allowed to act on
Arm C equals Arm B by construction off-gate, so a full-sample C - B is diluted
toward zero no matter what the agents do. The gated number answers the research
question; the full number is the honest headline. Report both.

TWO STATE COUNTS, always:
  - three-class : the full contract
  - two-class   : bull vs bear. Arm A has no `bull_volatile` state, so its
                  three-class agreement is structurally capped. The collapse is
                  the only comparison honest on state count.

TWO SKILL BASELINES, always:
  - persistence : yesterday's label. Regime labels are heavily autocorrelated,
                  so this is the bar that matters.
  - climatology : the expanding base rates. Less obvious and easier to lose to
                  than anyone expects -- on the FX study neither detector beat it.

THE AUTOCORRELATION CONSTRAINT. The target at t and t+1 share 19 of their 20
forward days, so daily observations are roughly 20x overcounted. Every
significance test blocks at >= the forward window; ignoring that shrinks every
standard error by about sqrt(20).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import STATES, TWO_STATE

SCOPES = ("full", "gated")
PROB_COLS = [f"p_{s}" for s in STATES]
EPS = 1e-12


# ------------------------------------------------------------------- alignment
def _align(probs: pd.DataFrame, target: pd.Series):
    idx = probs.dropna().index.intersection(target.dropna().index)
    P = np.clip(probs.loc[idx, PROB_COLS].to_numpy(dtype=float), EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = np.array([STATES.index(v) for v in target.loc[idx].astype(str)])
    return idx, P, y


def _onehot(P, y):
    o = np.zeros_like(P)
    o[np.arange(len(y)), y] = 1.0
    return o


# --------------------------------------------------------------- probabilistic
def brier_score(probs: pd.DataFrame, target: pd.Series) -> float:
    """Multiclass Brier: mean squared error against the one-hot outcome."""
    _, P, y = _align(probs, target)
    return float(((P - _onehot(P, y)) ** 2).sum(axis=1).mean())


def log_loss(probs: pd.DataFrame, target: pd.Series) -> float:
    _, P, y = _align(probs, target)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def pointwise_loss(probs: pd.DataFrame, target: pd.Series,
                   kind: str = "brier") -> pd.Series:
    """Per-day loss, the input to Diebold-Mariano."""
    idx, P, y = _align(probs, target)
    v = (((P - _onehot(P, y)) ** 2).sum(axis=1) if kind == "brier"
         else -np.log(P[np.arange(len(y)), y]))
    return pd.Series(v, index=idx, name=f"loss_{kind}")


def brier_skill_score(probs: pd.DataFrame, target: pd.Series,
                      baseline: pd.DataFrame) -> float:
    """BSS = 1 - BS / BS_baseline, on the days both are defined.

    Positive means better than the baseline. Above ~0.35 against persistence,
    treat it as a leakage alarm rather than a triumph and re-run the look-ahead
    checklist -- daily regime nowcasting does not legitimately score that well.
    """
    idx = (probs.dropna().index
           .intersection(baseline.dropna().index)
           .intersection(target.dropna().index))
    if len(idx) < 100:
        return float("nan")
    bs = brier_score(probs.loc[idx], target.loc[idx])
    bs0 = brier_score(baseline.loc[idx], target.loc[idx])
    return float(1 - bs / bs0) if bs0 > EPS else float("nan")


def accuracy(labels: pd.Series, target: pd.Series) -> float:
    idx = labels.dropna().index.intersection(target.dropna().index)
    return float((labels.loc[idx].astype(str) == target.loc[idx].astype(str)).mean())


# --------------------------------------------------------------- decomposition
def reliability(probs: pd.DataFrame, target: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram table, per class. Gap = |confidence - frequency|."""
    _, P, y = _align(probs, target)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for k, state in enumerate(STATES):
        p, hit = P[:, k], (y == k).astype(float)
        b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for j in range(bins):
            m = b == j
            if m.sum() < 20:
                continue
            rows.append({"state": state, "bin": f"{edges[j]:.1f}-{edges[j+1]:.1f}",
                         "n": int(m.sum()), "mean_p": float(p[m].mean()),
                         "observed": float(hit[m].mean()),
                         "gap": float(abs(p[m].mean() - hit[m].mean()))})
    return pd.DataFrame(rows)


def calibration_error(probs: pd.DataFrame, target: pd.Series, bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean gap across bins and classes."""
    r = reliability(probs, target, bins)
    return float(np.average(r["gap"], weights=r["n"])) if len(r) else float("nan")


def murphy_decomposition(probs: pd.DataFrame, target: pd.Series,
                         bins: int = 10) -> dict:
    """Brier = Reliability - Resolution + Uncertainty, averaged over classes.

    Reliability: are your 70% calls right 70% of the time? Lower is better.
    Resolution:  do the probabilities actually vary with the outcome? Higher is
                 better -- a perfectly calibrated forecast that never moves off
                 the base rate has zero resolution and is useless.
    Uncertainty: the base-rate variance. A property of the target, not the
                 forecast, and identical across arms.
    """
    _, P, y = _align(probs, target)
    edges = np.linspace(0, 1, bins + 1)
    rel = res = unc = 0.0
    for k in range(len(STATES)):
        p, o = P[:, k], (y == k).astype(float)
        obar, n = o.mean(), len(o)
        b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for j in range(bins):
            m = b == j
            if not m.sum():
                continue
            rel += m.sum() / n * (p[m].mean() - o[m].mean()) ** 2
            res += m.sum() / n * (o[m].mean() - obar) ** 2
        unc += obar * (1 - obar)
    return {"reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc)}


def calibrate_isotonic(probs: pd.DataFrame, target: pd.Series,
                       cfg: dict) -> pd.DataFrame:
    """Expanding-window isotonic calibration, one-vs-rest per class.

    Fit on every scored day BEFORE year y, apply to year y, roll. The first
    scoring year passes through uncalibrated -- calibrating it on its own
    outcomes would be look-ahead, which is the trap a naive in-sample isotonic
    fit falls straight into.
    """
    from sklearn.isotonic import IsotonicRegression

    out = probs.copy()
    idx = probs.dropna().index.intersection(target.dropna().index)
    tgt = target.loc[idx].astype(str)
    for y in sorted({d.year for d in idx})[1:]:
        past, cur = idx[idx.year < y], idx[idx.year == y]
        if len(past) < 250 or not len(cur):
            continue
        cal = np.zeros((len(cur), len(STATES)))
        for k, state in enumerate(STATES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs.loc[past, PROB_COLS[k]].to_numpy(),
                    (tgt.loc[past] == state).astype(float).to_numpy())
            cal[:, k] = iso.predict(probs.loc[cur, PROB_COLS[k]].to_numpy())
        s = cal.sum(axis=1, keepdims=True)
        out.loc[cur, PROB_COLS] = np.where(s > 0, cal / np.where(s == 0, 1, s),
                                           1.0 / len(STATES))
    return out


# ------------------------------------------------------------------- agreement
def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement.

    Between two ARMS this is agreement, not accuracy, and must be labelled so.
    Against the target it is accuracy.
    """
    idx = a.dropna().index.intersection(b.dropna().index)
    a, b = a.loc[idx].astype(str), b.loc[idx].astype(str)
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return float("nan")
    obs = float((a == b).mean())
    exp = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((obs - exp) / (1 - exp)) if exp < 1 else float("nan")


def to_two_state(labels: pd.Series) -> pd.Series:
    """Collapse to bull vs bear -- the only comparison Arm A enters fairly."""
    return labels.astype(str).map(TWO_STATE).fillna("bull")


# ------------------------------------------------------------------- structure
def episode_stats(labels: pd.Series) -> dict:
    """Episode duration, turnover, switches per year.

    Compared BETWEEN ARMS, never against the target: the target flips whenever
    forward vol crosses a tercile boundary, so its episode count is an artefact
    of the threshold rather than something a detector should reproduce.
    """
    lab = labels.dropna().astype(str)
    if not len(lab):
        return {}
    switch = lab != lab.shift()
    dur = lab.groupby(switch.cumsum()).size()
    years = max((lab.index[-1] - lab.index[0]).days / 365.25, 1e-9)
    return {"n_episodes": int(len(dur)), "median_duration": float(dur.median()),
            "mean_duration": float(dur.mean()), "max_duration": int(dur.max()),
            "turnover": float(switch.mean()),
            "switches_per_year": float(switch.sum() / years)}


def false_alarms_per_year(labels: pd.Series, target: pd.Series) -> float:
    """bear_stress called while the target says otherwise, counted as episodes."""
    idx = labels.dropna().index.intersection(target.dropna().index)
    lab, tgt = labels.loc[idx].astype(str), target.loc[idx].astype(str)
    call = lab == "bear_stress"
    onset = call & ~call.shift(fill_value=False)
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    return float((onset & (tgt != "bear_stress")).sum() / years)


def transition_lead_time(labels: pd.Series, cfg: dict,
                         window_days: int = 60) -> pd.DataFrame:
    """Days between an arm's bear_stress call and each episode onset.

    Positive means early. Measured in a window around onset rather than inside
    it: mid-episode a persistent detector is flat and has nothing left to
    signal. Reported per-episode -- a mean with an implied standard error over
    seven observations would be theatre.
    """
    lab = labels.dropna().astype(str)
    call = lab == "bear_stress"
    onset = call & ~call.shift(fill_value=False)
    rows = []
    for ep in cfg["ground_truth_episodes"]:
        s = pd.Timestamp(ep["start"])
        hits = onset.loc[s - pd.Timedelta(days=window_days):
                         s + pd.Timedelta(days=window_days)]
        hits = hits[hits]
        rows.append({"episode": ep["name"], "onset": s.date(),
                     "detected": hits.index[0].date() if len(hits) else None,
                     "lead_days": int((s - hits.index[0]).days) if len(hits) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- significance
def _newey_west_var(d: np.ndarray, lags: int) -> float:
    """Long-run variance of the mean, Bartlett weights."""
    d = d - d.mean()
    n = len(d)
    v = float(d @ d) / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * float(d[l:] @ d[:-l]) / n
    return max(v, EPS)


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, block: int = 20) -> dict:
    """Paired forecast-loss test with HAC standard errors.

    `block` sets the Bartlett truncation and MUST be at least the target's
    forward window: overlapping windows make the loss differential strongly
    autocorrelated, and ignoring that shrinks the standard error by ~sqrt(block).

    Reports the effective sample beside the statistic, because n days is not n
    independent observations and the power here is low either way.
    """
    from scipy import stats

    idx = loss_a.dropna().index.intersection(loss_b.dropna().index)
    d = (loss_a.loc[idx] - loss_b.loc[idx]).to_numpy(dtype=float)
    n = len(d)
    if n < 2 * block:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    stat = float(d.mean() / np.sqrt(_newey_west_var(d, block) / n))
    return {"mean_loss_diff": float(d.mean()), "dm_stat": stat,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
            "n": n, "n_effective": int(n / block), "block": block}


def stationary_bootstrap_diff(a: pd.Series, b: pd.Series, stat_fn,
                              n_boot: int = 2000, mean_block: int = 20,
                              seed: int = 0) -> dict:
    """Paired stationary bootstrap CI on a difference of statistics.

    Resamples both series TOGETHER with geometric block lengths, so the pairing
    and the serial dependence both survive resampling.
    """
    idx = a.dropna().index.intersection(b.dropna().index)
    x, y = a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)
    n = len(x)
    if n < 4 * mean_block:
        return {"diff": float("nan")}
    obs = stat_fn(x) - stat_fn(y)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pos, out = rng.integers(0, n), []
        while len(out) < n:
            out.append(pos)
            pos = rng.integers(0, n) if rng.random() < p else (pos + 1) % n
        sel = np.array(out[:n])
        diffs[i] = stat_fn(x[sel]) - stat_fn(y[sel])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
            "n_boot": n_boot, "mean_block": mean_block}


# ----------------------------------------------------------------------- scoring
def score_arm(out: pd.DataFrame, target: pd.Series, cfg: dict,
              baselines: dict[str, pd.DataFrame] | None = None,
              mask: pd.Series | None = None) -> dict:
    """Every metric for one arm at one scope."""
    df = out if mask is None else out.loc[mask.reindex(out.index, fill_value=False)]
    if not len(df):
        return {}
    probs, labels = df[PROB_COLS], df["label"]
    tgt = target.reindex(df.index)

    r = {"n": len(df),
         "accuracy": accuracy(labels, tgt),
         "kappa_vs_target": cohens_kappa(labels, tgt),
         "kappa_two_state": cohens_kappa(to_two_state(labels), to_two_state(tgt)),
         "brier": brier_score(probs, tgt),
         "log_loss": log_loss(probs, tgt),
         "calibration_error": calibration_error(probs, tgt)}
    r.update(murphy_decomposition(probs, tgt))
    for name, base in (baselines or {}).items():
        r[f"bss_vs_{name}"] = brier_skill_score(probs, tgt, base.reindex(df.index))
    r.update(episode_stats(labels))
    r["false_alarms_per_year"] = false_alarms_per_year(labels, tgt)
    return r
