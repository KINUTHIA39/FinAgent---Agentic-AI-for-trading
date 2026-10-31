"""Arm A: Hamilton two-state Markov switching. The thing to beat.

Implemented to the Markose specification (EC907 Note 4, the PDF in this repo):
Eq. 4 is r_t = mu(s_t) + sigma(s_t) * eps_t -- a switching mean and switching
variance with NO autoregressive term. So MarkovRegression(trend="c",
switching_variance=True) is the faithful estimator. The master plan's
MarkovAutoregression(order=4) would add four lags the lecture never specifies.

Equities suit this model far better than FX did. Markose's FTSE regimes are
"high return / low volatility" and "low return / high volatility" -- a genuine
mean gap. In FX the means were near zero and the variance did all the work; on
an equity index the mean separation is real, so the estimated parameters should
show mu_bull > 0 > mu_bear alongside sigma_bear > sigma_bull.

THE FAIRNESS CONSTRAINT: filtered_marginal_probabilities only. Smoothed
probabilities condition on the whole sample including the future and would hand
the benchmark a look-ahead advantage, invalidating the comparison. A fitted
statsmodels result exposes BOTH attributes, one character apart, so this is
asserted in code rather than left to discipline.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .detector import STATES

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")

SMOOTHED_FORBIDDEN = "Arm A must use filtered_marginal_probabilities."
TRADING_DAYS = 252


# ----------------------------------------------------------------------- fitting
def _build(endog: pd.Series, cfg: dict) -> MarkovRegression:
    b = cfg["benchmark"]
    return MarkovRegression(endog, k_regimes=int(b["k_regimes"]),
                            trend=b["trend"],
                            switching_variance=bool(b["switching_variance"]))


def fit_hamilton(returns: pd.Series, cfg: dict):
    """Fit the two-state model on index log returns through the fit window."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build(returns, cfg).fit(disp=False)


def bear_regime(res) -> int:
    """Index of the high-variance regime.

    statsmodels regime ordering is arbitrary between fits, exactly as hmmlearn's
    state indices are, so the economic identity must be recovered from the
    estimated variances every time rather than assumed to be regime 1.
    """
    sig = [float(res.params[f"sigma2[{i}]"]) for i in range(2)]
    return int(np.argmax(sig))


# --------------------------------------------------------------------- filtering
def score_filtered(res_frozen, cfg: dict) -> np.ndarray:
    """Filtered P(regime) as an (nobs, 2) array. Asserts smoothed is never used.

    The guard is deliberately loud: silently reading the smoothed attribute
    would produce a benchmark that looks strong and is invalid, and nothing
    downstream would reveal it.
    """
    assert cfg["benchmark"]["probabilities"] == "filtered", SMOOTHED_FORBIDDEN
    filt = np.asarray(res_frozen.filtered_marginal_probabilities)
    if filt.shape[0] == 2 and filt.shape[1] != 2:
        filt = filt.T
    assert filt.ndim == 2 and filt.shape[1] == 2, f"unexpected shape {filt.shape}"
    return filt


def _to_three_class(p_bear: np.ndarray, cfg: dict) -> np.ndarray:
    """Project the two-state output onto the shared three-class contract.

    Hamilton has no `bull_volatile` state. Its low-volatility regime spans both
    bull states and the model is genuinely agnostic between them, so that mass
    is split evenly by default.

    The alternative -- putting it all on bull_quiet and zero on bull_volatile --
    would hand Arm A an unbounded log loss on every day the target says
    bull_volatile, which is an artefact of the output contract rather than a
    property of the model. `p_bear_raw` is carried through separately so scoring
    can also run the strictly comparable two-state contest.
    """
    mode = cfg["benchmark"].get("two_state_mapping", "split_low_vol")
    out = np.zeros((len(p_bear), 3))
    out[:, STATES.index("bear_stress")] = p_bear
    bull = 1.0 - p_bear
    if mode == "split_low_vol":
        out[:, STATES.index("bull_quiet")] = bull / 2.0
        out[:, STATES.index("bull_volatile")] = bull / 2.0
    elif mode == "quiet_only":
        out[:, STATES.index("bull_quiet")] = bull
    else:
        raise ValueError(f"unknown two_state_mapping: {mode}")
    return out


# ------------------------------------------------------------------ walk forward
def walk_forward(returns: pd.Series, cfg: dict,
                 score_end: str | None = None) -> pd.DataFrame:
    """Same annual expanding-window schedule as the HMM.

    Fit through 31 Dec of y-1, freeze the four parameters plus the two
    transition probabilities, then run the filter from the sample start to each
    day in year y. Forward-only, so a value in year y depends on data up to that
    day and nothing later.

    Any asymmetry in schedule between arms would make B - A partly a comparison
    of refit cadence rather than of models.
    """
    scfg = cfg["sample"]
    r = returns.dropna()
    start = pd.Timestamp(scfg["score_start"])
    end = min(pd.Timestamp(score_end or scfg["score_end"]), r.index.max())

    rows, fit_log = [], []
    for y in range(start.year, end.year + 1):
        fit_end = pd.Timestamp(f"{y - 1}-12-31")
        insample = r.loc[:fit_end]
        if len(insample) < 250:
            continue

        res_is = fit_hamilton(insample, cfg)
        bear = bear_regime(res_is)

        through_y = r.loc[:pd.Timestamp(f"{y}-12-31")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frozen = _build(through_y, cfg).filter(res_is.params)
        filt = score_filtered(frozen, cfg)

        mask = through_y.index.year == y
        idx, p_bear = through_y.index[mask], filt[mask, bear]
        named = _to_three_class(p_bear, cfg)

        ent = -(named * np.log(np.clip(named, 1e-12, 1))).sum(axis=1) / np.log(3)
        model_id = f"hamilton2_{y}"
        rows.append(pd.DataFrame({
            "label": [STATES[i] for i in named.argmax(axis=1)],
            "p_bull_quiet": named[:, 0], "p_bull_volatile": named[:, 1],
            "p_bear_stress": named[:, 2],
            "p_bear_raw": p_bear,
            "confidence": named.max(axis=1), "entropy_confidence": 1.0 - ent,
            "model_id": model_id,
            "fit_start": insample.index[0], "fit_end": fit_end,
        }, index=idx))

        p = res_is.params
        mu = [float(p["const[0]"]), float(p["const[1]"])]
        sig = [float(np.sqrt(p["sigma2[0]"])), float(np.sqrt(p["sigma2[1]"]))]
        fit_log.append({
            "year": y, "model_id": model_id, "n_insample": len(insample),
            "loglik": float(res_is.llf), "bear_regime": bear,
            "mu": mu, "sigma": sig,
            "P": float(p["p[0->0]"]), "Q": float(1.0 - p["p[1->0]"]),
        })

    out = pd.concat(rows).sort_index()
    out.attrs["fit_log"] = fit_log
    return out


def collapse_to_two_states(out: pd.DataFrame) -> pd.DataFrame:
    """The native two-state view, for the like-for-like comparison with Arm B."""
    two = pd.DataFrame(index=out.index)
    two["p_bear"] = out["p_bear_raw"]
    two["p_bull"] = 1.0 - out["p_bear_raw"]
    two["label"] = np.where(two["p_bear"] > 0.5, "bear", "bull")
    two["confidence"] = two[["p_bull", "p_bear"]].max(axis=1)
    two["model_id"] = out["model_id"]
    return two


# -------------------------------------------------------------------- checkpoint
def report(out: pd.DataFrame, cfg: dict, arm_b: pd.DataFrame | None = None,
           save_plot: bool = True) -> dict:
    """Build-step-4 checkpoint: confirm filtered probabilities, compare to Arm B."""
    from . import config as _cfg

    res = {}
    log = out.attrs["fit_log"]
    print("=" * 92)
    print("BUILD STEP 4 CHECKPOINT - Hamilton two-state (Arm A)")
    print("=" * 92)
    print(f"\nscored {out.index[0].date()} -> {out.index[-1].date()}  n={len(out)}")
    print("estimator: MarkovRegression(k=2, trend='c', switching_variance=True)"
          "  -- no AR term, per Markose")
    print(f"probabilities: {cfg['benchmark']['probabilities'].upper()}  "
          "(smoothed would condition on the future)")

    print("\n-- estimated regimes by fit window --")
    print(f"{'year':>6s} {'n':>6s} {'mu_bull%':>9s} {'mu_bear%':>9s} "
          f"{'sig_bull%':>10s} {'sig_bear%':>10s} {'P':>7s} {'Q':>7s}")
    for r in log[::4] + [log[-1]]:
        b, g = r["bear_regime"], 1 - r["bear_regime"]
        print(f"{r['year']:6d} {r['n_insample']:6d} {r['mu'][g] * 100:9.4f} "
              f"{r['mu'][b] * 100:9.4f} {r['sigma'][g] * 100:10.3f} "
              f"{r['sigma'][b] * 100:10.3f} {r['P']:7.4f} {r['Q']:7.4f}")

    last = log[-1]
    b, g = last["bear_regime"], 1 - last["bear_regime"]
    ratio = last["sigma"][b] / last["sigma"][g]
    mu_gap = (last["mu"][g] - last["mu"][b]) * TRADING_DAYS * 100
    res["sigma_ratio"] = float(ratio)
    print(f"\n   final fit: sigma_bear / sigma_bull = {ratio:.2f}x")
    print(f"   annualized mean gap = {mu_gap:+.1f} percentage points")
    signs_ok = last["mu"][g] > 0 > last["mu"][b] and ratio > 1
    res["markose_pattern"] = bool(signs_ok)
    print(f"   Markose sign pattern (mu_bull > 0 > mu_bear, sigma_bear > "
          f"sigma_bull): {'HOLDS' if signs_ok else 'DOES NOT HOLD'}")

    print("\n-- label distribution (Arm A has no bull_volatile state) --")
    vc = out["label"].value_counts(normalize=True)
    for s in STATES:
        print(f"   {s:14s} {vc.get(s, 0) * 100:5.1f}%")

    print("\n-- crisis response --")
    gate = {}
    for ep in cfg["ground_truth_episodes"]:
        seg = out.loc[ep["start"]:ep["end"]]
        if not len(seg):
            continue
        gate[ep["name"]] = float(seg["p_bear_raw"].mean())
        print(f"   {ep['name']:14s} mean P(bear) = {gate[ep['name']]:5.3f}")
    for name, a, bb in (("calm 2017", "2017-01-01", "2017-12-31"),
                        ("calm 2021", "2021-01-01", "2021-12-31")):
        seg = out.loc[a:bb]
        if len(seg):
            gate[name] = float(seg["p_bear_raw"].mean())
            print(f"   {name:14s} mean P(bear) = {gate[name]:5.3f}")
    res["gate"] = gate

    print("\n-- episode structure --")
    lab = out["label"]
    sw = int((lab != lab.shift()).sum())
    dur = lab.groupby((lab != lab.shift()).cumsum()).size()
    res["n_episodes"] = sw
    print(f"   episodes: {sw}   median duration: {dur.median():.0f} days   "
          f"max: {dur.max()} days")

    print("\n-- confidence --")
    amb = ((out["p_bear_raw"] > .1) & (out["p_bear_raw"] < .9)).mean() * 100
    print(f"   P(bear) mean = {out['p_bear_raw'].mean():.3f}   "
          f"days in the ambiguous 0.1-0.9 band = {amb:.1f}%")

    if arm_b is not None:
        idx = out.index.intersection(arm_b.index)
        a_bear, b_bear = out.loc[idx, "p_bear_stress"], arm_b.loc[idx, "p_bear_stress"]
        res["corr_p_bear"] = float(np.corrcoef(a_bear, b_bear)[0, 1])
        bin_a, bin_b = (a_bear > .5).astype(int), (b_bear > .5).astype(int)
        res["binary_agreement"] = float((bin_a == bin_b).mean())
        print("\n-- Arm A vs Arm B --")
        print(f"   corr P(bear_stress):        {res['corr_p_bear']:+.3f}")
        print(f"   two-state agreement:        {res['binary_agreement'] * 100:.1f}%"
              "   <- the like-for-like comparison")
        a_sw = int((out.loc[idx, "label"] != out.loc[idx, "label"].shift()).sum())
        b_sw = int((arm_b.loc[idx, "label"] != arm_b.loc[idx, "label"].shift()).sum())
        print(f"   switch counts:  Arm A {a_sw}   Arm B {b_sw}"
              f"   -> Arm A is {a_sw / max(b_sw, 1):.1f}x as jumpy")

    res_dir = _cfg.paths()["results"]
    out.to_parquet(res_dir / "arm_a_hamilton.parquet")
    with open(res_dir / "arm_a_fit_log.json", "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    print(f"\nwritten: {res_dir / 'arm_a_hamilton.parquet'}")

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_ax = 2 if arm_b is None else 3
        fig, ax = plt.subplots(n_ax, 1, figsize=(14, 2.4 * n_ax + 2), sharex=True)
        ax[0].plot(out.index, out["p_bear_raw"], lw=.8, color="#1568A8")
        ax[0].axhline(.5, ls="--", lw=.8, color="k")
        ax[0].set_title("Arm A - Hamilton filtered P(bear regime)")
        ax[1].plot(out.index, out["p_bear_raw"], lw=.7, color="#1568A8",
                   label="Arm A (Hamilton, 2-state)")
        if arm_b is not None:
            ax[1].plot(arm_b.index, arm_b["p_bear_stress"], lw=.7, alpha=.85,
                       color="#C05A16", label="Arm B (HMM, 3-state)")
        ax[1].legend(loc="upper left", fontsize=8)
        ax[1].set_title("P(bear) - Arm A vs Arm B")
        if arm_b is not None:
            idx = out.index.intersection(arm_b.index)
            dis = ((out.loc[idx, "p_bear_stress"] > .5).astype(int)
                   != (arm_b.loc[idx, "p_bear_stress"] > .5).astype(int))
            ax[2].fill_between(idx, 0, dis.astype(int), color="#6a4c93", lw=0)
            ax[2].set_ylim(0, 1)
            ax[2].set_title("days the two arms disagree on bear vs bull")
        for ep in cfg["ground_truth_episodes"]:
            for a_ in ax:
                a_.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                           color="k", alpha=.08, lw=0)
        fig.tight_layout()
        path = res_dir / "step4_hamilton.png"
        fig.savefig(path, dpi=120, facecolor="white")
        plt.close(fig)
        print(f"plot: {path}")

    return res


if __name__ == "__main__":
    from . import config, features as F, ingest

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)
    returns = F.log_returns(blocks["index"]["close"])

    arm_b = None
    bpath = config.paths()["results"] / "arm_b_hmm.parquet"
    if bpath.exists():
        arm_b = pd.read_parquet(bpath)

    report(walk_forward(returns, cfg), cfg, arm_b=arm_b)
