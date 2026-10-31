"""Arm B: 3-state Gaussian HMM on one equity index, walk-forward.

Fit protocol:
  - Expanding window, refit annually on data through 31 Dec of year y-1
  - >= 10 random restarts, best log-likelihood wins. hmmlearn's GaussianHMM has
    NO n_init parameter -- the restart loop is manual.
  - Score year y with FILTERED probabilities (forward algorithm only).

THE FILTERED/SMOOTHED TRAP. hmmlearn's `predict_proba` and `score_samples`
return SMOOTHED posteriors from the full forward-backward pass, and `predict`
runs Viterbi over the whole path. Both condition on the future. Neither may be
used for scoring. The forward recursion is therefore written out explicitly
below -- about fifteen auditable lines, no dependence on hmmlearn internals --
so the distinction lives in the code rather than in a method name.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# hmmlearn logs "Model is not converging" whenever EM stops on the tolerance
# rather than the iteration cap -- i.e. on normal convergence. Across 10
# restarts x ~20 refits that is pure noise; real failures raise instead.
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")

STATES = ("bull_quiet", "bull_volatile", "bear_stress")
# Hamilton emits two states, so a like-for-like comparison needs the three
# collapsed to two. Both bull states become one; bear_stress stands alone.
TWO_STATE = {"bull_quiet": "bull", "bull_volatile": "bull", "bear_stress": "bear"}

OUTPUT_COLUMNS = ["label", "p_bull_quiet", "p_bull_volatile", "p_bear_stress",
                  "confidence", "entropy_confidence", "model_id",
                  "fit_start", "fit_end"]


@dataclass
class RegimeOutput:
    """Uniform output contract, shared by all three arms."""

    date: pd.Timestamp
    label: str
    probability_vector: dict[str, float]   # filtered, sums to 1
    confidence: float
    feature_values: dict[str, float]
    model_id: str
    fit_window: tuple[pd.Timestamp, pd.Timestamp]
    provenance: dict = field(default_factory=dict)


# ----------------------------------------------------------------------- params
def n_parameters(k: int, d: int) -> int:
    """Free parameters of a k-state Gaussian HMM with full covariance."""
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


# -------------------------------------------------------------------------- fit
def fit_hmm(X: np.ndarray, cfg: dict, base_seed: int,
            n_states: int | None = None) -> tuple[object, float, list[float]]:
    """Fit with manual restarts. Returns (best_model, best_loglik, all_logliks).

    Every restart's score comes back so the spread is logged rather than
    assumed: a narrow spread means the likelihood surface is benign, a wide one
    means single-start fits were a real risk.
    """
    from hmmlearn.hmm import GaussianHMM

    d = cfg["detector"]
    k = n_states or int(d["n_states"])
    best, best_ll, all_ll = None, -np.inf, []
    for i in range(int(d["n_restarts"])):
        m = GaussianHMM(n_components=k, covariance_type=d["covariance_type"],
                        n_iter=int(d["n_iter"]), tol=float(d["tol"]),
                        random_state=base_seed + i)
        try:
            m.fit(X)
            ll = float(m.score(X))
        except Exception:
            ll = -np.inf
        all_ll.append(ll)
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("every HMM restart failed")
    return best, best_ll, all_ll


# --------------------------------------------------------------------- filtering
def emission_logprob(model, X: np.ndarray) -> np.ndarray:
    """Gaussian emission log-densities, (n_obs, n_states).

    Built from model.means_ and model.covars_ directly rather than a private
    hmmlearn method, so the filter below depends on nothing internal.
    """
    out = np.empty((len(X), model.n_components))
    for j in range(model.n_components):
        out[:, j] = multivariate_normal.logpdf(
            X, mean=model.means_[j], cov=model.covars_[j], allow_singular=True)
    return out


def filtered_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Forward-algorithm posteriors P(s_t | x_1..x_t), (n_obs, n_states).

    Explicitly NOT predict_proba (smoothed, forward-backward) and NOT predict
    (Viterbi over the whole path). The value at t uses observations up to and
    including t and nothing after -- which is the whole point of the arm.
    """
    logb = emission_logprob(model, X)
    b = np.exp(logb - logb.max(axis=1, keepdims=True))
    A, alpha = model.transmat_, np.empty_like(logb)

    a = model.startprob_ * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ A) * b[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else np.full(len(a), 1.0 / len(a))
    return alpha


def smoothed_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Forward-backward posteriors. NEVER used for scoring -- diagnostics only."""
    return model.predict_proba(X)


# ------------------------------------------------------------------- state names
def map_states_to_labels(model, X_insample: np.ndarray, cfg: dict,
                         feature_names: list[str]) -> dict[int, str]:
    """Map unlabelled HMM states {0,1,2} to economic names.

    LOOK-AHEAD: uses IN-SAMPLE statistics only, computed inside the fit window.
    Naming states from full-sample properties is look-ahead bias, and it is
    invisible afterwards because the labels look sensible either way.

    Rule:
      1. highest mean drawdown_from_252d_high -> bear_stress
      2. of the remaining two, lowest mean realized_vol_20d -> bull_quiet
      3. the last -> bull_volatile
    """
    dd_i = feature_names.index("drawdown_from_252d_high")
    rv_i = feature_names.index("realized_vol_20d")

    post = model.predict_proba(X_insample)          # in-sample only, not scoring
    w = post / post.sum(axis=0, keepdims=True)
    means = w.T @ X_insample

    order = list(range(model.n_components))
    bear = max(order, key=lambda j: means[j, dd_i])
    rest = [j for j in order if j != bear]
    quiet = min(rest, key=lambda j: means[j, rv_i])
    vol = [j for j in rest if j != quiet][0]
    return {bear: "bear_stress", quiet: "bull_quiet", vol: "bull_volatile"}


# ------------------------------------------------------------------ walk forward
def walk_forward(features: pd.DataFrame, cfg: dict,
                 score_end: str | None = None) -> pd.DataFrame:
    """Annual expanding-window fit/score.

    For each scoring year y: fit on everything through 31 Dec y-1, freeze, then
    run the forward filter from the start of the sample to each day in y. The
    filter is forward-only, so a value in year y depends on data up to that day
    and nothing later -- and running it from the sample start rather than
    restarting each January means it carries its state across the boundary, as
    an operator's would.
    """
    dcfg, scfg = cfg["detector"], cfg["sample"]
    base_seed = int(cfg["run"]["seed"])
    names = list(features.columns)

    feats = features.dropna()
    start = pd.Timestamp(scfg["score_start"])
    end = min(pd.Timestamp(score_end or scfg["score_end"]), feats.index.max())

    rows, fit_log = [], []
    for y in range(start.year, end.year + 1):
        fit_end = pd.Timestamp(f"{y - 1}-12-31")
        insample = feats.loc[:fit_end]
        if len(insample) < 250:
            continue

        model, ll, all_ll = fit_hmm(insample.to_numpy(), cfg,
                                    base_seed + 1000 * y)
        mapping = map_states_to_labels(model, insample.to_numpy(), cfg, names)

        through_y = feats.loc[:pd.Timestamp(f"{y}-12-31")]
        probs = filtered_probabilities(model, through_y.to_numpy())
        mask = through_y.index.year == y
        idx, P = through_y.index[mask], probs[mask]

        named = np.zeros((len(P), len(STATES)))
        for j, nm in mapping.items():
            named[:, STATES.index(nm)] = P[:, j]

        # Gate threshold from IN-SAMPLE filtered confidence in this fit window,
        # so the cut adapts to the detector without seeing the scored year.
        in_mask = through_y.index <= fit_end
        thr = float(np.percentile(
            probs[in_mask].max(axis=1),
            float(cfg["agents"]["gate"]["confidence_percentile"])))

        model_id = f"hmm{dcfg['n_states']}_{y}_seed{base_seed + 1000 * y}"
        ent = -(named * np.log(np.clip(named, 1e-12, 1))).sum(axis=1) / np.log(
            len(STATES))
        rows.append(pd.DataFrame({
            "label": [STATES[i] for i in named.argmax(axis=1)],
            "p_bull_quiet": named[:, 0], "p_bull_volatile": named[:, 1],
            "p_bear_stress": named[:, 2],
            "confidence": named.max(axis=1), "entropy_confidence": 1.0 - ent,
            "gate_threshold": thr, "model_id": model_id,
            "fit_start": insample.index[0], "fit_end": fit_end,
        }, index=idx))

        fit_log.append({"year": y, "model_id": model_id,
                        "n_insample": len(insample), "best_loglik": ll,
                        "loglik_spread": float(np.ptp(all_ll)),
                        "all_logliks": [round(x, 2) for x in all_ll],
                        "mapping": {int(k): v for k, v in mapping.items()}})

    out = pd.concat(rows).sort_index()
    out.attrs["fit_log"] = fit_log
    return out


def collapse_to_two_states(out: pd.DataFrame) -> pd.DataFrame:
    """Three states -> two, for the like-for-like comparison with Hamilton.

    Hamilton has no third state, so a three-class agreement statistic against it
    is structurally capped. Collapsing both bull states into one makes at least
    one comparison honest on state count.
    """
    two = out.copy()
    two["label"] = out["label"].map(TWO_STATE)
    two["p_bull"] = out["p_bull_quiet"] + out["p_bull_volatile"]
    two["p_bear"] = out["p_bear_stress"]
    two["confidence"] = two[["p_bull", "p_bear"]].max(axis=1)
    return two[["label", "p_bull", "p_bear", "confidence", "model_id"]]


def candidate_days(out: pd.DataFrame, cfg: dict) -> pd.Series:
    """Days the agent panel may act on.

    Four triggers, unioned: the argmax changes; confidence falls below the
    in-sample percentile for that fit window; the probability vector moves
    materially in one day; or the day sits within a window of a label change.

    The last two exist because a confidence-only gate is blind to crises -- a
    saturated detector holds P=1.000 straight through a crash, so a
    confidence-based gate fires zero times exactly where the interesting
    decisions are.
    """
    g = cfg["agents"]["gate"]
    flip = out["label"] != out["label"].shift()
    mask = flip.copy() if g.get("on_argmax_change", True) else pd.Series(
        False, index=out.index)
    mask |= out["confidence"] < out["gate_threshold"]

    thr = g.get("prob_change_above")
    if thr is not None:
        pcols = [f"p_{s}" for s in STATES]
        mask |= out[pcols].diff().abs().sum(axis=1) > float(thr)

    w = int(g.get("window_around_change", 0))
    for shift in range(-w, w + 1):
        if shift:
            mask |= flip.shift(shift, fill_value=False)
    return mask.rename("is_candidate")


# ---------------------------------------------------------------- model selection
def selection_table(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """BIC/ICL across candidate state counts, on the in-sample window.

    ICL adds the posterior entropy to BIC, penalising models whose states
    overlap. Expect BIC to favour more than three: a Gaussian mixture absorbs
    fat tails by adding states, and equity returns have plenty. Three is a
    taxonomy choice, and the table should be shown saying so.
    """
    feats = features.dropna().loc[:pd.Timestamp(cfg["sample"]["burn_in_end"])]
    X = feats.to_numpy()
    n, d = X.shape
    seed = int(cfg["run"]["seed"])

    rows = []
    for k in cfg["detector"]["model_selection_table"]:
        model, ll, all_ll = fit_hmm(X, cfg, seed, n_states=k)
        p = n_parameters(k, d)
        bic = -2 * ll + p * np.log(n)
        post = np.clip(model.predict_proba(X), 1e-12, 1)
        rows.append({"k_states": k, "n_params": p, "loglik": ll, "BIC": bic,
                     "ICL": bic + 2 * -(post * np.log(post)).sum(),
                     "loglik_spread": float(np.ptp(all_ll))})
    return pd.DataFrame(rows).set_index("k_states")


# -------------------------------------------------------------------- checkpoint
def report(out: pd.DataFrame, cfg: dict, save_plot: bool = True) -> dict:
    """Build-step-3 checkpoint.

    THE GATE: bear_stress must light up in 2000-02, 2008, 2020 and 2022. If it
    does not, the state mapping or the filter is wrong and everything
    downstream measures the bug.
    """
    from . import config as _cfg

    res = {}
    log = out.attrs["fit_log"]
    print("=" * 92)
    print("BUILD STEP 3 CHECKPOINT - 3-state Gaussian HMM (Arm B)")
    print("=" * 92)
    print(f"\nscored {out.index[0].date()} -> {out.index[-1].date()}  n={len(out)}")

    print("\n-- restart spread (why >=10, and why not n_init, which does not exist) --")
    spreads = [r["loglik_spread"] for r in log]
    res["max_loglik_spread"] = max(spreads)
    print(f"   {len(log)} annual refits x {cfg['detector']['n_restarts']} restarts")
    print(f"   spread across restarts: median={np.median(spreads):.1f}  "
          f"max={max(spreads):.1f}")
    worst = max(log, key=lambda r: r["loglik_spread"])
    at_best = sum(1 for x in worst["all_logliks"]
                  if x > worst["best_loglik"] - 1)
    print(f"   worst year {worst['year']}: spread {worst['loglik_spread']:.1f}, "
          f"{at_best} of {len(worst['all_logliks'])} restarts reached the optimum")

    print("\n-- label distribution --")
    vc = out["label"].value_counts(normalize=True)
    for s in STATES:
        print(f"   {s:14s} {vc.get(s, 0) * 100:5.1f}%")
    res["label_shares"] = {s: float(vc.get(s, 0)) for s in STATES}

    print("\n-- THE GATE: mean P(bear_stress) in each episode --")
    gate = {}
    for ep in cfg["ground_truth_episodes"]:
        seg = out.loc[ep["start"]:ep["end"]]
        if not len(seg):
            continue
        gate[ep["name"]] = float(seg["p_bear_stress"].mean())
        share = float((seg["label"] == "bear_stress").mean())
        print(f"   {ep['name']:14s} mean P={gate[ep['name']]:5.3f}   "
              f"days labelled bear={share * 100:5.1f}%")
    for name, a, b in (("calm 2017", "2017-01-01", "2017-12-31"),
                       ("calm 2021", "2021-01-01", "2021-12-31")):
        seg = out.loc[a:b]
        if len(seg):
            gate[name] = float(seg["p_bear_stress"].mean())
            print(f"   {name:14s} mean P={gate[name]:5.3f}")
    res["gate"] = gate
    majors = [gate.get(k, 0) for k in ("gfc", "covid", "bear_2022")]
    ok = all(v > 0.5 for v in majors) and gate.get("calm 2017", 1) < 0.25
    res["gate_pass"] = ok
    print(f"\n   GATE: {'PASS' if ok else 'FAIL'} "
          f"(GFC, COVID and 2022 above 0.5; calm 2017 below 0.25)")

    print("\n-- episode structure --")
    lab = out["label"]
    switches = int((lab != lab.shift()).sum())
    dur = lab.groupby((lab != lab.shift()).cumsum()).size()
    res["n_episodes"] = switches
    print(f"   episodes: {switches}   median duration: {dur.median():.0f} days   "
          f"max: {dur.max()} days")
    print(f"   EFFECTIVE SAMPLE is {switches} episodes, not {len(out)} days -- "
          f"that is what the significance tests see")

    print("\n-- two-state collapse (for the like-for-like Hamilton comparison) --")
    two = collapse_to_two_states(out)
    tvc = two["label"].value_counts(normalize=True)
    print(f"   bull {tvc.get('bull', 0) * 100:.1f}%   "
          f"bear {tvc.get('bear', 0) * 100:.1f}%")

    print("\n-- gating load for Arm C --")
    cand = candidate_days(out, cfg)
    res["gated_days"] = int(cand.sum())
    print(f"   {int(cand.sum())} candidate days ({cand.mean() * 100:.1f}% of "
          f"scored days)  -> ~{int(cand.sum()) * 5} panel calls")

    res_dir = _cfg.paths()["results"]
    out.to_parquet(res_dir / "arm_b_hmm.parquet")
    with open(res_dir / "arm_b_fit_log.json", "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    print(f"\nwritten: {res_dir / 'arm_b_hmm.parquet'}")

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax[0].stackplot(out.index, out["p_bull_quiet"], out["p_bull_volatile"],
                        out["p_bear_stress"],
                        labels=["bull_quiet", "bull_volatile", "bear_stress"],
                        colors=["#1568A8", "#7B4BA8", "#C05A16"], lw=0)
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc="upper left", ncol=3, fontsize=8)
        ax[0].set_title("Arm B - filtered state probabilities (forward algorithm only)")
        ax[1].plot(out.index, out["p_bear_stress"], lw=.8, color="#C05A16")
        ax[1].axhline(.5, ls="--", lw=.8, color="k")
        for ep in cfg["ground_truth_episodes"]:
            for a_ in ax:
                a_.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                           color="k", alpha=.08, lw=0)
        ax[1].set_title("P(bear_stress) with hand-labelled episodes shaded")
        fig.tight_layout()
        path = res_dir / "step3_hmm_states.png"
        fig.savefig(path, dpi=120, facecolor="white")
        plt.close(fig)
        print(f"plot: {path}")

    return res


if __name__ == "__main__":
    from . import config, features as F, ingest

    cfg = config.load()
    blocks = ingest.build_panel(cfg=cfg)
    feats = F.build_features(blocks, cfg)

    print("model selection (in-sample window only)")
    print(selection_table(feats["hmm"], cfg).round(1).to_string())
    print()
    report(walk_forward(feats["hmm"], cfg), cfg)
