"""The look-ahead checklist as executable tests (spec 9).

These decide whether any number in the deck is trustworthy. This release keeps the
checks that apply to the equity build: the HMM, the Hamilton refit schedule,
the agent panel and the gate. Checks written for the retired FX version of the
project were removed.

Ingest-level tests use the parquet cache in data/raw, so they are fast after
the first `python -m src.ingest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import copy
import json

from src import agents, benchmark, config, detector, features, ingest, labels, scoring

CFG = config.load()
END = pd.Timestamp(CFG["data"]["end"])


@pytest.fixture(scope="module")
def blocks():
    return ingest.build_panel(END, CFG)


# ------------------------------------------------------------------ step 2 live
@pytest.fixture(scope="module")
def feats(blocks):
    return features.build_features(blocks, CFG)


def test_hmm_feature_order_matches_config(feats):
    assert list(feats["hmm"].columns) == CFG["features"]["hmm_features"]


# ------------------------------------------------------------------ step 3 live
def _small_cfg():
    """Cheap detector config so the suite stays fast."""
    c = copy.deepcopy(CFG)
    c["detector"]["n_restarts"] = 3
    c["detector"]["n_iter"] = 60
    return c


@pytest.fixture(scope="module")
def fitted(feats):
    """One 3-state fit on an in-sample slice, reused across the step-3 tests."""
    X = feats["hmm"].dropna().loc[:"2012-12-31"].to_numpy()
    cfg = _small_cfg()
    model, _, _ = detector.fit_hmm(X, cfg, base_seed=int(CFG["run"]["seed"]))
    return model, X


def test_hmm_scored_on_filtered_not_viterbi(fitted):
    """Scored probabilities at t must not change when future data is appended.

    The forward recursion satisfies this; predict_proba (forward-backward) and
    predict (Viterbi over the whole path) do not. This is the test that
    separates them.
    """
    model, X = fitted
    cut = len(X) // 2
    full = detector.filtered_probabilities(model, X)
    trunc = detector.filtered_probabilities(model, X[:cut])
    assert np.allclose(full[:cut], trunc, atol=1e-10), \
        "filtered probabilities changed when future data was appended"


def test_filtered_differs_from_smoothed(fitted):
    """Guards against silently swapping in predict_proba.

    If these ever coincide, the forward recursion has been replaced by
    forward-backward and every Arm B number is conditioned on the future.
    """
    model, X = fitted
    filt = detector.filtered_probabilities(model, X)
    smooth = detector.smoothed_probabilities(model, X)
    assert not np.allclose(filt, smooth, atol=1e-6)
    # Smoothing uses strictly more information, so it should be more decisive.
    assert smooth.max(axis=1).mean() >= filt.max(axis=1).mean() - 1e-9


def test_filtered_probabilities_are_a_distribution(fitted):
    model, X = fitted
    p = detector.filtered_probabilities(model, X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_hmm_state_mapping_uses_insample_only(fitted):
    """State->label mapping must not change when future data is appended.

    Mapping on full-sample properties is an easy accidental look-ahead and is
    invisible afterwards, because the labels look sensible either way.
    """
    model, X = fitted
    names = CFG["features"]["hmm_features"]
    cut = len(X) // 2
    m_trunc = detector.map_states_to_labels(model, X[:cut], CFG, names)
    m_trunc2 = detector.map_states_to_labels(model, X[:cut], CFG, names)
    assert m_trunc == m_trunc2, "mapping is not deterministic"
    assert set(m_trunc.values()) == set(detector.STATES)


# ------------------------------------------------------------------ step 4 live


def test_hamilton_refit_schedule_matches_hmm():
    """Both arms refit annually on the same boundaries.

    Different schedules would make B - A partly a comparison of refit cadence
    rather than of states and features.
    """
    ra = config.paths()["results"] / "arm_a_hamilton.parquet"
    rb = config.paths()["results"] / "arm_b_hmm.parquet"
    if not (ra.exists() and rb.exists()):
        pytest.skip("run `python -m src.benchmark` and `python -m src.detector` first")
    a, b = pd.read_parquet(ra), pd.read_parquet(rb)
    assert a.index.equals(b.index), "arms scored on different days"
    assert sorted(a["fit_end"].unique()) == sorted(b["fit_end"].unique())


# ------------------------------------------------------------------ step 6 live


def test_prompts_carry_no_calendar_dates(feats):
    """Anonymised payloads must contain no date, and no absolute price level.

    A calendar date is the cheapest possible way for the model to recognise a
    period it has memorised instead of inferring anything from the data.
    """
    import re

    views = agents.build_agent_views(feats, CFG)
    day = views["volatility"].dropna().index[-500]
    payload = {r: agents._window(v, day) for r, v in views.items()}
    blob = json.dumps(payload)
    assert not re.search(r"(19|20)\d{2}-\d{2}-\d{2}", blob), "ISO date leaked"
    assert not re.search(r"(19|20)\d{2}", blob), "bare year leaked"
    for key in payload["volatility"]:
        assert key == "t" or key.startswith("t-"), f"non-offset period key: {key}"

    # The dates-on variant must actually differ, or the ablation is vacuous.
    dated = agents._window(views["volatility"], day, anonymise=False)
    assert any(re.match(r"\d{4}-\d{2}-\d{2}", k) for k in dated)


def test_veto_requires_unanimity_and_critic():
    """Either condition alone must not change the label.

    The detector is a fitted model with a track record; one dissenting slice is
    not grounds to discard it, and a critic acting alone is a single opinion.
    """
    prop = {"label": "range", "confidence": 0.9}
    unan = {r: {"state": "risk_off", "informative": True} for r in agents.AGENT_INPUTS}
    split = dict(unan); split["rates"] = {"state": "risk_on", "informative": True}
    override = {"verdict": "override", "final_state": "risk_off",
                "confidence_multiplier": 1.0, "recency_bias_risk": "low",
                "objection": "x", "falsifier": "y"}
    accept = dict(override, verdict="accept", final_state="range")

    both = agents.apply_panel_verdict(prop, unan, override)
    assert both["label"] == "risk_off" and both["action"] == "veto"

    no_unanimity = agents.apply_panel_verdict(prop, split, override)
    assert no_unanimity["label"] == "range"
    assert no_unanimity["action"] == "override_blocked_not_unanimous"

    no_critic = agents.apply_panel_verdict(prop, unan, accept)
    assert no_critic["label"] == "range" and no_critic["action"] == "passthrough"


def test_arm_c_matches_arm_b_off_gate():
    """Off-gate, Arm C IS Arm B. Any drift there means the panel leaked outside
    its gate and the C - B decomposition stops being interpretable."""
    pc = config.paths()["results"] / "arm_c_agentic.parquet"
    pb = config.paths()["results"] / "arm_b_hmm.parquet"
    if not pc.exists():
        pytest.skip("run `python -m src.agents` first")
    c, b = pd.read_parquet(pc), pd.read_parquet(pb)
    off = ~detector.candidate_days(b, CFG)
    assert (c.loc[off, "label"] == b.loc[off, "label"]).all()
    assert np.allclose(c.loc[off, "confidence"], b.loc[off, "confidence"])


def test_arm_c_probabilities_agree_with_label():
    """The emitted label must be the argmax of the emitted probabilities.

    Without re-deriving the vector after an override, a vetoed day would carry a
    label its probabilities contradict and every probabilistic metric would be
    scoring the detector rather than the panel.
    """
    pc = config.paths()["results"] / "arm_c_agentic.parquet"
    if not pc.exists():
        pytest.skip("run `python -m src.agents` first")
    c = pd.read_parquet(pc)
    probs = c[scoring.PROB_COLS]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    argmax = probs.to_numpy().argmax(axis=1)
    implied = [detector.STATES[i] for i in argmax]
    assert (pd.Series(implied, index=c.index) == c["label"]).all()


def test_seeds_are_fixed_and_logged(fitted):
    """Reruns reproduce, and per-restart log-likelihoods are recorded."""
    _, X = fitted
    cfg = _small_cfg()
    m1, ll1, all1 = detector.fit_hmm(X, cfg, base_seed=7)
    m2, ll2, all2 = detector.fit_hmm(X, cfg, base_seed=7)
    # Threaded BLAS reorders reductions, so identical seeds agree to about the
    # 12th significant figure rather than bitwise. Reproducibility means the
    # same optimum, not the same rounding.
    assert np.allclose(all1, all2, rtol=1e-9), "fit is not reproducible at a fixed seed"
    assert np.isclose(ll1, ll2, rtol=1e-9)
    assert len(all1) == cfg["detector"]["n_restarts"]


def test_restart_spread_is_recorded_not_assumed():
    """The fit log must carry every restart's score.

    On this data one restart in ten reaches the optimum and the spread runs to
    ~2,250 log-likelihood points, so a single-start fit would land on a bad mode
    nine times out of ten. That is only visible if the scores are kept.
    """
    path = config.paths()["results"] / "arm_b_fit_log.json"
    if not path.exists():
        pytest.skip("run `python -m src.detector` first")
    log = json.loads(path.read_text())
    assert log, "empty fit log"
    for rec in log:
        assert len(rec["all_logliks"]) == CFG["detector"]["n_restarts"]
        # all_logliks is stored rounded to 2dp for readability; best_loglik is
        # full precision, so compare at the rounding granularity.
        assert rec["best_loglik"] >= max(rec["all_logliks"]) - 0.01


def test_gate_fires_at_episode_onsets():
    """The panel must get a look at the transitions it exists to judge.

    Measured around each episode ONSET, not inside it: mid-episode the detector
    is flat at P=1.000 and there is genuinely nothing to adjudicate, so counting
    gated days inside the window understates coverage badly.

    Two episodes get no gated day near onset -- the 2022 Fed cycle and the 2024
    yen unwind. The latter is structural: it was JPY-idiosyncratic (USDJPY 20d
    vol 17.1% while the other three sat below their calm-2017 means), so a
    dollar-factor detector cannot see it at all.
    """
    path = config.paths()["results"] / "arm_b_hmm.parquet"
    if not path.exists():
        pytest.skip("run `python -m src.detector` first")
    out = pd.read_parquet(path)
    mask = detector.candidate_days(out, CFG)
    covered = 0
    for ep in CFG["ground_truth_episodes"]:
        s = pd.Timestamp(ep["start"])
        win = mask.loc[s - pd.Timedelta(days=60): s + pd.Timedelta(days=60)]
        covered += int(win.sum() > 0)
    assert covered >= 5, f"only {covered}/7 episode onsets have a gated day"


def test_gated_subset_is_a_workable_size():
    """Too few gated days and C - B is unmeasurable; too many and it is unaffordable."""
    path = config.paths()["results"] / "arm_b_hmm.parquet"
    if not path.exists():
        pytest.skip("run `python -m src.detector` first")
    out = pd.read_parquet(path)
    n = int(detector.candidate_days(out, CFG).sum())
    assert 200 <= n <= 1200, f"{n} candidate days is outside the workable band"
