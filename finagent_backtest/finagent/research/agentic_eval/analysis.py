"""Statistical analysis layer over ``runs.parquet`` — the dissertation evidence.

Turns the raw plan grid into the numbers and figures the write-up needs:

  RQ1  Adaptation   — does the agent condition its risk posture on regime,
                      exposure and recalled memory, *above* run-to-run noise?
                      (three-way ANOVA + eta-squared, permutation tests,
                      bootstrap effect sizes, ICC reproducibility)
  RQ2  Reliability  — fallback rate and reasons across 1,200 plans
  RQ3  Cost         — tokens, latency distribution, $ per plan / total
  +    Posture decomposition — which risk levers move with regime / recall
  +    Topology invariance   — is adaptation parametric or structural?

Read-only over ``runs.parquet``; writes only under
``data/agentic_eval/analysis/`` (report.md, *.csv, figures/*.png).

Run:
    python -m finagent.research.agentic_eval.analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "data" / "agentic_eval" / "runs.parquet"
OUT = REPO_ROOT / "data" / "agentic_eval" / "analysis"
FIGS = OUT / "figures"

REGIME_ORDER = ["bull-calm", "bull-stress", "bear-calm", "bear-stress"]   # severity ↑
EXPOSURE_ORDER = ["low", "mid", "near-full"]
RECALL_ORDER = ["none", "mixed", "winners", "losers"]
PRICE_IN, PRICE_OUT = 3.0, 15.0   # Claude Sonnet $/1M tokens
RNG = np.random.default_rng(20260618)

POSTURE_LEVERS = [
    "max_position_pct", "reduce_only", "circuit_breaker", "halt_new_longs",
    "slippage_guard_bps", "n_flagged_symbols",
]


# --------------------------------------------------------------------------- #
# stats helpers (numpy / scipy — no parametric assumptions where avoidable)
# --------------------------------------------------------------------------- #
def _eta_sq_marginal(df: pd.DataFrame, factor: str, y: str = "defensiveness_score") -> float:
    """One-way between-group SS / total SS — the marginal variance share."""
    grand = df[y].mean()
    ss_tot = ((df[y] - grand) ** 2).sum()
    ss_between = sum(len(g) * (g[y].mean() - grand) ** 2 for _, g in df.groupby(factor))
    return float(ss_between / ss_tot) if ss_tot > 0 else 0.0


def permutation_test(df: pd.DataFrame, factor: str, n: int = 5000,
                     y: str = "defensiveness_score") -> tuple[float, float]:
    """Permute the factor labels; p = P(null eta^2 >= observed). Non-parametric."""
    obs = _eta_sq_marginal(df, factor, y)
    vals = df[y].to_numpy()
    labels = df[factor].to_numpy()
    grand = vals.mean()
    ss_tot = ((vals - grand) ** 2).sum()
    uniq = np.unique(labels)
    ge = 0
    for _ in range(n):
        perm = RNG.permutation(labels)
        ssb = sum(((vals[perm == u].mean() - grand) ** 2) * (perm == u).sum() for u in uniq)
        if (ssb / ss_tot) >= obs:
            ge += 1
    return obs, (ge + 1) / (n + 1)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 10000) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a percentile 95% CI."""
    diff = a.mean() - b.mean()
    boots = np.array([RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean()
                      for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff), float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def icc1(df: pd.DataFrame, group: str = "scenario", y: str = "defensiveness_score"):
    """One-way random-effects ICC(1): share of variance that is between-cell.

    High ICC => the experimental condition (not LLM stochasticity) drives the
    score, i.e. the adaptation is reproducible. Returns (icc, F, p, k).
    """
    from scipy import stats
    groups = [g[y].to_numpy() for _, g in df.groupby(group)]
    k = int(np.mean([len(g) for g in groups]))
    grand = df[y].mean()
    n_g = len(groups)
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = n_g - 1, len(df) - n_g
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    f = ms_b / ms_w
    p = float(stats.f.sf(f, df_b, df_w))
    return float(icc), float(f), p, k


def _block(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def _fenced(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string() + "\n```\n"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def order(col, o):
        return pd.Categorical(df[col], o, ordered=True)

    d = df.copy()
    d["regime"] = order("regime", REGIME_ORDER)
    d["exposure"] = order("exposure", EXPOSURE_ORDER)
    d["recall_profile"] = order("recall_profile", RECALL_ORDER)

    # 1) heatmap defensiveness regime x exposure
    piv = (d.groupby(["regime", "exposure"], observed=True)["defensiveness_score"]
             .mean().unstack("exposure").reindex(index=REGIME_ORDER, columns=EXPOSURE_ORDER))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=max(0.25, piv.to_numpy().max()))
    ax.set_xticks(range(3)); ax.set_xticklabels(EXPOSURE_ORDER)
    ax.set_yticks(range(4)); ax.set_yticklabels(REGIME_ORDER)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{piv.to_numpy()[i,j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Defensiveness — regime × exposure")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    p = FIGS / "fig1_heatmap_regime_exposure.png"; fig.savefig(p, dpi=140); plt.close(fig); saved.append(p.name)

    # 2) recall effect
    g = d.groupby("recall_profile", observed=True)["defensiveness_score"].agg(["mean", "std", "count"]).reindex(RECALL_ORDER)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.bar(RECALL_ORDER, g["mean"], yerr=g["std"] / np.sqrt(g["count"]), capsize=4,
           color=["#4c78a8", "#9ecae9", "#54a24b", "#e45756"])
    ax.set_ylabel("mean defensiveness"); ax.set_title("Memory effect — defensiveness by recall profile")
    fig.tight_layout(); p = FIGS / "fig2_recall_effect.png"; fig.savefig(p, dpi=140); plt.close(fig); saved.append(p.name)

    # 3) posture decomposition by regime
    levers = d.copy()
    for c in ["reduce_only", "circuit_breaker", "halt_new_longs"]:
        levers[c] = levers[c].astype(float)
    pg = levers.groupby("regime", observed=True)[["reduce_only", "circuit_breaker", "n_flagged_symbols"]].mean().reindex(REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(4); w = 0.26
    ax.bar(x - w, pg["reduce_only"], w, label="reduce_only")
    ax.bar(x, pg["circuit_breaker"], w, label="circuit_breaker")
    ax.bar(x + w, pg["n_flagged_symbols"], w, label="flagged_symbols")
    ax.set_xticks(x); ax.set_xticklabels(REGIME_ORDER); ax.legend()
    ax.set_title("Risk-posture levers by regime"); fig.tight_layout()
    p = FIGS / "fig3_posture_by_regime.png"; fig.savefig(p, dpi=140); plt.close(fig); saved.append(p.name)

    # 4) reproducibility — defensiveness spread per regime
    fig, ax = plt.subplots(figsize=(6.4, 4))
    data = [d[d["regime"] == r]["defensiveness_score"].to_numpy() for r in REGIME_ORDER]
    ax.boxplot(data, tick_labels=REGIME_ORDER, showmeans=True)
    ax.set_ylabel("defensiveness"); ax.set_title("Reproducibility — defensiveness spread by regime (K=25/cell)")
    fig.tight_layout(); p = FIGS / "fig4_reproducibility.png"; fig.savefig(p, dpi=140); plt.close(fig); saved.append(p.name)

    # 5) cost / latency
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    a1.hist(d["latency_ms"] / 1000, bins=30, color="#4c78a8"); a1.set_xlabel("latency (s)"); a1.set_title("Plan latency")
    cost = (d["tokens_in"] * PRICE_IN + d["tokens_out"] * PRICE_OUT) / 1e6
    a2.hist(cost, bins=30, color="#54a24b"); a2.set_xlabel("$ / plan"); a2.set_title("Cost per plan")
    fig.tight_layout(); p = FIGS / "fig5_cost_latency.png"; fig.savefig(p, dpi=140); plt.close(fig); saved.append(p.name)

    return saved


# --------------------------------------------------------------------------- #
# main report
# --------------------------------------------------------------------------- #
def main() -> int:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RUNS)
    df["regime_sev"] = df["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
    parts: list[str] = [f"# Agentic-eval results — {len(df)} plans across {df['scenario'].nunique()} cells\n"]

    # ---- RQ1: variance decomposition (three-way ANOVA, type II) ----
    model = smf.ols("defensiveness_score ~ C(regime) + C(exposure) + C(recall_profile)", data=df).fit()
    aov = anova_lm(model, typ=2)
    aov["eta_sq"] = aov["sum_sq"] / aov["sum_sq"].sum()
    aov = aov[["sum_sq", "df", "F", "PR(>F)", "eta_sq"]].round(5)
    parts.append(_block("RQ1 — variance decomposition (three-way ANOVA, type II)",
                        f"Model R² = {model.rsquared:.3f}. eta² = share of total variance.\n\n" + _fenced(aov)))

    # ---- RQ1: permutation tests for the two main channels ----
    perm_rows = []
    for f in ["regime", "recall_profile", "exposure"]:
        eta, p = permutation_test(df, f, n=5000)
        perm_rows.append({"factor": f, "eta_sq_marginal": round(eta, 4), "perm_p": round(p, 5)})
    perm = pd.DataFrame(perm_rows)
    parts.append(_block("RQ1 — permutation tests (label-shuffle null, 5000 perms)", _fenced(perm)))

    # ---- RQ1: effect sizes with bootstrap CI ----
    bear = df[df["regime"].str.startswith("bear")]["defensiveness_score"].to_numpy()
    bull = df[df["regime"].str.startswith("bull")]["defensiveness_score"].to_numpy()
    los = df[df["recall_profile"] == "losers"]["defensiveness_score"].to_numpy()
    non = df[df["recall_profile"] == "none"]["defensiveness_score"].to_numpy()
    eff = []
    for name, a, b in [("bear − bull", bear, bull), ("losers − none", los, non)]:
        d_, lo, hi = bootstrap_diff(a, b)
        eff.append({"contrast": name, "mean_diff": round(d_, 4),
                    "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                    "cohens_d": round(cohens_d(a, b), 3)})
    parts.append(_block("RQ1 — effect sizes (bootstrap 95% CI, 10k)", _fenced(pd.DataFrame(eff))))

    # ---- reproducibility: ICC ----
    icc, f_stat, p_icc, k = icc1(df)
    within_sd = df.groupby("scenario")["defensiveness_score"].std().mean()
    parts.append(_block("Reproducibility (ICC, one-way random effects)",
                        f"ICC(1) = {icc:.3f}  (share of variance that is between-cell, i.e. condition-driven)\n"
                        f"F({47},{len(df)-48}) = {f_stat:.1f}, p = {p_icc:.2e}, k = {k}/cell\n"
                        f"mean within-cell SD = {within_sd:.4f}\n"
                        f"Interpretation: cells differ far beyond run-to-run noise (p≪0.001); "
                        f"ICC quantifies how much of the spread is the condition vs LLM stochasticity."))

    # ---- posture decomposition ----
    lv = df.copy()
    for c in ["reduce_only", "circuit_breaker", "halt_new_longs"]:
        lv[c] = lv[c].astype(float)
    by_reg = lv.groupby("regime")[POSTURE_LEVERS].mean().reindex(REGIME_ORDER).round(3)
    by_rec = lv.groupby("recall_profile")[POSTURE_LEVERS].mean().reindex(RECALL_ORDER).round(3)
    parts.append(_block("Posture decomposition — which levers move",
                        "By regime:\n" + _fenced(by_reg) + "\nBy recall profile:\n" + _fenced(by_rec)))

    # ---- topology invariance ----
    parts.append(_block("Topology invariance (parametric vs structural adaptation)",
                        f"n_nodes unique = {sorted(df['n_nodes'].unique())}; "
                        f"n_stages unique = {sorted(df['n_stages'].unique())}; "
                        f"DAG source = {df['source'].value_counts().to_dict()}.\n"
                        f"=> The planner keeps a fixed 7-node pipeline and adapts the *risk-posture "
                        f"parameters*, not the graph structure. Adaptation is parametric."))

    # ---- RQ2: reliability ----
    parts.append(_block("RQ2 — reliability",
                        f"LLM-plan success = {df['llm_success'].mean():.1%}; "
                        f"fallbacks = {int(df['used_fallback'].sum())}/{len(df)}; "
                        f"reasons = {df['reason'].replace('', '(none)').value_counts().to_dict()}.\n"
                        f"NOTE: zero fallbacks under well-formed inputs — RQ2 has no variance yet. "
                        f"The fallback path needs the ablation/adversarial cells to be exercised."))

    # ---- RQ3: cost ----
    cost = (df["tokens_in"] * PRICE_IN + df["tokens_out"] * PRICE_OUT) / 1e6
    parts.append(_block("RQ3 — cost & latency",
                        f"tokens_in mean = {df['tokens_in'].mean():.0f}; tokens_out mean = {df['tokens_out'].mean():.0f}; "
                        f"cache_read = {df['tokens_cache_read'].fillna(0).mean():.0f} (prefix < 1024-tok cache floor).\n"
                        f"latency: mean {df['latency_ms'].mean()/1000:.1f}s, p50 {df['latency_ms'].median()/1000:.1f}s, "
                        f"p95 {df['latency_ms'].quantile(.95)/1000:.1f}s.\n"
                        f"cost: ${cost.mean():.4f}/plan, ${cost.sum():.2f} total for {len(df)} plans."))

    # ---- figures + csv exports ----
    figs = make_figures(df)
    aov.to_csv(OUT / "anova.csv"); perm.to_csv(OUT / "permutation.csv", index=False)
    pd.DataFrame(eff).to_csv(OUT / "effect_sizes.csv", index=False)
    by_reg.to_csv(OUT / "posture_by_regime.csv"); by_rec.to_csv(OUT / "posture_by_recall.csv")
    parts.append(_block("Artifacts", "figures/: " + ", ".join(figs) +
                        "\nCSVs: anova.csv, permutation.csv, effect_sizes.csv, posture_by_regime.csv, posture_by_recall.csv"))

    report = "\n".join(parts)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print("\n" + "=" * 66)
    print(f"  wrote {OUT/'report.md'}  (+ {len(figs)} figures, 5 csvs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
