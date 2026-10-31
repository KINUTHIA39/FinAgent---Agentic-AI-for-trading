"""Phase 4: monthly agentic backtest, 2005-2025.

Each month the agent reads a SESSION.md (point-in-time regime + real 6-factor signals
+ evolving portfolio state + endogenous memory) and returns a DAG plan; a deterministic
plan->weights mapping turns the plan's risk posture into target weights; we rebalance,
charge turnover costs, realise next month's return, and record per-stock outcomes into an
in-memory memory ledger. Reports cumulative wealth / Sharpe / drawdown vs equal-weight and
the S&P 500, at 0/10/20 bps cost sensitivity.

Regime is fit IN-SAMPLE on S&P monthly 1990-2004 and applied causally over 2005-2025
(no parameter look-ahead). Memory is endogenous and resets each run (no Neo4j).

Run:
    python -m finagent.research.agentic_eval.backtest --planner mock          # instant, $0
    python -m finagent.research.agentic_eval.backtest --planner llm --k 1      # ~1h, ~$6

The LLM path caches each month's planner decision under
``backtest/cache_{planner}_{regime}/`` as it goes, so an interrupted run resumes
instead of re-paying for months it already bought. Pass ``--fresh`` to ignore
the cache and re-sample every month.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from finagent.core.prompt_builder import SessionWriter
from finagent.research.agentic_eval import regime_hamilton as RH
from finagent.research.agentic_eval.regime_provider import get_provider

MD = REPO / "data" / "agentic_eval" / "market_data"
OUTDIR = REPO / "data" / "agentic_eval" / "backtest"
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "XOM", "JNJ"]
REGISTRY = (
    "- DataAgent: fetches OHLCV and market state\n"
    "- FeatureAgent: computes factor signals\n"
    "- AlphaAgent: generates BUY/SELL signals\n"
    "- RiskAgent: VaR + exposure checks\n"
    "- PortfolioAgent: position sizing\n"
    "- ExecutionAgent: places orders\n"
    "- AuditAgent: records cycle outcome\n"
)

# --- posture parameter synonyms (the LLM drifts on key names) ---
_CAP_POS = ["max_position_pct", "max_position_size_pct", "max_pos_pct", "position_size_pct",
            "position_size_pct_cap", "max_position_exposure_pct", "max_alloc_per_position_pct"]
# Only *clearly defensive* gross caps are honoured; the generic, ambiguous
# `max_portfolio_exposure_pct` / `max_cash_deploy_pct` are deliberately ignored
# (they throttled otherwise-aggressive bull plans).
_GROSS_DEFENSIVE = ["bear_stress_exposure_cap_pct", "crisis_exposure_cap_pct",
                    "max_gross_exposure", "gross_exposure_cap", "max_equity_exposure_pct"]
_GROSS_FLOOR = 0.20   # never let the gross multiplier fall below this
_SIZING = ["sizing_mode", "position_sizing_mode", "sizing_method"]
_FLAG = ["known_losing_symbols", "avoid_symbols", "suppress_symbols", "loser_symbols",
         "reduce_only_losers", "flagged_symbols"]


def _find(value, keys):
    """Recursively search a params value for the first of `keys`."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in keys:
                return v
        for v in value.values():
            r = _find(v, keys)
            if r is not None:
                return r
    elif isinstance(value, list):
        for v in value:
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def extract_posture(dag) -> dict:
    """Pull risk-posture params from a planned DAG (nodes with .params)."""
    params = [dict(getattr(n, "params", {}) or {}) for n in getattr(dag, "nodes", [])]
    def find(keys):
        for p in params:
            r = _find(p, keys)
            if r is not None:
                return r
        return None
    flagged = set()
    for keys in [_FLAG]:
        v = find(keys)
        if isinstance(v, list):
            flagged |= {str(x).upper() for x in v}
    # collect ALL flag-type lists, not just first
    for p in params:
        for k in _FLAG:
            v = _find(p, [k])
            if isinstance(v, list):
                flagged |= {str(x).upper() for x in v}
    return {
        "cap_pos": find(_CAP_POS),
        "defensive_cap": find(_GROSS_DEFENSIVE),
        "sizing": (str(find(_SIZING)).lower() if find(_SIZING) is not None else None),
        "regime_mult": find(["regime_multiplier"]),
        "reduce_only": bool(find(["reduce_only"]) is True) or (find(["reduce_only_losers"]) is not None),
        "halt": bool(find(["halt_new_longs"]) is True),
        "flagged": flagged,
    }


def _cap_alloc(weights: dict, cap: float) -> dict:
    """Water-fill: cap each weight at `cap`, redistribute excess to names with room.
    If every name hits the cap, the un-allocated remainder is left as cash."""
    w = dict(weights)
    for _ in range(12):
        over = {t: w[t] - cap for t in w if w[t] > cap + 1e-9}
        if not over:
            break
        excess = sum(over.values())
        for t in over:
            w[t] = cap
        room = {t: cap - w[t] for t in w if w[t] < cap - 1e-9}
        tot = sum(room.values())
        if tot <= 1e-12:
            break                      # all at cap -> excess becomes cash
        for t in room:
            w[t] += excess * room[t] / tot
    return w


def posture_to_weights(posture, scores, vols, w_prev, tickers):
    """Deterministic plan->weights (refined rule set). Long-only, sum<=1, rest cash.

    Gross target = sizing-tilt x regime_multiplier (floored), optionally tightened by a
    *clearly-defensive* exposure cap; the per-position cap then limits concentration
    (and structurally forces cash when n*cap < gross). reduce_only/halt vs prior weights.
    """
    cap = posture.get("cap_pos")
    cap = float(cap) if isinstance(cap, (int, float)) and cap > 0 else 0.25
    sizing = posture.get("sizing") or "score_weighted"
    rmult = posture.get("regime_mult")
    rmult = float(rmult) if isinstance(rmult, (int, float)) else 1.0
    dcap = posture.get("defensive_cap")
    flagged = posture.get("flagged") or set()

    # 1) eligible names
    elig = [t for t in tickers if t not in flagged and scores[t] > 0]
    if not elig:
        return {t: 0.0 for t in tickers}
    # 2) base weights by sizing mode (sum to 1 over eligible)
    if sizing in ("equal", "equal_weight"):
        b = {t: 1.0 for t in elig}
    elif sizing in ("risk_parity", "risk_weighted", "risk_adjusted", "inverse_vol"):
        b = {t: 1.0 / max(vols[t], 1e-4) for t in elig}
    else:                                           # score-proportional (incl. conservative/defensive)
        b = {t: max(scores[t], 0.0) for t in elig}
    s = sum(b.values()) or 1.0
    b = {t: v / s for t, v in b.items()}
    # 3) gross target G: sizing tilt x regime multiplier, floored; defensive cap if explicit
    g = {"conservative": 0.6, "defensive": 0.4, "reduced": 0.5, "minimal": 0.3}.get(sizing, 1.0)
    g *= rmult
    if isinstance(dcap, (int, float)) and dcap > 0:
        g = min(g, float(dcap))
    g = float(np.clip(g, _GROSS_FLOOR, 1.0))
    # 4) allocate G across eligible, then per-position cap (water-filled)
    target = {t: g * b[t] for t in elig}
    capped = _cap_alloc(target, cap)
    w = {t: capped.get(t, 0.0) for t in tickers}
    # 5) reduce_only / halt vs drifted prior weights
    if posture.get("halt"):
        w = {t: (w[t] if w_prev.get(t, 0.0) > 1e-6 else 0.0) for t in tickers}
    if posture.get("reduce_only"):
        w = {t: min(w[t], w_prev.get(t, 0.0)) for t in tickers}
    return w


class MemoryLedger:
    """Endogenous per-stock trade memory, built from realised monthly outcomes."""
    def __init__(self):
        self.rec = {t: [] for t in TICKERS}   # list of $ P&L per held-month

    def record(self, symbol, pnl_dollars):
        self.rec[symbol].append(pnl_dollars)

    def recall_block(self, min_trades: int = 3) -> str:
        lines, flags = [], []
        for t in TICKERS:
            pnls = self.rec[t]
            if len(pnls) < min_trades:
                continue
            n = len(pnls); wr = float(np.mean([x > 0 for x in pnls]))
            avg = float(np.mean(pnls))
            conf = round(0.4 + 0.8 * wr, 2)            # 0.4..1.2 by win-rate
            tag = "profitable" if avg > 0 else "LOSING"
            lines.append(f"  - {t}: | {n} trades | WR: {wr*100:.0f}% ({tag}) | "
                         f"Avg P&L: ${avg:+,.0f} | Conf: {conf:.2f}x")
            if wr < 0.4:
                flags.append(t)
        if not lines:
            return ""
        block = ("=== HISTORICAL CONTEXT FROM KNOWLEDGE GRAPH ===\n\n## Symbol History:\n"
                 + "\n".join(lines))
        if flags:
            block += "\n\n!! LOW CONFIDENCE (consider skipping): " + ", ".join(flags)
        return block + "\n=== END HISTORICAL CONTEXT ==="


def load_inputs(regime: str | None = None):
    """Load the monthly panel.

    ``reg`` is a regime *provider*, not a dataframe. It still supports
    ``d in reg.index`` for month selection, but the per-month read goes through
    ``reg.month(d)`` so the signal can be swapped without touching this loop.
    ``regime=None`` keeps the Hamilton filter, so behaviour is unchanged unless
    a caller asks for something else.
    """
    spx = pd.read_parquet(MD / "spx_monthly.parquet")
    rf = pd.read_parquet(MD / "rf_monthly.parquet")["rf"]
    stk = pd.read_parquet(MD / "stocks_monthly.parquet")[TICKERS]
    sig = pd.read_parquet(MD / "signals_monthly.parquet")[TICKERS]
    # regime: fit IN-SAMPLE 1990-2004, apply causally over full series
    spx_ret = (spx["ret"].dropna() * 100.0)
    is_ret = spx_ret[spx_ret.index <= "2004-12-31"]
    bull_var = (RH.BULL_VOL_ANNUAL / np.sqrt(12)) ** 2
    params = RH.fit(is_ret.values, bull_var)
    ham, _ = RH.classify(spx_ret, params=params)
    ham.index = spx_ret.index
    reg = get_provider(regime, ham, stk.index)
    # trailing 12m vol per stock (for risk-parity sizing)
    vol = stk.rolling(12, min_periods=6).std()
    return spx, rf, stk, sig, reg, vol, params


def mock_posture(regime_label, ledger):
    """Deterministic stand-in for the LLM: regime-driven defensiveness (validation only)."""
    flags = {t for t in TICKERS if len(ledger.rec[t]) >= 3
             and np.mean([x > 0 for x in ledger.rec[t]]) < 0.4}
    if regime_label == "bull":
        return {"cap_pos": 0.20, "defensive_cap": None, "sizing": "score_weighted",
                "regime_mult": 1.0, "reduce_only": False, "halt": False, "flagged": set()}
    return {"cap_pos": 0.12, "defensive_cap": None, "sizing": "conservative",
            "regime_mult": 1.0, "reduce_only": False, "halt": False, "flagged": flags}


def _posture_cache_dir(sfx: str, path_i: int) -> Path:
    return OUTDIR / f"cache{sfx}" / f"path{path_i}"


def _session_digest(session_md: str) -> str:
    """Digest of a briefing, ignoring its wall-clock timestamp line.

    SESSION.md carries a ``Timestamp (UTC)`` line that changes on every write,
    so a digest over the raw text would never match across runs. Everything
    else in the briefing is a deterministic function of the replayed state,
    which is exactly what a resume needs to compare.
    """
    body = "\n".join(ln for ln in session_md.splitlines()
                      if not ln.startswith("- Timestamp (UTC):"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _posture_to_json(posture: dict, fb: bool, session_md: str, dag=None) -> dict:
    """Serialise one month's planner decision for replay.

    Only the posture is kept, because that is the whole of what the planner
    contributes: everything downstream (weights, turnover, returns, the ledger)
    is deterministic given it. The briefing digest is stored alongside so a
    resumed run can prove the cached decision was taken against the same
    briefing rather than silently splicing two different experiments together.
    """
    return {
        "posture": {**posture, "flagged": sorted(posture.get("flagged") or set())},
        "used_fallback": bool(fb),
        "session_sha256": _session_digest(session_md),
        "n_nodes": len(getattr(dag, "nodes", []) or []),
        "cached_at": pd.Timestamp.utcnow().isoformat(),
    }


def _posture_from_json(payload: dict) -> tuple[dict, bool]:
    posture = dict(payload["posture"])
    posture["flagged"] = set(posture.get("flagged") or [])
    return posture, bool(payload.get("used_fallback"))


def run(planner: str = "mock", cost_bps: float = 10.0, k: int = 1,
        ckpt_every: int = 20, regime: str | None = None,
        fresh: bool = False):
    spx, rf, stk, sig, reg, vol, params = load_inputs(regime)
    # `path0.parquet` is the archived dissertation run and is treated as
    # read-only: every new run writes under its own planner+regime name.
    # Sharing a filename across planners once silently replaced the
    # archived LLM result with a mock one, and an LLM re-run would do the
    # same non-reproducibly for $6.
    sfx = f"_{planner}_{reg.name}"
    # backtest months = where signals exist and a next-month return exists
    months = [d for d in sig.index if d in stk.index and d in reg.index]
    months = [d for d in months if d <= stk.index[-2]]   # need t+1 return
    months.sort()

    # Built on the first cache miss, so a fully cached run replays without an
    # API key.
    llm = None

    def _llm():
        nonlocal llm
        if llm is None:
            from finagent.core.claude_client import ClaudeClient
            from finagent.core.dag_planner import LLMDAGPlanner
            _load_env()
            llm = LLMDAGPlanner(client=ClaudeClient())
        return llm

    sw_dir = tempfile.mkdtemp(prefix="bt_session_")
    writer = SessionWriter(config_dir=sw_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for path_i in range(k):
        ledger = MemoryLedger()
        cache_dir = _posture_cache_dir(sfx, path_i)
        called = replayed = 0
        nav = 1.0
        w_prev = {t: 0.0 for t in TICKERS}     # drifted weights from last month
        last_ret = 0.0
        recs = []
        t0 = time.perf_counter()
        for i, d in enumerate(months):
            nxt = stk.index[stk.index.get_loc(d) + 1]
            scores = {t: float(sig.loc[d, t]) for t in TICKERS}
            vols = {t: float(vol.loc[d, t]) if pd.notna(vol.loc[d, t]) else 0.05 for t in TICKERS}
            rg = reg.month(d)
            reg_label, p_bull = rg.label, rg.p_bull

            # build SESSION.md
            portfolio = {"nav": round(nav * 1_000_000, 2),
                         "cash": round(w_prev_cash(w_prev) * nav * 1_000_000, 2),
                         "position_count": int(sum(v > 1e-6 for v in w_prev.values())),
                         "daily_pnl_pct": round(last_ret, 4)}
            regime_ctx = rg.briefing_ctx()
            candidates = [{"symbol": t, "signal": "BUY", "score": round(scores[t], 3),
                           "rsi": 0.0, "win_rate": _wr(ledger, t)} for t in TICKERS]
            recall = ledger.recall_block()
            session_md = Path(writer.write(portfolio_state=portfolio, regime=regime_ctx,
                                           memory_recall=recall, candidates=candidates,
                                           cycle_id=f"bt_{d.strftime('%Y%m')}")).read_text(encoding="utf-8")

            # plan -> posture. A cached month is replayed rather than re-bought;
            # because the cache holds the planner's decision and every later step
            # is deterministic given it, the state trajectory a resume rebuilds is
            # identical to the one an uninterrupted run would have had, so the
            # briefing at the resume point is the same too.
            if planner == "llm":
                ck = cache_dir / f"posture_{d.strftime('%Y-%m')}.json"
                if ck.exists() and not fresh:
                    payload = json.loads(ck.read_text(encoding="utf-8"))
                    if payload.get("session_sha256") != _session_digest(session_md):
                        raise SystemExit(
                            f"stale cache: the stored decision for {d:%Y-%m} was "
                            f"taken against a different briefing.\n"
                            f"  cache: {ck}\n"
                            f"Delete that directory or pass --fresh to re-sample.")
                    posture, fb = _posture_from_json(payload)
                    replayed += 1
                else:
                    result = _llm().plan(session_md, REGISTRY)
                    posture = extract_posture(result.dag)
                    fb = bool(result.used_fallback)
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    ck.write_text(
                        json.dumps(_posture_to_json(posture, fb, session_md, result.dag),
                                   indent=2), encoding="utf-8")
                    called += 1
            else:
                posture = mock_posture(reg_label, ledger)
                fb = False

            w = posture_to_weights(posture, scores, vols, w_prev, TICKERS)

            # turnover + cost
            turnover = sum(abs(w[t] - w_prev.get(t, 0.0)) for t in TICKERS)
            cost = (cost_bps / 1e4) * turnover

            # realise next-month return
            stock_ret = {t: float(stk.loc[nxt, t]) for t in TICKERS}
            w_cash = max(0.0, 1.0 - sum(w.values()))
            gross_ret = sum(w[t] * stock_ret[t] for t in TICKERS) + w_cash * float(rf.get(nxt, 0.0))
            net_ret = gross_ret - cost
            nav_next = nav * (1 + net_ret)

            # record per-stock $ P&L into the ledger (only held names)
            for t in TICKERS:
                if w[t] > 1e-6:
                    ledger.record(t, w[t] * nav * 1_000_000 * stock_ret[t])

            recs.append({"date": nxt, "regime": reg_label, "p_bull": p_bull,
                         "nav": nav_next, "gross_ret": gross_ret, "net_ret": net_ret,
                         "turnover": turnover, "cost_bps_applied": cost_bps,
                         "w_cash": w_cash, "used_fallback": fb,
                         **{f"w_{t}": w[t] for t in TICKERS},
                         **{f"r_{t}": stock_ret[t] for t in TICKERS}})
            # drift weights for next turnover/halt/reduce calc
            denom = (1 + gross_ret) or 1.0
            w_prev = {t: w[t] * (1 + stock_ret[t]) / denom for t in TICKERS}
            nav, last_ret = nav_next, net_ret

            if planner == "llm" and (i + 1) % ckpt_every == 0:
                pd.DataFrame(recs).to_parquet(OUTDIR / f"path{path_i}{sfx}_partial.parquet", index=False)
                print(f"  [{i+1}/{len(months)}] {d.strftime('%Y-%m')} nav={nav:.3f} "
                      f"regime={reg_label} new={called} cached={replayed} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        df = pd.DataFrame(recs).set_index("date")
        df.to_parquet(OUTDIR / f"path{path_i}{sfx}.parquet")
        paths.append(df)
        tail = (f"  [{called} new calls, {replayed} replayed from cache]"
                if planner == "llm" else "")
        print(f"path {path_i}: final NAV {nav:.3f}  "
              f"({time.perf_counter()-t0:.0f}s){tail}", flush=True)

    # ---- baselines + metrics (use path 0) ----
    _report(paths, stk, spx, rf, months, cost_bps, planner, params)
    return paths


def w_prev_cash(w_prev):
    return max(0.0, 1.0 - sum(w_prev.values()))


def _wr(ledger, t):
    pnls = ledger.rec[t]
    return round(float(np.mean([x > 0 for x in pnls])) if len(pnls) >= 3 else 0.0, 2)


def _ann_metrics(monthly_ret: pd.Series, rf_m: pd.Series):
    r = monthly_ret.dropna()
    ex = r - rf_m.reindex(r.index).fillna(0)
    cagr = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (ex.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    nav = (1 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return {"CAGR%": cagr * 100, "vol%": vol * 100, "Sharpe": sharpe,
            "maxDD%": mdd * 100, "final_x": float((1 + r).prod())}


def _report(paths, stk, spx, rf, months, cost_bps, planner, params):
    df = paths[0]
    idx = df.index
    rf_m = rf.reindex(idx).fillna(0)
    # agent at 0/10/20 bps (recompute net return from gross - cost; weights fixed)
    print("\n" + "=" * 64)
    print(f"BACKTEST RESULTS  (planner={planner}, {len(months)} months "
          f"{idx.min().date()}..{idx.max().date()})")
    print("=" * 64)
    print(f"\nHamilton params: BULL mu={params['mu'][0]:.2f} BEAR mu={params['mu'][1]:.2f} "
          f"P={params['P']:.3f} Q={params['Q']:.3f}")
    print(f"\nAgent, transaction-cost sensitivity:")
    print(f"  {'cost':>6s}{'CAGR%':>9s}{'vol%':>8s}{'Sharpe':>8s}{'maxDD%':>9s}{'final_x':>9s}")
    for c in (0, 10, 20):
        net = df["gross_ret"] - (c / 1e4) * df["turnover"]
        m = _ann_metrics(net, rf_m)
        star = "  <- live" if c == cost_bps else ""
        print(f"  {c:>4d}bp{m['CAGR%']:>9.1f}{m['vol%']:>8.1f}{m['Sharpe']:>8.2f}"
              f"{m['maxDD%']:>9.1f}{m['final_x']:>9.2f}{star}")
    # baselines
    ew = stk.reindex(idx)[TICKERS].mean(axis=1)              # monthly-rebalanced equal weight
    spx_ret = spx["ret"].reindex(idx)
    print("\nBaselines:")
    for name, series in [("Equal-weight 8", ew), ("S&P 500", spx_ret)]:
        m = _ann_metrics(series, rf_m)
        print(f"  {name:14s}{m['CAGR%']:>9.1f}{m['vol%']:>8.1f}{m['Sharpe']:>8.2f}"
              f"{m['maxDD%']:>9.1f}{m['final_x']:>9.2f}")
    bull = (df["regime"] == "bull").mean()
    print(f"\nRegime exposure: {bull*100:.0f}% bull-months; mean cash weight {df['w_cash'].mean()*100:.0f}%")
    if "used_fallback" in df:
        print(f"Planner fallbacks: {int(df['used_fallback'].sum())}/{len(df)}")


def _load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["mock", "llm"], default="mock")
    ap.add_argument("--regime", choices=["hamilton", "arm_c"], default=None,
                    help="regime signal for the briefing "
                         "(default: hamilton, or $FINAGENT_REGIME)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the posture cache and re-call the planner for "
                         "every month (costs full price)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run(planner=args.planner, cost_bps=args.cost_bps, k=args.k,
        regime=args.regime, fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
