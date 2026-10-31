"""Unit-test the sentiment pieces on synthetic data, before the corpus lands."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src import config, sentiment as S

cfg = config.load()

print("1. LOUGHRAN-McDONALD SCORING")
tests = ["Apple posts record profit and strong growth",
         "Company reports loss amid declining demand and litigation",
         "Results may be uncertain pending possible regulatory approval",
         "Firm reports tax liability on capital depreciation of crude assets"]
lm = S.lm_scores(tests)
for t, r in zip(tests, lm.itertuples()):
    print(f"   tone {r.lm_net_tone:+.2f}  neg% {r.lm_negative_pct:.3f}  "
          f"unc% {r.lm_uncertainty_pct:.3f}  | {t}")
print("   row 4 is the finance-specific check: tax/liability/capital/")
print("   depreciation/crude are NEUTRAL here and negative in Harvard-IV.")
assert lm.lm_net_tone.iloc[0] > 0 and lm.lm_net_tone.iloc[1] < 0
assert lm.lm_uncertainty_pct.iloc[2] > 0
assert lm.lm_negative_pct.iloc[3] == 0, "finance words leaked as negative"
print("   PASS")

print("\n2. DECAY IMPUTATION FOR ZERO-NEWS DAYS")
cal = pd.bdate_range("2020-01-01", periods=14)
arts = pd.DataFrame({
    "trading_day": [cal[0], cal[0], cal[5]],
    "finbert_score": [-0.8, -0.6, 0.4],
    "lm_net_tone": [-1.0, -1.0, 1.0],
    "lm_negative_pct": [0.2, 0.2, 0.0],
    "lm_uncertainty_pct": [0.1, 0.1, 0.0]})
d = S.aggregate_daily(arts, cal, cfg)
hl = cfg["news"]["decay_half_life_days"]
print(f"   half-life {hl} days; two articles on day 0, one on day 5")
print(f"   {'date':12s} {'articles':>9s} {'finbert':>9s}")
for i, (dt, row) in enumerate(d.iterrows()):
    print(f"   {str(dt.date()):12s} {row['article_count']:9.0f} "
          f"{row['mean_finbert']:+9.4f}")
    if i == 7:
        break
d0 = d["mean_finbert"].iloc[0]
d3 = d["mean_finbert"].iloc[3]
print(f"   day 0 = {d0:+.4f} (mean of -0.8, -0.6)")
print(f"   day 3 = {d3:+.4f}, expected {d0 * 0.5**(3/hl):+.4f} after decay")
assert abs(d3 - d0 * 0.5 ** (3 / hl)) < 1e-9, "decay is wrong"
assert d["article_count"].iloc[1] == 0
print("   PASS -- decays toward neutral rather than holding a stale headline")

print("\n3. AGGREGATION SHAPE")
need = ["mean_finbert", "mean_lm_tone", "lm_uncertainty", "article_count",
        "sentiment_disp", "mean_finbert_ewma", "article_count_ewma"]
missing = [c for c in need if c not in d.columns]
print(f"   columns present: {len(d.columns)}   missing: {missing or 'none'}")
assert not missing
assert len(d) == len(cal)
print("   PASS")
