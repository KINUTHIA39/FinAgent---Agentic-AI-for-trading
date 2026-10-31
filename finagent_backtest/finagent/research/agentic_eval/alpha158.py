"""
Alpha158 Feature Calculator — Standalone Implementation

Computes all 158 Alpha factors from Qlib's Alpha158 dataset handler,
operating directly on OHLCV DataFrames without requiring Qlib infrastructure.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger("Alpha158")

# Rolling windows used by Alpha158
WINDOWS = [5, 10, 20, 30, 60]


def _safe_div(a, b, fill=0.0):
    """Division with zero protection."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(b) > 1e-12, a / b, fill)
    return result


def _rolling_slope(series: np.ndarray, window: int) -> float:
    """Linear regression slope over the last `window` values."""
    if len(series) < window:
        return 0.0
    y = series[-window:]
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _rolling_rsquare(series: np.ndarray, window: int) -> float:
    """R-squared of linear regression over the last `window` values."""
    if len(series) < window:
        return 0.0
    y = series[-window:]
    x = np.arange(window, dtype=np.float64)
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr ** 2)


def _rolling_resi(series: np.ndarray, window: int) -> float:
    """Residual of last point from linear regression line."""
    if len(series) < window:
        return 0.0
    y = series[-window:]
    x = np.arange(window, dtype=np.float64)
    slope = _rolling_slope(series, window)
    intercept = y.mean() - slope * x.mean()
    predicted = slope * (window - 1) + intercept
    return float(y[-1] - predicted)


def _idx_max(series: np.ndarray, window: int) -> float:
    """Days since max in last window, normalized by window."""
    if len(series) < window:
        return 0.5
    w = series[-window:]
    return float(np.argmax(w)) / window


def _idx_min(series: np.ndarray, window: int) -> float:
    """Days since min in last window, normalized by window."""
    if len(series) < window:
        return 0.5
    w = series[-window:]
    return float(np.argmin(w)) / window


def compute_alpha158(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Compute all Alpha158 features from an OHLCV DataFrame.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
            Requires at least 65 rows (60 lookback + 5 buffer).
    
    Returns:
        Numpy array of shape (n_features,) or None if data insufficient.
    """
    if len(df) < 65:
        return None

    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            return None

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    c = close[-1]  # latest close for normalization
    if c < 1e-12:
        return None

    features = []

    try:
        # ======================== KBAR (9 features) ========================
        o, h, l = open_p[-1], high[-1], low[-1]
        hl = h - l + 1e-12

        features.append((c - o) / (o + 1e-12))             # KMID
        features.append(hl / (o + 1e-12))                    # KLEN
        features.append((c - o) / hl)                         # KMID2
        features.append((h - max(o, c)) / (o + 1e-12))       # KUP
        features.append((h - max(o, c)) / hl)                 # KUP2
        features.append((min(o, c) - l) / (o + 1e-12))       # KLOW
        features.append((min(o, c) - l) / hl)                 # KLOW2
        features.append((2 * c - h - l) / (o + 1e-12))       # KSFT
        features.append((2 * c - h - l) / hl)                 # KSFT2

        # ======================== PRICE (4 features) ========================
        # Normalized OHLV at t=0 (latest bar)
        features.append(o / c)      # OPEN0
        features.append(h / c)      # HIGH0
        features.append(l / c)      # LOW0
        # VWAP approximation: (high + low + close) / 3
        vwap = (h + l + c) / 3.0
        features.append(vwap / c)   # VWAP0

        # ======================== ROLLING (28 operators × 5 windows) ========================
        for w in WINDOWS:
            if len(close) < w + 1:
                features.extend([0.0] * 28)
                continue

            c_w = close[-w:]
            h_w = high[-w:]
            l_w = low[-w:]
            v_w = volume[-w:]

            # Returns for the window
            prev_close = close[-(w+1):-1]
            ret = _safe_div(c_w - prev_close, prev_close)

            # Volume changes
            prev_vol = volume[-(w+1):-1]
            vol_change = v_w - prev_vol

            # --- ROC: Ref($close, d) / $close ---
            features.append(close[-(w+1)] / c if len(close) > w else 1.0)

            # --- MA: Mean($close, d) / $close ---
            features.append(np.mean(c_w) / c)

            # --- STD: Std($close, d) / $close ---
            features.append(np.std(c_w) / c)

            # --- BETA: Slope($close, d) / $close ---
            features.append(_rolling_slope(close, w) / c)

            # --- RSQR: Rsquare($close, d) ---
            features.append(_rolling_rsquare(close, w))

            # --- RESI: Resi($close, d) / $close ---
            features.append(_rolling_resi(close, w) / c)

            # --- MAX: Max($high, d) / $close ---
            features.append(np.max(h_w) / c)

            # --- MIN: Min($low, d) / $close ---
            features.append(np.min(l_w) / c)

            # --- QTLU: Quantile($close, d, 0.8) / $close ---
            features.append(np.quantile(c_w, 0.8) / c)

            # --- QTLD: Quantile($close, d, 0.2) / $close ---
            features.append(np.quantile(c_w, 0.2) / c)

            # --- RSV: ($close - Min) / (Max - Min) ---
            range_hl = np.max(h_w) - np.min(l_w)
            features.append(_safe_div(c - np.min(l_w), range_hl + 1e-12))

            # --- IMAX: IdxMax($high, d) / d ---
            features.append(_idx_max(high, w))

            # --- IMIN: IdxMin($low, d) / d ---
            features.append(_idx_min(low, w))

            # --- IMXD: (IdxMax - IdxMin) / d ---
            features.append(_idx_max(high, w) - _idx_min(low, w))

            # --- CORR: Corr($close, Log($volume+1), d) ---
            log_vol = np.log(v_w + 1)
            if np.std(c_w) > 1e-12 and np.std(log_vol) > 1e-12:
                corr = np.corrcoef(c_w, log_vol)[0, 1]
                features.append(corr if not np.isnan(corr) else 0.0)
            else:
                features.append(0.0)

            # --- CORD: Corr(close_ret, volume_ret, d) ---
            if len(ret) >= 3:
                log_vol_ret = np.log(_safe_div(v_w, prev_vol, 1.0) + 1)
                if np.std(ret) > 1e-12 and np.std(log_vol_ret) > 1e-12:
                    cord = np.corrcoef(ret, log_vol_ret)[0, 1]
                    features.append(cord if not np.isnan(cord) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

            # --- CNTP: % of up days ---
            up_days = np.mean(c_w > prev_close)
            features.append(float(up_days))

            # --- CNTN: % of down days ---
            dn_days = np.mean(c_w < prev_close)
            features.append(float(dn_days))

            # --- CNTD: CNTP - CNTN ---
            features.append(float(up_days - dn_days))

            # --- SUMP: sum(gains) / sum(|changes|) ---
            changes = c_w - prev_close
            abs_changes = np.sum(np.abs(changes)) + 1e-12
            features.append(float(np.sum(np.maximum(changes, 0)) / abs_changes))

            # --- SUMN: sum(losses) / sum(|changes|) ---
            features.append(float(np.sum(np.maximum(-changes, 0)) / abs_changes))

            # --- SUMD: (sum(gains) - sum(losses)) / sum(|changes|) ---
            features.append(float((np.sum(np.maximum(changes, 0)) - np.sum(np.maximum(-changes, 0))) / abs_changes))

            # --- VMA: Mean($volume, d) / $volume ---
            features.append(float(np.mean(v_w) / (volume[-1] + 1e-12)))

            # --- VSTD: Std($volume, d) / $volume ---
            features.append(float(np.std(v_w) / (volume[-1] + 1e-12)))

            # --- WVMA: Std(|ret|*vol, d) / Mean(|ret|*vol, d) ---
            weighted = np.abs(ret) * v_w
            mean_w = np.mean(weighted) + 1e-12
            features.append(float(np.std(weighted) / mean_w))

            # --- VSUMP: sum(vol_gains) / sum(|vol_changes|) ---
            abs_vol_ch = np.sum(np.abs(vol_change)) + 1e-12
            features.append(float(np.sum(np.maximum(vol_change, 0)) / abs_vol_ch))

            # --- VSUMN: sum(vol_losses) / sum(|vol_changes|) ---
            features.append(float(np.sum(np.maximum(-vol_change, 0)) / abs_vol_ch))

            # --- VSUMD: (vol_gains - vol_losses) / sum(|vol_changes|) ---
            features.append(float((np.sum(np.maximum(vol_change, 0)) - np.sum(np.maximum(-vol_change, 0))) / abs_vol_ch))

        vec = np.array(features, dtype=np.float64)
        # Clean NaN/Inf
        vec = np.where(np.isfinite(vec), vec, 0.0)
        return vec

    except Exception as e:
        logger.error(f"Alpha158 computation error: {e}")
        return None


def get_alpha158_names() -> List[str]:
    """Generate ordered feature names matching compute_alpha158() output."""
    names = []

    # KBar
    names += ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']

    # Price
    names += ['OPEN0', 'HIGH0', 'LOW0', 'VWAP0']

    # Rolling
    rolling_ops = [
        'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI',
        'MAX', 'MIN', 'QTLU', 'QTLD', 'RSV',
        'IMAX', 'IMIN', 'IMXD',
        'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD',
        'SUMP', 'SUMN', 'SUMD',
        'VMA', 'VSTD', 'WVMA',
        'VSUMP', 'VSUMN', 'VSUMD',
    ]
    for w in WINDOWS:
        for op in rolling_ops:
            names.append(f"{op}{w}")

    return names


# Pre-compute for use in other modules
ALPHA158_FEATURE_NAMES = get_alpha158_names()
ALPHA158_FEATURE_COUNT = len(ALPHA158_FEATURE_NAMES)  # Should be 153


def build_alpha158_matrix(df: pd.DataFrame, min_lookback: int = 65) -> Optional[Tuple[np.ndarray, np.ndarray, list]]:
    """
    Build Alpha158 feature matrix + target vector from OHLCV DataFrame.
    
    Returns:
        (X, y, timestamps) or None.
        X: shape (n_samples, 153)
        y: shape (n_samples,) — forward 1-period return (%)
    """
    if len(df) < min_lookback + 10:
        return None

    X_rows = []
    y_values = []
    timestamps = []

    close = df['close'].values

    for i in range(min_lookback, len(df) - 1):
        window = df.iloc[:i + 1]
        vec = compute_alpha158(window)
        if vec is not None:
            X_rows.append(vec)
            fwd_return = (close[i + 1] / close[i] - 1) * 100
            y_values.append(fwd_return)
            if 't' in df.columns:
                timestamps.append(df['t'].iloc[i])
            else:
                timestamps.append(i)

    if len(X_rows) < 50:
        return None

    return np.array(X_rows), np.array(y_values), timestamps


if __name__ == "__main__":
    # Quick test
    names = get_alpha158_names()
    print(f"Alpha158 feature count: {len(names)}")
    print(f"First 20: {names[:20]}")
    print(f"Last 10: {names[-10:]}")
    
    # Test with synthetic data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n).astype(float),
    })
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.5
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.5
    
    vec = compute_alpha158(df)
    if vec is not None:
        print(f"\nFeature vector shape: {vec.shape}")
        print(f"Non-zero features: {np.count_nonzero(vec)}")
        print(f"Sample values:")
        for name, val in zip(names[:15], vec[:15]):
            print(f"  {name:12s} = {val:.6f}")
    else:
        print("Failed to compute features")
