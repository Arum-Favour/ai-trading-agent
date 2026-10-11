"""Technical indicators for rule-based signals."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _htf_ema(close: pd.Series, rule: str, span: int) -> pd.Series:
    """Higher-timeframe EMA, forward-filled onto the base index."""
    if not isinstance(close.index, pd.DatetimeIndex) or len(close) < span + 5:
        return ema(close, min(200, max(span * 4, 50)))

    htf_close = close.resample(rule).last().dropna()
    if len(htf_close) < max(10, span // 2):
        return ema(close, min(200, max(span * 4, 50)))

    htf = ema(htf_close, span)
    return htf.reindex(close.index, method="ffill")


def add_indicators(
    df: pd.DataFrame,
    htf_rule: str = "4h",
    htf_ema_span: int = 50,
    funding_lookback: int = 24,
) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], 12)
    out["ema_slow"] = ema(out["close"], 26)
    out["ema_trend"] = ema(out["close"], 200)
    out["rsi"] = rsi(out["close"], 14)
    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    mid, upper, lower = bollinger(out["close"], 20, 2.0)
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_signal"] = ema(out["macd"], 9)
    out["htf_ema"] = _htf_ema(out["close"], htf_rule, htf_ema_span)
    out["htf_bull"] = out["close"] > out["htf_ema"]
    if "funding_rate" in out.columns:
        out["funding_avg"] = (
            out["funding_rate"]
            .astype(float)
            .rolling(funding_lookback, min_periods=max(3, funding_lookback // 3))
            .mean()
        )
    return out
