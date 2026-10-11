"""Synthetic OHLCV for offline backtests when live APIs are unavailable."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gbm_ohlcv(
    bars: int = 2000,
    start_price: float = 50_000.0,
    bar_hours: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = bar_hours / (24 * 365)
    mu, sigma = 0.08, 0.55
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), bars)
    closes = start_price * np.exp(np.cumsum(shocks))
    opens = np.roll(closes, 1)
    opens[0] = start_price
    spread = rng.uniform(0.001, 0.008, bars)
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    volume = rng.uniform(100, 5000, bars)
    idx = pd.date_range("2024-01-01", periods=bars, freq=f"{bar_hours}h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )
