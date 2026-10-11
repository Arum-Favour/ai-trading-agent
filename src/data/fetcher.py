"""Fetch OHLCV from Binance-compatible public APIs (no API key required)."""

from __future__ import annotations

import time
from typing import Literal

import pandas as pd
import requests

Interval = Literal[
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
]

# Prefer vision mirror first — api.binance.com is often blocked by DNS/geo.
KLINE_BASES = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
)

_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

_active_base: str | None = None


def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_KLINE_COLUMNS)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _get_json(params: dict) -> list:
    """GET klines JSON, trying mirrors until one works."""
    global _active_base
    errors: list[str] = []
    bases = list(KLINE_BASES)
    if _active_base and _active_base in bases:
        bases.remove(_active_base)
        bases.insert(0, _active_base)

    for base in bases:
        try:
            resp = requests.get(base, params=params, timeout=30)
            if resp.status_code == 451:
                errors.append(f"{base}: geo-restricted (451)")
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                errors.append(f"{base}: unexpected payload {type(data).__name__}")
                continue
            _active_base = base
            return data
        except requests.RequestException as exc:
            errors.append(f"{base}: {exc}")
            continue

    detail = "; ".join(errors[:4])
    raise OSError(
        f"All Binance kline endpoints failed. Tried {len(bases)} hosts. "
        f"Use --csv or --synthetic. Last errors: {detail}"
    )


def fetch_klines(
    symbol: str,
    interval: Interval,
    limit: int = 500,
) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    rows = _get_json(params)
    if not rows:
        raise ValueError("No kline data returned")
    return _rows_to_df(rows)


def fetch_klines_paginated(
    symbol: str,
    interval: Interval,
    total_bars: int,
) -> pd.DataFrame:
    """Fetch up to several thousand bars via pagination."""
    if total_bars <= 1000:
        return fetch_klines(symbol, interval, limit=total_bars)

    frames: list[pd.DataFrame] = []
    remaining = total_bars
    end_time: int | None = None

    while remaining > 0:
        batch = min(remaining, 1000)
        params: dict = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": batch,
        }
        if end_time is not None:
            params["endTime"] = end_time
        rows = _get_json(params)
        if not rows:
            break
        frames.append(_rows_to_df(rows))
        end_time = int(rows[0][0]) - 1
        remaining -= len(rows)
        time.sleep(0.15)

    if not frames:
        raise ValueError("No kline data returned")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.tail(total_bars)
