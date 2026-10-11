"""Binance USDT-M perpetual funding rates via data.binance.vision archives."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_VISION_MONTHLY = (
    "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    "/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
)


def _project_cache(symbol: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "cache" / f"{symbol.upper()}_funding.csv"


def fetch_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    url = _VISION_MONTHLY.format(symbol=symbol.upper(), year=year, month=month)
    resp = requests.get(url, timeout=60, headers={"User-Agent": "ai-trading-agent/1.0"})
    if resp.status_code == 404:
        return pd.DataFrame(columns=["funding_rate"])
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    df = pd.read_csv(io.BytesIO(raw))
    # Vision schema: calc_time, funding_interval_hours, last_funding_rate
    time_col = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rate_col = (
        "last_funding_rate"
        if "last_funding_rate" in df.columns
        else ("fundingRate" if "fundingRate" in df.columns else df.columns[-1])
    )
    out = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(df[time_col], unit="ms", utc=True),
            "funding_rate": df[rate_col].astype(float),
        }
    )
    return out.set_index("funding_time").sort_index()


def fetch_funding_range(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download monthly archives covering [start, end]."""
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    frames: list[pd.DataFrame] = []
    cursor = datetime(start.year, start.month, 1)
    end_m = datetime(end.year, end.month, 1)
    while cursor <= end_m:
        try:
            chunk = fetch_funding_month(symbol, cursor.year, cursor.month)
            if not chunk.empty:
                frames.append(chunk)
                print(f"  funding {symbol} {cursor.year}-{cursor.month:02d}: {len(chunk)} rows")
            else:
                print(f"  funding {symbol} {cursor.year}-{cursor.month:02d}: missing")
        except requests.RequestException as exc:
            print(f"  funding {symbol} {cursor.year}-{cursor.month:02d}: fail ({exc})")
        # next month
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)

    if not frames:
        raise OSError(f"No funding data for {symbol} between {start.date()} and {end.date()}")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[(out.index >= start) & (out.index <= end + pd.Timedelta(days=1))]


def load_or_fetch_funding(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = cache_path or _project_cache(symbol)
    if cache_path.is_file() and not refresh:
        cached = pd.read_csv(cache_path)
        if "funding_time" not in cached.columns:
            cached = pd.read_csv(cache_path, index_col=0)
            cached.index = pd.to_datetime(cached.index, utc=True, format="ISO8601")
        else:
            cached["funding_time"] = pd.to_datetime(
                cached["funding_time"], utc=True, format="ISO8601"
            )
            cached = cached.set_index("funding_time")
        cached = cached.sort_index()
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index, utc=True, format="ISO8601")
        elif cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")
        start_utc = pd.Timestamp(start)
        end_utc = pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        # Vision archives lag; accept cache that covers the start of the window
        if cached.index.min() <= start_utc + pd.Timedelta(days=2) and len(cached) > 10:
            return cached
    print(f"Fetching funding rates for {symbol}...")
    df = fetch_funding_range(symbol, start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "funding_time"
    out.to_csv(cache_path)
    print(f"Cached funding -> {cache_path} ({len(out)} rows)")
    return df


def attach_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill funding onto OHLCV bars; mark bars where funding settles."""
    out = ohlcv.copy()
    if funding.empty:
        out["funding_rate"] = 0.0
        out["funding_event"] = False
        return out

    rate = funding["funding_rate"].sort_index()
    # asof join: each bar gets latest funding info
    aligned = rate.reindex(out.index.union(rate.index)).sort_index().ffill()
    out["funding_rate"] = aligned.reindex(out.index)
    # funding event if a funding timestamp falls in this bar's open..next open
    events = pd.Series(False, index=out.index)
    bar_starts = out.index
    # map each funding time to the bar at or before it
    for ts in rate.index:
        if ts < bar_starts[0] or ts > bar_starts[-1] + pd.Timedelta(days=1):
            continue
        idx = bar_starts.get_indexer([ts], method="ffill")[0]
        if idx >= 0:
            events.iloc[idx] = True
    out["funding_event"] = events
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    return out
