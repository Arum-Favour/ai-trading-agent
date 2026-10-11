"""Load OHLCV from CSV with flexible column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_TIME_ALIASES = (
    "open_time",
    "timestamp",
    "time",
    "date",
    "datetime",
    "Date",
    "Timestamp",
)
_OHLCV = {
    "open": ("open", "Open", "o"),
    "high": ("high", "High", "h"),
    "low": ("low", "Low", "l"),
    "close": ("close", "Close", "c", "price"),
    "volume": ("volume", "Volume", "vol", "v"),
}


def _pick(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def load_ohlcv_csv(path: str | Path, bars: int | None = None) -> pd.DataFrame:
    """Read CSV into a DatetimeIndex OHLCV frame (UTC when possible)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    time_col = _pick(df.columns, _TIME_ALIASES)
    if time_col is None and not isinstance(df.index, pd.DatetimeIndex):
        # Try first column as time if it parses
        candidate = df.columns[0]
        parsed = pd.to_datetime(df[candidate], utc=True, errors="coerce")
        if parsed.notna().mean() > 0.9:
            time_col = candidate
        else:
            raise ValueError(
                f"CSV missing time column (tried {_TIME_ALIASES}). Columns: {list(df.columns)}"
            )

    rename: dict[str, str] = {}
    for dest, aliases in _OHLCV.items():
        src = _pick(df.columns, aliases)
        if src is None:
            if dest == "volume":
                continue
            raise ValueError(
                f"CSV missing '{dest}' column. Columns: {list(df.columns)}"
            )
        rename[src] = dest

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col)

    df = df.rename(columns=rename)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if "volume" not in cols:
        df["volume"] = 0.0
        cols.append("volume")
    out = df[cols].astype(float).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if bars is not None:
        out = out.tail(bars)
    return out


def save_ohlcv_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "open_time"
    out.to_csv(path)
    return path
