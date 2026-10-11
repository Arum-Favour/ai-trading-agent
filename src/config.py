"""Load settings from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from data.fetcher import Interval
from risk.manager import RiskConfig
from strategy.params import StrategyConfig, StrategyMode


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_mode(name: str, default: StrategyMode) -> StrategyMode:
    raw = (os.getenv(name) or default).strip().lower()
    if raw in ("pullback", "trend_ride", "funding_tilt"):
        return raw  # type: ignore[return-value]
    return default


@dataclass
class AppConfig:
    symbol: str
    interval: Interval
    lookback_bars: int
    risk: RiskConfig
    strategy: StrategyConfig
    initial_equity: float
    use_ai: bool


def load_config() -> AppConfig:
    load_dotenv()
    risk = RiskConfig(
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
        max_position_fraction=float(os.getenv("MAX_POSITION_FRACTION", "0.25")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.02")),
        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.04")),
        max_drawdown_halt=float(os.getenv("MAX_DRAWDOWN_HALT", "0.15")),
        use_atr_stops=_env_bool("USE_ATR_STOPS", True),
        atr_stop_mult=float(os.getenv("ATR_STOP_MULT", "1.5")),
        atr_tp_mult=float(os.getenv("ATR_TP_MULT", "3.0")),
        min_stop_pct=float(os.getenv("MIN_STOP_PCT", "0.005")),
        max_stop_pct=float(os.getenv("MAX_STOP_PCT", "0.06")),
    )
    mode = _env_mode("STRATEGY_MODE", "funding_tilt")
    warmup_default = "48" if mode == "funding_tilt" else "220"
    strategy = StrategyConfig(
        mode=mode,
        min_score=float(os.getenv("MIN_SIGNAL_SCORE", "0.60")),
        long_only=_env_bool("LONG_ONLY", True),
        require_htf_align=_env_bool("REQUIRE_HTF_ALIGN", True),
        require_pullback=_env_bool("REQUIRE_PULLBACK", True),
        rsi_pullback_max=float(os.getenv("RSI_PULLBACK_MAX", "38")),
        rsi_bounce_min=float(os.getenv("RSI_BOUNCE_MIN", "62")),
        require_bb_pullback=_env_bool("REQUIRE_BB_PULLBACK", True),
        require_ema_stack=_env_bool("REQUIRE_EMA_STACK", True),
        htf_rule=os.getenv("HTF_RULE", "4h"),
        htf_ema_span=int(os.getenv("HTF_EMA_SPAN", "50")),
        warmup_bars=int(os.getenv("WARMUP_BARS", warmup_default)),
        exit_on_signal_flip=_env_bool("EXIT_ON_SIGNAL_FLIP", True),
        trend_position_fraction=float(os.getenv("TREND_POSITION_FRACTION", "0.95")),
        trend_atr_stop_mult=float(os.getenv("TREND_ATR_STOP_MULT", "8.0")),
        funding_lookback=int(os.getenv("FUNDING_LOOKBACK", "24")),
        funding_long_max=float(os.getenv("FUNDING_LONG_MAX", "0.0")),
        funding_flat_min=float(os.getenv("FUNDING_FLAT_MIN", "0.00005")),
    )
    interval = os.getenv("INTERVAL", "1h")
    valid: tuple[Interval, ...] = (
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"
    )
    if interval not in valid:
        interval = "1h"
    return AppConfig(
        symbol=os.getenv("SYMBOL", "BTCUSDT"),
        interval=interval,  # type: ignore[arg-type]
        lookback_bars=int(os.getenv("LOOKBACK_BARS", "2000")),
        risk=risk,
        strategy=strategy,
        initial_equity=float(os.getenv("INITIAL_EQUITY", "10000")),
        use_ai=bool(os.getenv("OPENAI_API_KEY", "").strip()),
    )
