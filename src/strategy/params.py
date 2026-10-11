"""Strategy knobs (filters / thresholds) separate from risk sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StrategyMode = Literal["pullback", "trend_ride", "funding_tilt"]


@dataclass
class StrategyConfig:
    """Entry / regime filters for rule signals."""

    mode: StrategyMode = "funding_tilt"
    min_score: float = 0.60
    long_only: bool = True
    require_htf_align: bool = True
    require_pullback: bool = True
    # Long: RSI must be at/below this; short: at/above rsi_bounce_min
    rsi_pullback_max: float = 38.0
    rsi_bounce_min: float = 62.0
    # Price must be on pullback side of Bollinger mid
    require_bb_pullback: bool = True
    # Same-TF EMA stack must agree with trade direction
    require_ema_stack: bool = True
    htf_rule: str = "4h"
    htf_ema_span: int = 50
    # Bars needed before trading (EMA/HTF warmup; funding_tilt can use ~48)
    warmup_bars: int = 220
    # Exit when signal leaves the position side
    exit_on_signal_flip: bool = True
    # Deploy this fraction while "in regime" (trend_ride / funding_tilt)
    trend_position_fraction: float = 0.95
    # Wide catastrophic stop; regime flip is the primary exit
    trend_atr_stop_mult: float = 8.0
    # funding_tilt: rolling window on forward-filled funding rate (bars)
    funding_lookback: int = 24
    # Long when rolling avg funding <= this (negative => longs get paid)
    funding_long_max: float = 0.0
    # Flat when rolling avg funding >= this (hysteresis if > long_max)
    funding_flat_min: float = 0.00005
