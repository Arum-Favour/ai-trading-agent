"""Combine rule-based signals with optional AI confidence adjustment."""

from __future__ import annotations

import pandas as pd

from strategy.ai_advisor import AiAdvisor, AiVerdict
from strategy.indicators import add_indicators
from strategy.params import StrategyConfig
from strategy.types import Side, Signal

_DEFAULT_STRATEGY = StrategyConfig()


def _warmup_ok(row: pd.Series) -> bool:
    if pd.isna(row.get("rsi")) or pd.isna(row.get("macd")):
        return False
    if pd.isna(row.get("htf_ema")) or pd.isna(row.get("atr")):
        return False
    if pd.isna(row.get("bb_mid")) or pd.isna(row.get("ema_trend")):
        return False
    return True


def trend_ride_signal(row: pd.Series, cfg: StrategyConfig) -> Signal:
    """Hypothesis: most crypto upside is captured by riding EMA200 regime.

    Hysteresis: stay long while price is above EMA200 (4h used as soft confirm
    only). Avoids HTF flicker exits that churn capital.
    """
    if not _warmup_ok(row):
        return Signal(Side.FLAT, 0.0, "warmup")

    htf_bull = bool(row["htf_bull"])
    above_ema = float(row["close"]) > float(row["ema_trend"])

    if cfg.long_only:
        if above_ema:
            conf = 0.75
            reasons = ["trend_ride", "above_ema200"]
            if htf_bull:
                conf += 0.10
                reasons.append("htf_bull")
            if row["ema_fast"] > row["ema_slow"]:
                conf += 0.08
                reasons.append("ema_fast_ok")
            return Signal(Side.LONG, min(conf, 1.0), "|".join(reasons))
        return Signal(Side.FLAT, 0.0, "trend_ride|below_ema200")

    # Two-sided: side of EMA200
    if above_ema:
        return Signal(Side.LONG, 0.85, "trend_ride|long")
    return Signal(Side.SHORT, 0.85, "trend_ride|short")


def pullback_signal(row: pd.Series, cfg: StrategyConfig) -> Signal:
    """Hypothesis: buy dips / sell rips inside the HTF trend (selective)."""
    if not _warmup_ok(row):
        return Signal(Side.FLAT, 0.0, "warmup")

    htf_bull = bool(row["htf_bull"])

    if cfg.long_only:
        if not htf_bull:
            return Signal(Side.FLAT, 0.0, "long_only_htf_bear")
        side = Side.LONG
        reasons: list[str] = ["long_only", "htf_bull"]
    else:
        side = Side.LONG if htf_bull else Side.SHORT
        reasons = ["htf_bull" if htf_bull else "htf_bear"]

    if cfg.require_ema_stack:
        if side == Side.LONG and row["close"] < row["ema_trend"]:
            return Signal(Side.FLAT, 0.0, "ema200_block_long")
        if side == Side.SHORT and row["close"] > row["ema_trend"]:
            return Signal(Side.FLAT, 0.0, "ema200_block_short")
        reasons.append("ema200_ok")

    if cfg.require_pullback:
        if side == Side.LONG:
            if row["rsi"] > cfg.rsi_pullback_max:
                return Signal(Side.FLAT, 0.0, f"rsi_not_low:{row['rsi']:.1f}")
            reasons.append(f"rsi<{cfg.rsi_pullback_max:.0f}")
            if cfg.require_bb_pullback and row["close"] > row["bb_mid"]:
                return Signal(Side.FLAT, 0.0, "need_below_bb_mid")
            if cfg.require_bb_pullback:
                reasons.append("below_bb_mid")
        else:
            if row["rsi"] < cfg.rsi_bounce_min:
                return Signal(Side.FLAT, 0.0, f"rsi_not_high:{row['rsi']:.1f}")
            reasons.append(f"rsi>{cfg.rsi_bounce_min:.0f}")
            if cfg.require_bb_pullback and row["close"] < row["bb_mid"]:
                return Signal(Side.FLAT, 0.0, "need_above_bb_mid")
            if cfg.require_bb_pullback:
                reasons.append("above_bb_mid")

    conf = 0.48
    if side == Side.LONG:
        conf += max(0.0, (cfg.rsi_pullback_max - float(row["rsi"])) / 100.0) * 1.5
        if row["close"] <= row["bb_lower"]:
            conf += 0.18
            reasons.append("at_bb_lower")
        elif row["close"] <= row["bb_mid"]:
            conf += 0.06
        if row["ema_fast"] > row["ema_slow"]:
            conf += 0.08
            reasons.append("ema_fast_ok")
        if row["macd"] > row["macd_signal"]:
            conf += 0.06
            reasons.append("macd_ok")
    else:
        conf += max(0.0, (float(row["rsi"]) - cfg.rsi_bounce_min) / 100.0) * 1.5
        if row["close"] >= row["bb_upper"]:
            conf += 0.18
            reasons.append("at_bb_upper")
        elif row["close"] >= row["bb_mid"]:
            conf += 0.06
        if row["ema_fast"] < row["ema_slow"]:
            conf += 0.08
            reasons.append("ema_fast_ok")
        if row["macd"] < row["macd_signal"]:
            conf += 0.06
            reasons.append("macd_ok")

    conf = min(conf, 1.0)
    if conf < cfg.min_score:
        return Signal(Side.FLAT, conf, f"low_conf:{'+'.join(reasons)}")

    if cfg.long_only and side != Side.LONG:
        return Signal(Side.FLAT, conf, "long_only_block_short")

    return Signal(side, conf, "+".join(reasons))


def funding_tilt_signal(row: pd.Series, cfg: StrategyConfig) -> Signal:
    """Placeholder side for funding_tilt — PnL comes from delta-neutral harvest.

    The backtest engine applies funding cashflows on settlement bars when this
    mode is active (approx short-perp + long-spot carry). Signal stays FLAT so
    we do not take directional price risk.
    """
    if pd.isna(row.get("funding_avg")):
        return Signal(Side.FLAT, 0.0, "warmup")
    avg = float(row["funding_avg"])
    return Signal(Side.FLAT, 0.0, f"funding_harvest|avg={avg:.6f}")


def rule_signal(
    row: pd.Series,
    strategy: StrategyConfig | None = None,
) -> Signal:
    cfg = strategy or _DEFAULT_STRATEGY
    if cfg.mode == "funding_tilt":
        return funding_tilt_signal(row, cfg)
    if cfg.mode == "trend_ride":
        return trend_ride_signal(row, cfg)
    return pullback_signal(row, cfg)


def generate_signal(
    df: pd.DataFrame,
    ai: AiAdvisor | None = None,
    symbol: str = "BTCUSDT",
    strategy: StrategyConfig | None = None,
) -> Signal:
    cfg = strategy or _DEFAULT_STRATEGY
    enriched = add_indicators(
        df,
        htf_rule=cfg.htf_rule,
        htf_ema_span=cfg.htf_ema_span,
        funding_lookback=cfg.funding_lookback,
    )
    row = enriched.iloc[-1]
    base = rule_signal(row, strategy=cfg)

    if ai is None or base.side == Side.FLAT:
        return base

    context = enriched.tail(48)
    verdict: AiVerdict = ai.review(symbol, base, context)
    if verdict.veto:
        return Signal(Side.FLAT, 0.0, f"ai_veto:{verdict.note}")

    adj_conf = max(0.0, min(1.0, base.confidence * verdict.confidence_multiplier))
    if adj_conf < 0.4:
        return Signal(Side.FLAT, adj_conf, f"ai_low_conf:{verdict.note}")
    return Signal(base.side, adj_conf, f"{base.reason}|ai:{verdict.note}")
