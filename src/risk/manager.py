"""Position sizing and drawdown guardrails."""

from __future__ import annotations

from dataclasses import dataclass

from strategy.types import Side, Signal


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.25
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_halt: float = 0.15
    use_atr_stops: bool = True
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 3.0
    min_stop_pct: float = 0.005
    max_stop_pct: float = 0.06


@dataclass
class OrderPlan:
    side: Side
    notional_usdt: float
    stop_price: float | None
    take_profit_price: float | None


class RiskManager:
    def __init__(self, config: RiskConfig, initial_equity: float) -> None:
        self.config = config
        self.peak_equity = initial_equity
        self.halted = False

    def update_equity(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd >= self.config.max_drawdown_halt:
            self.halted = True

    def _stop_tp_distances(
        self,
        price: float,
        atr: float | None,
        atr_stop_mult: float | None = None,
        disable_tp: bool = False,
    ) -> tuple[float, float | None]:
        cfg = self.config
        stop_mult = atr_stop_mult if atr_stop_mult is not None else cfg.atr_stop_mult
        if cfg.use_atr_stops and atr is not None and atr > 0:
            stop_dist = atr * stop_mult
            tp_dist: float | None = None if disable_tp else atr * cfg.atr_tp_mult
        else:
            stop_dist = price * cfg.stop_loss_pct
            tp_dist = None if disable_tp else price * cfg.take_profit_pct

        min_dist = price * cfg.min_stop_pct
        max_dist = price * cfg.max_stop_pct
        stop_dist = min(max(stop_dist, min_dist), max_dist)
        if tp_dist is not None:
            rr = cfg.atr_tp_mult / stop_mult if stop_mult else 2.0
            if cfg.use_atr_stops and atr is not None and atr > 0:
                tp_dist = max(tp_dist, stop_dist * rr)
            else:
                tp_dist = max(tp_dist, stop_dist * (cfg.take_profit_pct / cfg.stop_loss_pct))
        return stop_dist, tp_dist

    def plan_entry(
        self,
        signal: Signal,
        equity: float,
        price: float,
        atr: float | None = None,
        position_fraction: float | None = None,
        atr_stop_mult: float | None = None,
        disable_tp: bool = False,
    ) -> OrderPlan | None:
        if self.halted or signal.side == Side.FLAT or price <= 0:
            return None

        stop_dist, tp_dist = self._stop_tp_distances(
            price, atr, atr_stop_mult=atr_stop_mult, disable_tp=disable_tp
        )
        if stop_dist <= 0:
            return None

        if position_fraction is not None:
            notional = equity * min(max(position_fraction, 0.0), 1.0)
        else:
            risk_budget = equity * self.config.risk_per_trade * signal.confidence
            size_from_risk = risk_budget / stop_dist * price
            max_notional = equity * self.config.max_position_fraction
            notional = min(size_from_risk, max_notional)
        if notional < 10:
            return None

        if signal.side == Side.LONG:
            stop = price - stop_dist
            tp = (price + tp_dist) if tp_dist is not None else None
        else:
            stop = price + stop_dist
            tp = (price - tp_dist) if tp_dist is not None else None

        return OrderPlan(signal.side, notional, stop, tp)
