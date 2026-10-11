"""Bar-by-bar backtest with stops, take-profit, fees, and slippage."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from risk.manager import RiskConfig, RiskManager
from strategy.ai_advisor import AiAdvisor
from strategy.indicators import add_indicators
from strategy.params import StrategyConfig
from strategy.signals import generate_signal, rule_signal
from strategy.types import Side


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[dict]
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    funding_pnl: float = 0.0


@dataclass
class _Position:
    side: Side
    entry_price: float
    qty: float
    stop: float
    tp: float | None


@dataclass
class BacktestEngine:
    risk: RiskConfig
    initial_equity: float
    fee_bps: float = 10.0
    slippage_bps: float = 0.0
    use_ai: bool = False
    strategy: StrategyConfig | None = None

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        trade_start: int | None = None,
    ) -> BacktestResult:
        """Run bar-by-bar simulation.

        ``trade_start`` — bar index into ``df`` before which only indicators warm up
        (no entries). Defaults to strategy.warmup_bars.
        """
        strat = self.strategy or StrategyConfig()
        data = add_indicators(
            df,
            htf_rule=strat.htf_rule,
            htf_ema_span=strat.htf_ema_span,
            funding_lookback=strat.funding_lookback,
        )
        cash = self.initial_equity
        pos: _Position | None = None
        # Regime modes use BH-like exposure; a tight DD halt locks out recoveries
        risk_cfg = self.risk
        regime_mode = strat.mode in ("trend_ride", "funding_tilt")
        if regime_mode:
            risk_cfg = replace(self.risk, max_drawdown_halt=1.0)
        rm = RiskManager(risk_cfg, self.initial_equity)
        ai = AiAdvisor() if self.use_ai else None
        curve: list[float] = []
        curve_index: list = []
        trades: list[dict] = []
        funding_pnl_total = 0.0
        fee_rate = self.fee_bps / 10_000
        slip = self.slippage_bps / 10_000
        default_start = strat.warmup_bars
        start = default_start if trade_start is None else max(trade_start, 30)
        harvest = strat.mode == "funding_tilt"

        for i in range(start, len(data)):
            row = data.iloc[i]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else None
            signal = rule_signal(row, strategy=strat)
            if ai and ai.enabled and pos is None:
                signal = generate_signal(
                    data.iloc[: i + 1], ai=ai, symbol=symbol, strategy=strat
                )

            # Delta-neutral funding harvest: earn when funding > 0 (shorts receive)
            if (
                harvest
                and bool(row.get("funding_event", False))
                and "funding_rate" in data.columns
            ):
                rate = float(row["funding_rate"])
                notional = cash * strat.trend_position_fraction
                # Always-on hedged short-perp: PnL = +notional * rate
                # (positive funding => collect; negative => pay)
                payment = notional * rate
                cash += payment
                funding_pnl_total += payment

            # Directional funding payments when holding a position (non-harvest modes)
            if (
                not harvest
                and pos is not None
                and bool(row.get("funding_event", False))
                and "funding_rate" in data.columns
            ):
                rate = float(row["funding_rate"])
                payment = pos.qty * price * rate
                if pos.side == Side.LONG:
                    cash -= payment
                    funding_pnl_total -= payment
                else:
                    cash += payment
                    funding_pnl_total += payment

            if pos is not None and not harvest:
                exit_price = self._check_exit(pos, high, low)
                band = "|band" in (signal.reason or "")
                if (
                    exit_price is None
                    and strat.exit_on_signal_flip
                    and not band
                    and signal.side != pos.side
                ):
                    exit_price = price
                if exit_price is not None:
                    fill = self._slip_exit(pos.side, exit_price, slip)
                    cash, pnl = self._close_position(pos, fill, cash, fee_rate)
                    trades.append(
                        {
                            "entry": pos.entry_price,
                            "exit": fill,
                            "side": pos.side.value,
                            "pnl": pnl,
                        }
                    )
                    pos = None

            if pos is None and not harvest and not rm.halted and signal.side != Side.FLAT:
                equity = self._mark_equity(cash, pos, price)
                plan_kwargs = {}
                if regime_mode:
                    plan_kwargs = {
                        "position_fraction": strat.trend_position_fraction,
                        "atr_stop_mult": strat.trend_atr_stop_mult,
                        "disable_tp": True,
                    }
                plan = rm.plan_entry(
                    signal, equity, price, atr=atr_val, **plan_kwargs
                )
                if plan and plan.side != Side.FLAT:
                    fill = self._slip_entry(plan.side, price, slip)
                    plan = rm.plan_entry(
                        signal, equity, fill, atr=atr_val, **plan_kwargs
                    )
                    if plan is None:
                        continue
                    notional = plan.notional_usdt
                    fee = notional * fee_rate
                    qty = (notional - fee) / fill
                    if plan.side == Side.LONG:
                        cash -= notional
                    else:
                        cash += notional - fee
                    pos = _Position(
                        side=plan.side,
                        entry_price=fill,
                        qty=qty,
                        stop=float(plan.stop_price or fill),
                        tp=plan.take_profit_price,
                    )

            equity = self._mark_equity(cash, pos, price)
            rm.update_equity(equity)
            curve.append(equity)
            curve_index.append(data.index[i])

        series = pd.Series(curve, index=curve_index)
        result = self._summarize(series, trades)
        result.funding_pnl = funding_pnl_total
        if harvest:
            # Represent harvest settlements as trade count for reporting
            result.num_trades = int(
                data.iloc[start:]["funding_event"].sum()
            ) if "funding_event" in data.columns else 0
            wins = sum(1 for t in trades if t["pnl"] > 0)
            # win_rate unused for pure harvest; leave 0 unless directional trades exist
            if result.num_trades and funding_pnl_total:
                result.win_rate = 100.0 if funding_pnl_total > 0 else 0.0
        return result

    def _slip_entry(self, side: Side, price: float, slip: float) -> float:
        if side == Side.LONG:
            return price * (1 + slip)
        return price * (1 - slip)

    def _slip_exit(self, side: Side, price: float, slip: float) -> float:
        if side == Side.LONG:
            return price * (1 - slip)
        return price * (1 + slip)

    def _check_exit(self, pos: _Position, high: float, low: float) -> float | None:
        if pos.side == Side.LONG:
            if low <= pos.stop:
                return pos.stop
            if pos.tp is not None and high >= pos.tp:
                return pos.tp
        else:
            if high >= pos.stop:
                return pos.stop
            if pos.tp is not None and low <= pos.tp:
                return pos.tp
        return None

    def _close_position(
        self, pos: _Position, exit_price: float, cash: float, fee_rate: float
    ) -> tuple[float, float]:
        if pos.side == Side.LONG:
            proceeds = pos.qty * exit_price
            fee = proceeds * fee_rate
            pnl = pos.qty * (exit_price - pos.entry_price) - fee
            return cash + proceeds - fee, pnl
        cover_cost = pos.qty * exit_price
        fee = cover_cost * fee_rate
        pnl = pos.qty * (pos.entry_price - exit_price) - fee
        return cash - cover_cost - fee, pnl

    def _mark_equity(self, cash: float, pos: _Position | None, price: float) -> float:
        if pos is None:
            return cash
        if pos.side == Side.LONG:
            return cash + pos.qty * price
        # Short open credits proceeds into cash; equity is cash minus buyback liability.
        return cash - pos.qty * price

    def _summarize(self, equity: pd.Series, trades: list[dict]) -> BacktestResult:
        if equity.empty:
            return BacktestResult(
                equity, [], self.initial_equity, 0.0, 0.0, 0.0, 0
            )
        peak = equity.cummax()
        dd = (peak - equity) / peak
        wins = sum(1 for t in trades if t["pnl"] > 0)
        n = len(trades)
        final = float(equity.iloc[-1])
        ret = (final / self.initial_equity - 1) * 100
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            final_equity=final,
            total_return_pct=ret,
            max_drawdown_pct=float(dd.max() * 100) if len(dd) else 0.0,
            win_rate=(wins / n * 100) if n else 0.0,
            num_trades=n,
        )


def buy_and_hold_return_pct(
    df: pd.DataFrame,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
) -> float:
    """Long-only buy at first close, sell at last - after round-trip costs."""
    if len(df) < 2:
        return 0.0
    fee = fee_bps / 10_000
    slip = slippage_bps / 10_000
    entry = float(df["close"].iloc[0]) * (1 + slip)
    exit_ = float(df["close"].iloc[-1]) * (1 - slip)
    qty = (1.0 - fee) / entry
    proceeds = qty * exit_ * (1 - fee)
    return (proceeds - 1.0) * 100
