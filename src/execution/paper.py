"""Paper trading loop — polls latest bar and logs decisions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import AppConfig
from data.fetcher import fetch_klines
from risk.manager import RiskManager
from strategy.ai_advisor import AiAdvisor
from strategy.indicators import add_indicators
from strategy.signals import generate_signal
from strategy.types import Side


@dataclass
class PaperState:
    cash: float
    side: Side = Side.FLAT
    entry_price: float = 0.0
    qty: float = 0.0
    stop: float = 0.0
    tp: float | None = None


def run_paper(config: AppConfig, journal_path: Path, poll_seconds: int = 300) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    state = PaperState(cash=config.initial_equity)
    rm = RiskManager(config.risk, config.initial_equity)
    ai = AiAdvisor() if config.use_ai else None
    bars = max(300, config.strategy.warmup_bars + 50)
    strat = config.strategy
    trend_mode = strat.mode == "trend_ride"

    print(f"Paper trading {config.symbol} ({config.interval}). Journal: {journal_path}")
    print(
        f"Mode: {strat.mode}, long_only={strat.long_only}, "
        f"HTF={strat.htf_rule} ema{strat.htf_ema_span}, "
        f"ATR stops={config.risk.use_atr_stops}"
    )
    print("Ctrl+C to stop.")

    while True:
        df = fetch_klines(config.symbol, config.interval, limit=min(bars, 1000))
        enriched = add_indicators(
            df,
            htf_rule=strat.htf_rule,
            htf_ema_span=strat.htf_ema_span,
        )
        row = enriched.iloc[-1]
        price = float(row["close"])
        atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else None
        signal = generate_signal(
            df, ai=ai, symbol=config.symbol, strategy=strat
        )

        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "mode": strat.mode,
            "signal": signal.side.value,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "atr": atr_val,
        }

        if state.side != Side.FLAT:
            exit_price = None
            if state.side == Side.LONG:
                if price <= state.stop:
                    exit_price = price
                elif state.tp is not None and price >= state.tp:
                    exit_price = price
            else:
                if price >= state.stop:
                    exit_price = price
                elif state.tp is not None and price <= state.tp:
                    exit_price = price
            if (
                exit_price is None
                and strat.exit_on_signal_flip
                and signal.side != state.side
            ):
                exit_price = price
            if exit_price is not None:
                if state.side == Side.LONG:
                    pnl = state.qty * (exit_price - state.entry_price)
                    state.cash += state.qty * exit_price
                else:
                    pnl = state.qty * (state.entry_price - exit_price)
                    state.cash -= state.qty * exit_price
                event["closed"] = {"pnl": pnl, "exit": exit_price}
                state = PaperState(cash=state.cash)

        if state.side == Side.FLAT:
            equity = state.cash
            rm.update_equity(equity)
            plan_kwargs = {}
            if trend_mode:
                plan_kwargs = {
                    "position_fraction": strat.trend_position_fraction,
                    "atr_stop_mult": strat.trend_atr_stop_mult,
                    "disable_tp": True,
                }
            plan = rm.plan_entry(signal, equity, price, atr=atr_val, **plan_kwargs)
            if plan:
                notional = plan.notional_usdt
                qty = notional / price
                cash_after = (
                    state.cash - notional
                    if plan.side == Side.LONG
                    else state.cash + notional
                )
                state = PaperState(
                    cash=cash_after,
                    side=plan.side,
                    entry_price=price,
                    qty=qty,
                    stop=float(plan.stop_price or price),
                    tp=plan.take_profit_price,
                )
                event["opened"] = asdict(plan)

        with journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

        print(json.dumps(event, indent=2, default=str))
        time.sleep(poll_seconds)
