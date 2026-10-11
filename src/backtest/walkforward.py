"""Walk-forward folds and fee/slippage stress matrix for rule signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, buy_and_hold_return_pct
from risk.manager import RiskConfig
from strategy.params import StrategyConfig


@dataclass
class FoldResult:
    fold: int
    start: str
    end: str
    bars: int
    strategy_return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    beat_buy_hold: bool


@dataclass
class StressCell:
    fee_bps: float
    slippage_bps: float
    total_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    buy_hold_return_pct: float


@dataclass
class ValidationReport:
    folds: list[FoldResult]
    stress: list[StressCell]
    full: BacktestResult
    full_buy_hold_pct: float
    oos_mean_return_pct: float
    oos_median_return_pct: float
    oos_positive_folds: int
    oos_folds_beat_bh: int


def _engine(
    risk: RiskConfig,
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
    use_ai: bool,
    strategy: StrategyConfig | None = None,
) -> BacktestEngine:
    return BacktestEngine(
        risk=risk,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        use_ai=use_ai,
        strategy=strategy,
    )


def walk_forward(
    df: pd.DataFrame,
    risk: RiskConfig,
    initial_equity: float,
    symbol: str = "BTCUSDT",
    n_folds: int = 5,
    lookback: int | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    use_ai: bool = False,
    strategy: StrategyConfig | None = None,
) -> list[FoldResult]:
    """Evaluate strategy on sequential out-of-sample windows.

    Each fold prepends ``lookback`` bars for indicator warmup; trading and
    metrics cover only the OOS segment. Equity resets each fold.
    """
    strat = strategy or StrategyConfig()
    lookback = lookback if lookback is not None else strat.warmup_bars
    n = len(df)
    min_bars = lookback + 40
    if n < min_bars * 2 or n_folds < 2:
        raise ValueError(
            f"Need enough bars for walk-forward (have {n}, want >= {min_bars * 2})."
        )

    # Reserve lookback from the front; split the remainder into n_folds OOS chunks
    tradeable = n - lookback
    fold_size = tradeable // n_folds
    if fold_size < 40:
        raise ValueError(
            f"Fold size too small ({fold_size} bars). Use fewer folds or more history."
        )

    folds: list[FoldResult] = []
    for k in range(n_folds):
        oos_start = lookback + k * fold_size
        oos_end = lookback + (k + 1) * fold_size if k < n_folds - 1 else n
        hist_start = max(0, oos_start - lookback)
        window = df.iloc[hist_start:oos_end]
        trade_start = oos_start - hist_start

        engine = _engine(risk, initial_equity, fee_bps, slippage_bps, use_ai, strat)
        result = engine.run(window, symbol=symbol, trade_start=trade_start)
        oos_slice = df.iloc[oos_start:oos_end]
        bh = buy_and_hold_return_pct(oos_slice, fee_bps, slippage_bps)

        start_ts = str(oos_slice.index[0])
        end_ts = str(oos_slice.index[-1])
        folds.append(
            FoldResult(
                fold=k + 1,
                start=start_ts,
                end=end_ts,
                bars=len(oos_slice),
                strategy_return_pct=result.total_return_pct,
                buy_hold_return_pct=bh,
                max_drawdown_pct=result.max_drawdown_pct,
                win_rate=result.win_rate,
                num_trades=result.num_trades,
                beat_buy_hold=result.total_return_pct > bh,
            )
        )
    return folds


def stress_matrix(
    df: pd.DataFrame,
    risk: RiskConfig,
    initial_equity: float,
    symbol: str = "BTCUSDT",
    fee_grid: tuple[float, ...] = (5.0, 10.0, 20.0),
    slip_grid: tuple[float, ...] = (0.0, 5.0, 15.0),
    use_ai: bool = False,
    strategy: StrategyConfig | None = None,
) -> list[StressCell]:
    """Full-sample backtests across fee x slippage scenarios."""
    cells: list[StressCell] = []
    for fee in fee_grid:
        for slip in slip_grid:
            engine = _engine(risk, initial_equity, fee, slip, use_ai, strategy)
            result = engine.run(df, symbol=symbol)
            warm = (strategy or StrategyConfig()).warmup_bars
            tradeable = df.iloc[warm:] if len(df) > warm else df
            bh = buy_and_hold_return_pct(tradeable, fee, slip)
            cells.append(
                StressCell(
                    fee_bps=fee,
                    slippage_bps=slip,
                    total_return_pct=result.total_return_pct,
                    max_drawdown_pct=result.max_drawdown_pct,
                    num_trades=result.num_trades,
                    win_rate=result.win_rate,
                    buy_hold_return_pct=bh,
                )
            )
    return cells


def run_validation(
    df: pd.DataFrame,
    risk: RiskConfig,
    initial_equity: float,
    symbol: str = "BTCUSDT",
    n_folds: int = 5,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    use_ai: bool = False,
    strategy: StrategyConfig | None = None,
) -> ValidationReport:
    strat = strategy or StrategyConfig()
    folds = walk_forward(
        df,
        risk=risk,
        initial_equity=initial_equity,
        symbol=symbol,
        n_folds=n_folds,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        use_ai=use_ai,
        strategy=strat,
    )
    stress = stress_matrix(
        df,
        risk=risk,
        initial_equity=initial_equity,
        symbol=symbol,
        use_ai=use_ai,
        strategy=strat,
    )
    engine = _engine(risk, initial_equity, fee_bps, slippage_bps, use_ai, strat)
    full = engine.run(df, symbol=symbol)
    # Align BH window with when the strategy actually starts trading
    tradeable = df.iloc[strat.warmup_bars :] if len(df) > strat.warmup_bars else df
    full_bh = buy_and_hold_return_pct(tradeable, fee_bps, slippage_bps)

    rets = [f.strategy_return_pct for f in folds]
    return ValidationReport(
        folds=folds,
        stress=stress,
        full=full,
        full_buy_hold_pct=full_bh,
        oos_mean_return_pct=float(sum(rets) / len(rets)),
        oos_median_return_pct=float(pd.Series(rets).median()),
        oos_positive_folds=sum(1 for r in rets if r > 0),
        oos_folds_beat_bh=sum(1 for f in folds if f.beat_buy_hold),
    )


def format_report(report: ValidationReport, fee_bps: float, slippage_bps: float) -> str:
    lines = [
        "",
        "=== Validation report ===",
        f"Baseline costs: fee={fee_bps:.0f} bps/side, slippage={slippage_bps:.0f} bps/side",
        "",
        "--- Full-sample (in-sample, optimistic) ---",
        f"Strategy return:  {report.full.total_return_pct:+.2f}%",
        f"Buy & hold:       {report.full_buy_hold_pct:+.2f}%",
        f"Max drawdown:     {report.full.max_drawdown_pct:.2f}%",
        f"Trades / win rate:{report.full.num_trades} / {report.full.win_rate:.1f}%",
        f"Funding PnL:      ${report.full.funding_pnl:+,.2f}",
        "",
        "--- Walk-forward OOS folds ---",
        f"{'Fold':>4}  {'Bars':>5}  {'Strat%':>8}  {'BH%':>8}  {'DD%':>7}  "
        f"{'Trades':>6}  {'Win%':>6}  BeatBH",
    ]
    for f in report.folds:
        lines.append(
            f"{f.fold:>4}  {f.bars:>5}  {f.strategy_return_pct:>+7.2f}%  "
            f"{f.buy_hold_return_pct:>+7.2f}%  {f.max_drawdown_pct:>6.2f}%  "
            f"{f.num_trades:>6}  {f.win_rate:>5.1f}%  "
            f"{'yes' if f.beat_buy_hold else 'no'}"
        )
    n = len(report.folds)
    lines.extend(
        [
            "",
            f"OOS mean return:     {report.oos_mean_return_pct:+.2f}%",
            f"OOS median return:   {report.oos_median_return_pct:+.2f}%",
            f"Positive OOS folds:  {report.oos_positive_folds}/{n}",
            f"Folds beat buy&hold: {report.oos_folds_beat_bh}/{n}",
            "",
            "--- Cost stress matrix (full sample) ---",
            f"{'Fee':>6}  {'Slip':>6}  {'Strat%':>8}  {'BH%':>8}  {'DD%':>7}  {'Trades':>6}",
        ]
    )
    for c in report.stress:
        lines.append(
            f"{c.fee_bps:>5.0f}  {c.slippage_bps:>5.0f}  "
            f"{c.total_return_pct:>+7.2f}%  {c.buy_hold_return_pct:>+7.2f}%  "
            f"{c.max_drawdown_pct:>6.2f}%  {c.num_trades:>6}"
        )

    # Verdict
    lines.append("")
    funding_sleeve = report.full.funding_pnl != 0 and report.full.num_trades > 20
    if funding_sleeve:
        carry_ok = (
            report.oos_positive_folds >= max(1, n // 2 + 1)
            and report.oos_median_return_pct > 0
            and report.full.total_return_pct > 0
            and report.full.max_drawdown_pct < 5.0
        )
        if carry_ok:
            lines.append(
                "Verdict: Funding carry sleeve looks tentatively usable "
                "(positive OOS yield, low DD). It is NOT meant to beat buy-and-hold "
                "in a bull market - size as a yield sleeve, paper-trade hedges first."
            )
        else:
            lines.append(
                "Verdict: Funding carry not reliable on this window. "
                "Do not deploy hedged short-perp live yet."
            )
    else:
        robust = (
            report.oos_positive_folds >= max(1, n // 2 + 1)
            and report.oos_folds_beat_bh >= max(1, n // 2)
            and report.oos_median_return_pct > 0
        )
        mid = next(
            (c for c in report.stress if c.fee_bps == 10 and c.slippage_bps == 5),
            report.stress[len(report.stress) // 2] if report.stress else None,
        )
        if mid and mid.total_return_pct < report.full_buy_hold_pct:
            robust = False

        if robust:
            lines.append(
                "Verdict: OOS results look tentatively usable - still paper-trade before live."
            )
        else:
            lines.append(
                "Verdict: Edge does NOT look robust under walk-forward / costs. "
                "Do not wire live execution yet; improve or replace the signal first."
            )
    return "\n".join(lines)
