"""AI trading agent CLI: backtest, validate, fetch, or paper trade."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python src/main.py` from ai-trading-agent/
_ROOT = Path(__file__).resolve().parent
_PROJECT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

from backtest.engine import BacktestEngine, buy_and_hold_return_pct  # noqa: E402
from backtest.walkforward import format_report, run_validation  # noqa: E402
from config import load_config  # noqa: E402
from data.csv_loader import load_ohlcv_csv, save_ohlcv_csv  # noqa: E402
from data.fetcher import fetch_klines_paginated  # noqa: E402
from data.funding import attach_funding, load_or_fetch_funding  # noqa: E402
from data.synthetic import generate_gbm_ohlcv  # noqa: E402
from execution.paper import run_paper  # noqa: E402


def _default_cache_path(symbol: str, interval: str) -> Path:
    return _PROJECT / "data" / "cache" / f"{symbol.upper()}_{interval}.csv"


def _prepare_df(config, df):
    """Attach funding rates when running funding_tilt."""
    if config.strategy.mode != "funding_tilt":
        return df
    if df.empty:
        return df
    try:
        funding = load_or_fetch_funding(
            config.symbol, df.index[0], df.index[-1]
        )
    except OSError as exc:
        print(f"Funding data unavailable ({exc}).")
        raise SystemExit(1) from exc
    merged = attach_funding(df, funding)
    events = int(merged["funding_event"].sum()) if "funding_event" in merged else 0
    print(
        f"Attached funding: {len(funding)} settlements, "
        f"{events} bars with funding events"
    )
    return merged


def _load_ohlcv(
    config,
    bars: int,
    csv_path: str | None,
    synthetic: bool,
    use_cache: bool = True,
    save_cache: bool = True,
):
    if csv_path:
        print(f"Loading CSV {csv_path}...")
        return load_ohlcv_csv(csv_path, bars=bars)
    if synthetic:
        print(f"Using synthetic GBM data ({bars} bars)...")
        return generate_gbm_ohlcv(bars=bars)

    cache = _default_cache_path(config.symbol, config.interval)
    print(f"Fetching {bars} bars for {config.symbol} ({config.interval})...")
    try:
        df = fetch_klines_paginated(config.symbol, config.interval, bars)
        if save_cache:
            saved = save_ohlcv_csv(df, cache)
            print(f"Cached OHLCV -> {saved}")
        return df
    except (OSError, ValueError, requests.RequestException) as exc:
        if use_cache and cache.is_file():
            print(f"Live fetch failed ({exc}). Falling back to cache {cache}...")
            return load_ohlcv_csv(cache, bars=bars)
        print(
            f"Live fetch failed ({exc}).\n"
            f"Re-run with --csv path\\to\\ohlcv.csv or --synthetic, "
            f"or fix network and retry."
        )
        raise SystemExit(1) from exc


def cmd_fetch(args: argparse.Namespace) -> None:
    config = load_config()
    bars = args.bars or max(config.lookback_bars, 3000)
    out = Path(args.out) if args.out else _default_cache_path(config.symbol, config.interval)
    print(f"Fetching {bars} bars for {config.symbol} ({config.interval})...")
    try:
        df = fetch_klines_paginated(config.symbol, config.interval, bars)
    except (OSError, ValueError) as exc:
        print(f"Fetch failed: {exc}")
        raise SystemExit(1) from exc
    path = save_ohlcv_csv(df, out)
    print(f"Wrote {len(df)} bars -> {path}")
    print(f"Range: {df.index[0]} .. {df.index[-1]}")


def cmd_fetch_funding(args: argparse.Namespace) -> None:
    config = load_config()
    import pandas as pd

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=args.days)
    try:
        df = load_or_fetch_funding(config.symbol, start, end, refresh=True)
    except OSError as exc:
        print(f"Funding fetch failed: {exc}")
        raise SystemExit(1) from exc
    print(f"Funding rows: {len(df)}  ({df.index[0]} .. {df.index[-1]})")
    print(f"Mean rate: {df['funding_rate'].mean():.6f}  "
          f"Negative share: {(df['funding_rate'] < 0).mean()*100:.1f}%")


def _apply_mode(config, mode: str | None):
    if mode:
        config.strategy.mode = mode  # type: ignore[assignment]
        if mode == "funding_tilt" and config.strategy.warmup_bars > 100:
            config.strategy.warmup_bars = 48
    return config


def cmd_backtest(args: argparse.Namespace) -> None:
    config = _apply_mode(load_config(), getattr(args, "mode", None))
    bars = args.bars or config.lookback_bars
    df = _prepare_df(config, _load_ohlcv(config, bars, args.csv, args.synthetic))
    engine = BacktestEngine(
        risk=config.risk,
        initial_equity=config.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        use_ai=config.use_ai and not args.no_ai,
        strategy=config.strategy,
    )
    result = engine.run(df, symbol=config.symbol)
    bh = buy_and_hold_return_pct(df, args.fee_bps, args.slippage_bps)
    print("\n--- Backtest summary ---")
    print(f"Mode:            {config.strategy.mode}")
    print(f"Bars:            {len(df)}  ({df.index[0]} .. {df.index[-1]})")
    print(f"Costs:           fee={args.fee_bps:.0f} bps, slip={args.slippage_bps:.0f} bps")
    print(f"Final equity:    ${result.final_equity:,.2f}")
    print(f"Total return:    {result.total_return_pct:+.2f}%")
    print(f"Buy & hold:      {bh:+.2f}%")
    print(f"Max drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"Trades:          {result.num_trades}")
    print(f"Win rate:        {result.win_rate:.1f}%")
    if result.num_trades == 0:
        print("\nNo trades - try a longer history or different INTERVAL in .env")


def cmd_validate(args: argparse.Namespace) -> None:
    config = _apply_mode(load_config(), getattr(args, "mode", None))
    bars = args.bars or max(config.lookback_bars, 3000)
    df = _prepare_df(config, _load_ohlcv(config, bars, args.csv, args.synthetic))

    print(
        f"Validating {config.symbol} [{config.strategy.mode}] on {len(df)} bars "
        f"({df.index[0]} .. {df.index[-1]})\n"
        f"folds={args.folds}, fee={args.fee_bps:.0f} bps, "
        f"slip={args.slippage_bps:.0f} bps..."
    )
    try:
        report = run_validation(
            df,
            risk=config.risk,
            initial_equity=config.initial_equity,
            symbol=config.symbol,
            n_folds=args.folds,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            use_ai=False,  # rules-only: LLM would burn API + leak non-determinism
            strategy=config.strategy,
        )
    except ValueError as exc:
        print(f"Validation failed: {exc}")
        raise SystemExit(1) from exc

    print(format_report(report, args.fee_bps, args.slippage_bps))


def cmd_paper(args: argparse.Namespace) -> None:
    config = load_config()
    journal = Path(args.journal)
    run_paper(config, journal, poll_seconds=args.poll)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bars", type=int, help="Override LOOKBACK_BARS")
    p.add_argument("--synthetic", action="store_true", help="Offline random-walk OHLCV")
    p.add_argument(
        "--csv",
        help="OHLCV CSV (open_time/timestamp + open,high,low,close[,volume])",
    )
    p.add_argument(
        "--mode",
        choices=("funding_tilt", "trend_ride", "pullback"),
        help="Override STRATEGY_MODE for this run",
    )


def _add_cost_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fee-bps",
        type=float,
        default=10.0,
        help="Trading fee in basis points per side (default 10)",
    )
    p.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Adverse slippage in basis points per fill (default 5)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule + optional LLM crypto trading agent "
        "(backtest / validate / fetch / paper)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run historical simulation")
    _add_data_args(bt)
    _add_cost_args(bt)
    bt.add_argument("--no-ai", action="store_true", help="Disable LLM even if API key set")
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser(
        "validate",
        help="Walk-forward OOS folds + fee/slippage stress vs buy-and-hold",
    )
    _add_data_args(val)
    _add_cost_args(val)
    val.add_argument("--folds", type=int, default=5, help="Number of OOS folds (default 5)")
    val.set_defaults(func=cmd_validate)

    fe = sub.add_parser("fetch", help="Download OHLCV and save CSV for offline validate")
    fe.add_argument("--bars", type=int, help="Bars to download (default max(LOOKBACK, 3000))")
    fe.add_argument(
        "--out",
        help="Output CSV path (default data/cache/SYMBOL_INTERVAL.csv)",
    )
    fe.set_defaults(func=cmd_fetch)

    ff = sub.add_parser(
        "fetch-funding",
        help="Download Binance Vision perpetual funding history",
    )
    ff.add_argument("--days", type=int, default=400, help="Lookback days (default 400)")
    ff.set_defaults(func=cmd_fetch_funding)

    pp = sub.add_parser("paper", help="Poll market and paper-trade")
    pp.add_argument(
        "--journal",
        default="logs/paper_journal.jsonl",
        help="Path to JSONL log file",
    )
    pp.add_argument("--poll", type=int, default=300, help="Seconds between polls")
    pp.set_defaults(func=cmd_paper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
