# AI Trading Agent

A **research and paper-trading** stack that combines rule-based signals (EMA, MACD, RSI, Bollinger) with an **optional** LLM risk reviewer. It is **not** a guarantee of profit — markets change, fees and slippage matter, and past backtests do not predict future results.

## What it does

| Mode | Command | Purpose |
|------|---------|---------|
| Fetch | `python src/main.py fetch` | Download OHLCV to `data/cache/` (uses Binance vision mirror) |
| Backtest | `python src/main.py backtest` | Simulate on Binance public OHLCV |
| Validate | `python src/main.py validate` | Walk-forward OOS + fee/slippage stress vs buy-and-hold |
| Paper | `python src/main.py paper` | Live prices, simulated fills, JSONL journal |

**Risk controls:** position sizing from `% risk per trade`, ATR (or fixed %) stop-loss / take-profit, max position size, halt after max drawdown.

**Strategy modes** (`STRATEGY_MODE` or `--mode`):

| Mode | Hypothesis |
|------|------------|
| `funding_tilt` (default) | Delta-neutral funding harvest approx (short perp hedged with spot): collect when funding > 0 |
| `trend_ride` | Ride EMA200 regimes with high exposure |
| `pullback` | Buy dips inside the HTF trend (selective RSI/BB) |

Funding history is pulled from Binance Vision archives (`fetch-funding`). Also: `LONG_ONLY=true` by default.

## Quick start

```bash
cd ai-trading-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # edit SYMBOL, INTERVAL, risk knobs
python src/main.py fetch --bars 3000
python src/main.py fetch-funding --days 400
python src/main.py validate --mode funding_tilt --csv data/cache/BTCUSDT_1h.csv
python src/main.py validate --mode trend_ride --csv data/cache/BTCUSDT_1h.csv
python src/main.py backtest --synthetic
```

If `api.binance.com` is blocked, the fetcher falls back to `data-api.binance.vision` and other mirrors, then to a local cache CSV.

Optional: set `OPENAI_API_KEY` in `.env` so the agent can veto or down-weight weak setups (extra API cost).

## Configuration

See `.env.example`. Defaults: `BTCUSDT`, `1h`, 2000 bars, $10k paper equity, 1% risk per trade, 2% stop / 4% target.

## Live trading

This repo ships **paper mode only**. Wiring real orders requires exchange API keys, compliance, and your own audit — e.g. OKX or another venue you trust. Do not enable live trading until backtests and paper runs look acceptable **to you**.

## Disclaimer

This software is for education. You can lose money trading. The authors make no promise of profitability.
