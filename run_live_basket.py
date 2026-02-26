"""
Live basket trader for cross-sectional strategies (e.g., quintile value).

Examples:
  python run_live_basket.py --dry-run --strategy quintile --timeframe 1Day --iterations 1
  python run_live_basket.py --live --strategy quintile --timeframe 1Day --sleep 86400
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from strategies import get_strategy_class, list_strategies


DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live/dry basket trading loop.")
    parser.add_argument("--strategy", default="quintile", help="Strategy name (default: quintile)")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-strategy position scale")
    parser.add_argument("--timeframe", default="1Day", help="Alpaca timeframe (default: 1Day)")
    parser.add_argument("--lookback", type=int, default=5, help="Bars fetched per symbol each iteration")
    parser.add_argument("--feed", default="iex", help="Stock data feed (default: iex)")
    parser.add_argument("--symbols-file", default="data/EPS_data.csv", help="CSV with at least 'Ticker' column")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations for non-live mode")
    parser.add_argument("--sleep", type=int, default=86400, help="Seconds between live iterations")
    parser.add_argument("--live", action="store_true", help="Run continuously and submit orders")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders without submitting")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies and exit")
    return parser.parse_args()


def get_alpaca_api() -> tradeapi.REST:
    load_dotenv()
    key_id = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_API_URL", DEFAULT_BASE_URL)
    if not key_id or not secret_key:
        raise ValueError("Missing ALPACA_API_KEY/ALPACA_API_SECRET in .env")
    return tradeapi.REST(key_id=key_id, secret_key=secret_key, base_url=base_url, api_version="v2")


def load_symbols(path: str) -> list[str]:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError(f"{path} must include a Ticker column")
    tickers = (
        df[cols["ticker"]]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    if not tickers:
        raise ValueError(f"No symbols found in {path}")
    return tickers


def build_strategy(name: str, position_size: float):
    strategy_cls = get_strategy_class(name)
    try:
        return strategy_cls(position_size=position_size)
    except TypeError:
        return strategy_cls()


def fetch_latest_basket_data(
    api: tradeapi.REST, symbols: list[str], timeframe: str, lookback: int, feed: str
) -> pd.DataFrame:
    rows = []
    end_ts = pd.Timestamp.now(tz="UTC")
    # Explicit window improves reliability with Alpaca data endpoints.
    start_ts = end_ts - pd.Timedelta(days=max(30, lookback * 5))
    for symbol in symbols:
        try:
            bars = api.get_bars(
                symbol,
                timeframe,
                start=start_ts.isoformat().replace("+00:00", "Z"),
                end=end_ts.isoformat().replace("+00:00", "Z"),
                limit=lookback,
                adjustment="raw",
                feed=feed,
            ).df
            if bars.empty:
                continue
            bars = bars.tail(lookback)
            for ts, row in bars.iterrows():
                rows.append(
                    {
                        "Datetime": ts,
                        "Ticker": symbol,
                        "Open": float(row["open"]),
                        "High": float(row["high"]),
                        "Low": float(row["low"]),
                        "Close": float(row["close"]),
                        "Volume": float(row["volume"]),
                    }
                )
        except Exception as exc:
            print(f"[WARN] {symbol}: fetch failed: {exc}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    return df.dropna(subset=["Datetime"]).sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


def _current_positions(api: tradeapi.REST) -> dict[str, int]:
    positions: dict[str, int] = {}
    for p in api.list_positions():
        qty = int(float(p.qty))
        if getattr(p, "side", "long") == "short":
            qty = -qty
        positions[p.symbol] = qty
    return positions


def rebalance_portfolio(
    api: tradeapi.REST, symbols: list[str], targets: dict[str, int], dry_run: bool
) -> None:
    current = _current_positions(api)
    print(f"\n--- REBALANCE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    for symbol in symbols:
        target_qty = int(targets.get(symbol, 0))
        current_qty = int(current.get(symbol, 0))
        diff = target_qty - current_qty
        if diff == 0:
            continue

        side = "buy" if diff > 0 else "sell"
        qty = abs(diff)
        print(f"{symbol}: {current_qty} -> {target_qty} | {side.upper()} {qty}")

        if dry_run:
            print("  [DRY RUN] no order submitted")
            continue

        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
        except Exception as exc:
            print(f"  [ERROR] order failed: {exc}")


def run_once(
    api: tradeapi.REST,
    strategy,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    feed: str,
    dry_run: bool,
) -> None:
    df = fetch_latest_basket_data(api, symbols, timeframe=timeframe, lookback=lookback, feed=feed)
    if df.empty:
        print("[WAIT] No market data returned.")
        return
    if df["Ticker"].nunique() < max(3, int(len(symbols) * 0.5)):
        print("[WAIT] Too few symbols returned this iteration; skipping rebalance.")
        return

    results = strategy.run(df)
    latest = results.sort_values("Datetime").groupby("Ticker", as_index=False).tail(1).copy()
    latest["target_qty"] = pd.to_numeric(latest.get("target_qty", 0), errors="coerce").fillna(0.0)
    targets = {row["Ticker"]: int(round(row["target_qty"])) for _, row in latest.iterrows()}
    non_zero = sum(1 for v in targets.values() if v != 0)
    print(
        f"[INFO] Strategy evaluated {df['Ticker'].nunique()} symbols over {df['Datetime'].nunique()} bars. "
        f"Non-zero targets: {non_zero}"
    )
    rebalance_portfolio(api, symbols=symbols, targets=targets, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    if args.list_strategies:
        print("Available strategies:")
        for s in list_strategies():
            print(f"  - {s}")
        return

    # Safety: live only when explicitly set and not overridden by dry-run.
    dry_run = args.dry_run or not args.live
    symbols = load_symbols(args.symbols_file)
    strategy = build_strategy(args.strategy, position_size=args.position_size)
    api = get_alpaca_api()

    mode = "LIVE" if not dry_run else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Strategy: {args.strategy}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Symbols: {len(symbols)} from {args.symbols_file}")

    if args.live and not args.dry_run:
        print(f"Running continuously. Sleep={args.sleep}s")
        try:
            while True:
                run_once(
                    api=api,
                    strategy=strategy,
                    symbols=symbols,
                    timeframe=args.timeframe,
                    lookback=args.lookback,
                    feed=args.feed,
                    dry_run=dry_run,
                )
                time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        for i in range(args.iterations):
            print(f"\nIteration {i + 1}/{args.iterations}")
            run_once(
                api=api,
                strategy=strategy,
                symbols=symbols,
                timeframe=args.timeframe,
                lookback=args.lookback,
                feed=args.feed,
                dry_run=dry_run,
            )
            if i < args.iterations - 1:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
