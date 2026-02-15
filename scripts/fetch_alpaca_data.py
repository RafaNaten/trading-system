#!/usr/bin/env python3
"""
Standalone Alpaca data fetcher.

Usage examples:
  python scripts/fetch_alpaca_data.py --symbol AAPL --asset stock --timeframe 1Min --limit 1000
  python scripts/fetch_alpaca_data.py --symbol BTC/USD --asset crypto --timeframe 1Min --limit 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timezone
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DATA_FEED = "iex"
PLACEHOLDERS = {"PKB56UAGKUE2ADNYKHAUCGMPCY", "D8LToFobXqb3jJ1FoCBrGJECK3LVdGu5xtMzB6Z7E6jo", "PKB56UAGKUE2ADNYKHAUCGMPCY", "yD8LToFobXqb3jJ1FoCBrGJECK3LVdGu5xtMzB6Z7E6jo"}


def _load_env() -> None:
    env_paths = [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
    load_dotenv(override=False)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.lower() in PLACEHOLDERS or value.lower().startswith("your_"):
        raise RuntimeError(f"Missing valid {name}. Set it in .env or environment.")
    return value


def _parse_timeframe(timeframe: str):
    try:
        return tradeapi.TimeFrame(timeframe)
    except Exception:
        return timeframe


def _to_rfc3339(ts: pd.Timestamp) -> str:
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

    if isinstance(df.index, pd.MultiIndex):
        level0 = df.index.get_level_values(0)
        symbol_key = symbol.upper()
        if symbol in level0:
            df = df.xs(symbol, level=0)
        elif symbol_key in level0:
            df = df.xs(symbol_key, level=0)
        else:
            lower_map = {str(val).lower(): val for val in level0}
            match = lower_map.get(symbol.lower())
            if match is not None:
                df = df.xs(match, level=0)

    df = df.reset_index()
    rename_map = {}
    for col in df.columns:
        name = str(col).lower()
        if name in {"timestamp", "time", "t", "index", "datetime", "date"}:
            rename_map[col] = "Datetime"
        elif name in {"open", "o"}:
            rename_map[col] = "Open"
        elif name in {"high", "h"}:
            rename_map[col] = "High"
        elif name in {"low", "l"}:
            rename_map[col] = "Low"
        elif name in {"close", "c"}:
            rename_map[col] = "Close"
        elif name in {"volume", "v"}:
            rename_map[col] = "Volume"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Bars missing columns: {', '.join(missing)}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    return df[required]


def _build_client() -> tradeapi.REST:
    _load_env()
    api_key = _require_env("ALPACA_API_KEY")
    api_secret = _require_env("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_API_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    return tradeapi.REST(api_key, api_secret, base_url, api_version="v2")


def _fetch_stock(
    api: tradeapi.REST,
    symbol: str,
    timeframe: str,
    limit: int,
    feed: str,
    fallback_days: int,
) -> pd.DataFrame:
    tf = _parse_timeframe(timeframe)
    bars = api.get_bars(symbol, tf, limit=limit, feed=feed).df
    df = _normalize_bars(bars, symbol)
    if df.empty and fallback_days > 0:
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=fallback_days)
        bars = api.get_bars(
            symbol,
            tf,
            start=_to_rfc3339(start),
            end=_to_rfc3339(end),
            limit=limit,
            feed=feed,
        ).df
        df = _normalize_bars(bars, symbol)
    return df


def _fetch_crypto(
    api: tradeapi.REST,
    symbol: str,
    timeframe: str,
    limit: int,
    fallback_days: int,
) -> pd.DataFrame:
    if not hasattr(api, "get_crypto_bars"):
        raise RuntimeError("This alpaca_trade_api version does not support get_crypto_bars.")
    tf = _parse_timeframe(timeframe)
    bars = api.get_crypto_bars(symbol, tf, limit=limit).df
    df = _normalize_bars(bars, symbol)
    if df.empty and fallback_days > 0:
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=fallback_days)
        bars = api.get_crypto_bars(
            symbol,
            tf,
            start=_to_rfc3339(start),
            end=_to_rfc3339(end),
            limit=limit,
        ).df
        df = _normalize_bars(bars, symbol)
    return df


def _default_output(symbol: str, timeframe: str, asset: str) -> Path:
    safe_symbol = symbol.upper().replace("/", "")
    return REPO_ROOT / "data" / f"{safe_symbol}_{timeframe}_{asset}_alpaca_raw.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch candles from Alpaca and save CSV.")
    parser.add_argument("--symbol", default="AAPL", help="Asset symbol, e.g. AAPL or BTC/USD (default: AAPL)")
    parser.add_argument("--asset", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--timeframe", default="1Min", help="Examples: 1Min, 5Min, 1Hour, 1Day")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--feed", default=DEFAULT_DATA_FEED, help="Only used for stock (default: iex)")
    parser.add_argument("--fallback-days", type=int, default=10)
    parser.add_argument("--out", default=None, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser() if args.out else _default_output(args.symbol, args.timeframe, args.asset)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        api = _build_client()
        if args.asset == "stock":
            df = _fetch_stock(api, args.symbol, args.timeframe, args.limit, args.feed, args.fallback_days)
        else:
            df = _fetch_crypto(api, args.symbol, args.timeframe, args.limit, args.fallback_days)

        if df.empty:
            print(
                f"No bars returned for {args.symbol}. Try a larger timeframe (e.g. 1Day) "
                "or run during market hours for intraday stock bars.",
                file=sys.stderr,
            )
            return 2

        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
