"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import os
import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df

class QuintileFactorArbitrage(Strategy):
    """
    Cross-sectional value strategy with low-frequency rebalancing.

    Why this version:
    - P/E is not a meaningful 1-minute signal (EPS updates quarterly).
    - Rebalances once per period (default daily) using prior period close data.
    - Supports broader universes; warns/neutralizes if the universe is too small.
    """
    def __init__(
        self,
        eps_file: str = "EPS_data.csv",
        position_size: float = 10.0,
        rebalance_frequency: str = "D",
        long_quantile: float = 0.2,
        short_quantile: float = 0.8,
        min_universe: int = 8,
    ):
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if not 0 < long_quantile < short_quantile < 1:
            raise ValueError("Require 0 < long_quantile < short_quantile < 1.")
        if min_universe < 4:
            raise ValueError("min_universe must be at least 4.")

        self.position_size = float(position_size)
        self.rebalance_frequency = rebalance_frequency.upper()
        self.long_quantile = float(long_quantile)
        self.short_quantile = float(short_quantile)
        self.min_universe = int(min_universe)
        self.eps_lookup = self._load_eps_lookup(eps_file)

    def _load_eps_lookup(self, eps_file: str) -> dict[str, float]:
        strategies_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(strategies_dir)
        candidates = [
            os.path.join(root_dir, "data", eps_file),
            os.path.join(root_dir, eps_file),
        ]
        eps_path = next((path for path in candidates if os.path.exists(path)), None)
        if eps_path is None:
            raise FileNotFoundError(
                f"EPS file not found. Looked for {os.path.basename(candidates[0])} in data/ and repo root."
            )

        eps_df = pd.read_csv(eps_path)
        if eps_df.empty:
            raise ValueError(f"EPS file is empty: {eps_path}")

        normalized = {col.strip().lower(): col for col in eps_df.columns}
        if "ticker" not in normalized or "eps" not in normalized:
            raise ValueError("EPS file must contain 'Ticker' and 'EPS' columns.")

        ticker_col = normalized["ticker"]
        eps_col = normalized["eps"]
        eps_df = eps_df[[ticker_col, eps_col]].copy()
        eps_df.columns = ["Ticker", "EPS"]
        eps_df["Ticker"] = eps_df["Ticker"].astype(str).str.upper().str.strip()
        eps_df["EPS"] = pd.to_numeric(eps_df["EPS"], errors="coerce")
        eps_df = eps_df.dropna(subset=["Ticker", "EPS"])
        eps_df = eps_df[eps_df["EPS"] > 0]

        if eps_df.empty:
            raise ValueError("No valid positive EPS values found in EPS file.")
        return dict(zip(eps_df["Ticker"], eps_df["EPS"]))

    def _rebalance_bucket(self, dt: pd.Series) -> pd.Series:
        if self.rebalance_frequency in {"D", "1D"}:
            return dt.dt.floor("D")
        if self.rebalance_frequency in {"W", "1W"}:
            return dt.dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
        if self.rebalance_frequency in {"M", "1M"}:
            return dt.dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
        raise ValueError(f"Unsupported rebalance_frequency={self.rebalance_frequency}. Use D, W, or M.")

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"Datetime", "Ticker", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"QuintileFactorArbitrage requires columns: {', '.join(sorted(required))}.")

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["Datetime", "Close"])

        eps = df["Ticker"].map(self.eps_lookup)
        df["current_pe"] = df["Close"] / eps
        df["current_pe"] = df["current_pe"].replace([np.inf, -np.inf], np.nan)
        df["rebalance_ts"] = self._rebalance_bucket(df["Datetime"])
        return df

    def _build_rebalance_weights(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for rebalance_ts, period in snapshots.groupby("rebalance_ts", sort=True):
            period = period.dropna(subset=["current_pe"]).copy()
            period = period[period["current_pe"] > 0]
            tickers = period["Ticker"].unique().tolist()

            if len(tickers) < self.min_universe:
                for ticker in tickers:
                    rows.append({"rebalance_ts": rebalance_ts, "Ticker": ticker, "target_weight": 0.0})
                continue

            low_cut = period["current_pe"].quantile(self.long_quantile)
            high_cut = period["current_pe"].quantile(self.short_quantile)

            longs = period.loc[period["current_pe"] <= low_cut, "Ticker"].unique().tolist()
            shorts = period.loc[period["current_pe"] >= high_cut, "Ticker"].unique().tolist()

            long_w = (1.0 / len(longs)) if longs else 0.0
            short_w = (-1.0 / len(shorts)) if shorts else 0.0
            weights = {ticker: 0.0 for ticker in tickers}
            for ticker in longs:
                weights[ticker] = long_w
            for ticker in shorts:
                weights[ticker] = short_w

            for ticker, weight in weights.items():
                rows.append({"rebalance_ts": rebalance_ts, "Ticker": ticker, "target_weight": weight})

        weights_df = pd.DataFrame(rows)
        if weights_df.empty:
            return pd.DataFrame(columns=["rebalance_ts", "Ticker", "target_weight"])

        # Apply at next rebalance bucket to avoid lookahead bias.
        weights_df = weights_df.sort_values(["Ticker", "rebalance_ts"])
        weights_df["target_weight"] = weights_df.groupby("Ticker")["target_weight"].shift(1).fillna(0.0)
        return weights_df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["Datetime", "Ticker"]).copy()

        snapshots = (
            df.sort_values("Datetime")
            .groupby(["rebalance_ts", "Ticker"], as_index=False)
            .tail(1)
        )
        weights_df = self._build_rebalance_weights(snapshots)

        df = df.merge(weights_df, on=["rebalance_ts", "Ticker"], how="left")
        df["target_weight"] = df.groupby("Ticker")["target_weight"].ffill().fillna(0.0)

        # Signed target_qty supports long/short in the multi-asset backtester.
        df["target_qty"] = df["target_weight"] * self.position_size
        df["position"] = np.sign(df["target_qty"]).astype(int)

        # Emit signal only when desired target changes.
        delta_target = df.groupby("Ticker")["target_qty"].diff().fillna(df["target_qty"])
        df["signal"] = 0
        df.loc[delta_target > 0, "signal"] = 1
        df.loc[delta_target < 0, "signal"] = -1
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##
