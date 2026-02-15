"""
Offline backtest runner for a CSV file.

Usage:
    python run_backtest.py --csv data/AAPL_1Min.csv --csv data/MSFT_1Min.csv --strategy RealTimePERatioStrategy --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# We keep these for the plotting utility, but we will use our own Backtester class
from core.backtester import plot_equity
from strategies import MovingAverageStrategy, TemplateStrategy, get_strategy_class

DATA_DIR = Path("data")

# --- CUSTOM MULTI-ASSET BACKTESTER ---
class MultiAssetBacktester:
    """
    A custom backtester that handles multiple tickers in a single stream.
    Correctly tracks positions and cash for each stock separately.
    """
    def __init__(self, strategy, initial_capital=50000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Tracks how many shares we hold of each ticker: {'AAPL': 10, 'TSLA': -5}
        self.positions = defaultdict(float)
        
        # Tracks the last seen price for each ticker: {'AAPL': 150.20}
        self.prices = {}
        
        self.trades = []
        self.equity_history = []

    def run(self, df):
        print(f"[INFO] Running Strategy on {len(df)} rows...")
        
        # 1. Run the Strategy logic to get 'target_qty' and 'signal' for every row
        # This assumes your strategy handles the 'Ticker' column correctly (like your PE strategy does)
        df = self.strategy.run(df)
        
        # 2. Sort by time to simulate real market flow across all stocks
        # We process minute-by-minute
        df = df.sort_values('Datetime')
        
        print("[INFO] Simulating execution...")
        
        # Iterate through the timeline
        for i, row in df.iterrows():
            ticker = row['Ticker']
            price = row['Close']
            target = row['target_qty']
            dt = row['Datetime']
            
            # Update the last known price for this ticker (for equity calc)
            self.prices[ticker] = price
            
            # --- EXECUTION LOGIC ---
            current_pos = self.positions[ticker]
            
            # If the strategy wants a different position than we have, trade the difference
            if current_pos != target:
                qty_to_trade = target - current_pos
                
                # Check for NaNs or Infinity
                if pd.isna(qty_to_trade) or np.isinf(qty_to_trade):
                    continue

                cost = qty_to_trade * price
                
                # Execute Trade (Simplified: No commission/slippage model yet)
                self.cash -= cost
                self.positions[ticker] = target
                
                # Log the trade
                action = "BUY" if qty_to_trade > 0 else "SELL"
                if abs(qty_to_trade) > 0:
                    self.trades.append({
                        'Datetime': dt,
                        'Ticker': ticker,
                        'Action': action,
                        'Qty': abs(qty_to_trade),
                        'Price': price,
                        'Value': -cost
                    })
            
            # --- MARK TO MARKET (Equity Calculation) ---
            # Value = Cash + Sum of (Shares * Current Price)
            holdings_value = sum(self.positions[t] * self.prices.get(t, 0) for t in self.positions)
            total_equity = self.cash + holdings_value
            self.equity_history.append(total_equity)
            
        # Return a DataFrame compatible with the plotter
        return pd.DataFrame({'equity': self.equity_history}, index=df['Datetime'])

# --- HELPER FUNCTIONS ---

def create_sample_data(path: Path, periods: int = 200) -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range(start="2024-01-01 09:30", periods=periods, freq="T"),
            "Open": np.random.uniform(100, 105, periods),
            "High": np.random.uniform(105, 110, periods),
            "Low": np.random.uniform(95, 100, periods),
            "Close": np.random.uniform(100, 110, periods),
            "Volume": np.random.randint(1_000, 5_000, periods),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline CSV backtest.")
    parser.add_argument("--csv", action="append", required=True, help="Path to CSV file(s).")
    parser.add_argument("--strategy", default="ma", help="Strategy name.")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade size.")
    parser.add_argument("--capital", type=float, default=50_000, help="Initial capital.")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1. LOAD AND ALIGN DATA ---
    all_dfs = []
    fixed_now = pd.Timestamp.now(tz='UTC').floor('min')
    
    print(f"[INFO] Processing {len(args.csv)} data files...")

    for path_str in args.csv:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
            
        ticker = path.name.split('_')[0].upper()
        df = pd.read_csv(path)
        
        # Standardize headers
        df.columns = [c.capitalize() if c.lower() == 'close' else c for c in df.columns]
        df.columns = [c.capitalize() if c.lower() in ['datetime', 'timestamp', 'time'] else c for c in df.columns]
        
        df['Ticker'] = ticker

        if '1970' in str(df['Datetime'].iloc[0]):
            df['Datetime'] = pd.date_range(end=fixed_now, periods=len(df), freq='min')
        
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        all_dfs.append(df)

    master_df = pd.concat(all_dfs).sort_values(['Datetime', 'Ticker']).reset_index(drop=True)

    # --- 2. INITIALIZE STRATEGY ---
    strategy_cls = get_strategy_class(args.strategy)
    try:
        strategy = strategy_cls(position_size=args.position_size)
    except TypeError:
        strategy = strategy_cls()

    # --- 3. RUN MULTI-ASSET BACKTESTER ---
    backtester = MultiAssetBacktester(strategy=strategy, initial_capital=args.capital)
    equity_df = backtester.run(master_df)
    
    # --- 4. PRINT SUMMARY RESULTS ---
    final_val = equity_df.iloc[-1]['equity']
    pnl = final_val - args.capital
    total_return = (pnl / args.capital) * 100
    
    # Calculate Sharpe
    returns = equity_df['equity'].pct_change().fillna(0)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 390) 
    else:
        sharpe = 0.0

    print("\n" + "="*40)
    print(f" FINAL RESULTS (10-Stock Basket)")
    print("="*40)
    print(f"Initial Capital:   ${args.capital:,.2f}")
    print(f"Final Value:       ${final_val:,.2f}")
    print(f"Net Profit (PnL):  ${pnl:,.2f} ({total_return:.2f}%)")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Total Trades:      {len(backtester.trades)}")
    print("="*40)

    # --- 5. DETAILED TRADE LOG (NEW) ---
    if backtester.trades:
        trades_df = pd.DataFrame(backtester.trades)
        
        # Reorder columns for readability
        cols = ['Datetime', 'Ticker', 'Action', 'Qty', 'Price', 'Value']
        trades_df = trades_df[cols]

        print("\n=== RECENT TRADES (Last 10) ===")
        # Print the last 10 trades to the console
        print(trades_df.tail(10).to_string(index=False))
        
        # Save ALL trades to a CSV file
        log_path = DATA_DIR / "trade_log_full.csv"
        trades_df.to_csv(log_path, index=False)
        print(f"\n[INFO] Full trade history saved to: {log_path.resolve()}")
        
    else:
        print("\n[WARN] No trades were executed! Check your strategy logic or EPS file.")

    if args.plot:
        plot_equity(equity_df)

if __name__ == "__main__":
    main()