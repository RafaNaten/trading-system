"""
Live Basket Trader for P/E Ratio Strategy.

Usage:
    python run_live_basket.py --live
    python run_live_basket.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import clean_market_data, save_bars
from strategies import RealTimePERatioStrategy, MovingAverageStrategy, TemplateStrategy, CryptoTrendStrategy, DemoStrategy, get_strategy_class, list_strategies
import time
import argparse
import os
import pandas as pd
from datetime import datetime, timezone
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv  # <--- NEW IMPORT


# --- CONFIGURATION ---
BASKET_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'TSLA', 'NVDA', 'AMD', 'AVGO', 'ADBE'
]
TIMEFRAME = "1Min"
# Paper Trading URL (Change to https://api.alpaca.markets for Real Money)
BASE_URL = "https://paper-api.alpaca.markets"

def get_alpaca_api():
    # 1. Load the .env file
    load_dotenv() 
    
    # 2. Retrieve keys from environment
    # We look for 'ALPACA_API_KEY' because that's what your .env likely uses
    key_id = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_API_SECRET')
    
    # 3. Safety Check
    if not key_id or not secret_key:
        raise ValueError("Critical Error: ALPACA_API_KEY or ALPACA_API_SECRET not found in .env file.")

    # 4. Connect
    return tradeapi.REST(
        key_id=key_id,
        secret_key=secret_key,
        base_url=BASE_URL, 
        api_version='v2'
    )

def fetch_latest_basket_data(api, symbols):
    """
    Fetches the most recent minute bar for all 10 stocks.
    Returns a DataFrame formatted exactly like your backtest Master CSV.
    """
    # Get last 10 minutes to ensure we have a valid recent bar for everyone
    # (Sometimes one stock is a few seconds slower to report)
    end_dt = pd.Timestamp.now(tz='UTC')
    start_dt = end_dt - pd.Timedelta(minutes=10)
    
    all_rows = []
    
    # Iterate symbols because get_bars for multi-symbol can be tricky with formatting
    # For a basket of 10, a loop is fast enough (ms).
    for symbol in symbols:
        try:
            bars = api.get_bars(symbol, TIMEFRAME, limit=5, adjustment='raw').df
            if not bars.empty:
                # Take the very last row (most recent completed minute)
                latest = bars.iloc[-1].copy()
                
                # Format into the shape our Strategy expects
                row = {
                    'Datetime': latest.name, # The index is usually the timestamp
                    'Ticker': symbol,
                    'Open': latest['open'],
                    'High': latest['high'],
                    'Low': latest['low'],
                    'Close': latest['close'],
                    'Volume': latest['volume']
                }
                all_rows.append(row)
        except Exception as e:
            print(f"[WARN] Could not fetch {symbol}: {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def rebalance_portfolio(api, targets, dry_run=False):
    """
    Compares current Alpaca positions with Target positions and executes trades.
    targets: dict { 'AAPL': 20, 'TSLA': -10 }
    """
    # 1. Get Current Positions
    current_positions = {}
    try:
        alpaca_positions = api.list_positions()
        for p in alpaca_positions:
            current_positions[p.symbol] = int(p.qty)
    except Exception as e:
        print(f"[ERROR] Could not fetch positions: {e}")
        return

    print(f"\n--- REBALANCING ({datetime.now().strftime('%H:%M:%S')}) ---")
    
    # 2. Calculate Diff and Trade
    for symbol in BASKET_SYMBOLS:
        target_qty = targets.get(symbol, 0)
        current_qty = current_positions.get(symbol, 0)
        
        diff = target_qty - current_qty
        
        if diff == 0:
            print(f"{symbol}: Matched ({current_qty})")
            continue
            
        action = "BUY" if diff > 0 else "SELL"
        qty = abs(diff)
        
        print(f"{symbol}: Have {current_qty} -> Want {target_qty} | Action: {action} {qty}")
        
        if not dry_run:
            try:
                # Alpaca requires 'sell' orders for long positions and 'buy' to cover shorts.
                # But the API simplifies this: 
                # If we want to SELL, we submit a sell order. 
                # If we are Short and want to Buy, we submit a buy order.
                
                side = 'buy' if diff > 0 else 'sell'
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"   >>> Order Sent: {side.upper()} {qty} {symbol}")
            except Exception as e:
                print(f"   >>> [ERROR] Order Failed: {e}")
        else:
            print(f"   >>> [DRY RUN] Would {action} {qty} {symbol}")

def main():
    parser = argparse.ArgumentParser()
    # We add both flags to be explicit. 
    # If --live is present, we trade. If --dry-run is present (or neither), we don't.
    parser.add_argument("--live", action="store_true", help="Execute real orders")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders (default)")
    parser.add_argument("--position-size", type=float, default=10.0)
    args = parser.parse_args()

    # Safety Logic: If user types BOTH or NEITHER, default to Dry Run.
    # We only go live if --live is True AND --dry-run is False.
    is_live_trading = args.live and not args.dry_run

    api = get_alpaca_api()
    
    # Initialize your Strategy
    # ensure EPS_data.csv is in the folder
    strategy = RealTimePERatioStrategy(position_size=args.position_size)
    
    print(f"Starting Basket Trader on {len(BASKET_SYMBOLS)} symbols.")
    print(f"Mode: {'LIVE TRADING (Real Money)' if is_live_trading else 'DRY RUN (Simulation)'}")
    
    try:
        while True:
            # 1. Fetch Data
            df = fetch_latest_basket_data(api, BASKET_SYMBOLS)
            
            if df.empty or len(df) < len(BASKET_SYMBOLS):
                print("[WAIT] Not all data available yet (Market might be closed)...")
                time.sleep(10)
                continue

            # 2. Run Strategy
            results = strategy.run(df)
            
            # 3. Extract Targets
            targets = dict(zip(results['Ticker'], results['target_qty']))
            
            # 4. Rebalance
            # Pass 'dry_run' as True if we are NOT live trading
            rebalance_portfolio(api, targets, dry_run=not is_live_trading)
            
            # 5. Sleep
            print("[SLEEP] Waiting 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n[STOP] Trader stopped by user.")

if __name__ == "__main__":
    main()