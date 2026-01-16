import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from scipy.stats import norm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()
console = Console()

# --- SETTINGS ---
SYMBOL = "AMD"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
LOOKBACK_DAYS = 480              # Set to 180 days for a more robust dataset
TARGET_EXPIRY_DAYS = 7
RISK_FREE_RATE = 0.045
CACHE_FILE = f"{SYMBOL}_historical_data_cache.csv"
CONFIG_FILE = "strategy_params.json"

# --- SEARCH SPACE ---
# Widened ranges to find more "outlier" successes
Z_CALL_RANGE = [2.0, 2.3, 2.5, 2.8, 3.0]
Z_PUT_RANGE = [2.3, 2.5, 2.8, 3.0, 3.2, 3.5]
VOL_RANGE = [1.2, 1.5, 1.8, 2.0, 2.2]

# --- COPYING SIMULATION LOGIC FROM BACKTEST.PY ---
def get_bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else \
           K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def simulate_campaign(df, start_idx):
    # Constants for simulation (Standardized for the search)
    ATR_MULT = 5.5
    SCALE_IN_STEP_ATR = 1.25
    MAX_CONTRACTS = 10
    INITIAL_SIZE = 1

    entry_row = df.iloc[start_idx]
    atr = entry_row['atr']
    z_score = entry_row['macd_z']
    
    opt_type = 'call' if z_score < 0 else 'put'
    direction = 1 if z_score < 0 else -1
    strike = round(entry_row['Close']) 
    t_start = TARGET_EXPIRY_DAYS / 365 
    vol = df['Close'].pct_change().rolling(30).std().iloc[start_idx] * np.sqrt(252 * 390)
    
    contracts = INITIAL_SIZE
    initial_opt_price = get_bs_price(entry_row['Close'], strike, t_start, RISK_FREE_RATE, vol, opt_type)
    total_cost = initial_opt_price * contracts * 100
    
    future = df.iloc[start_idx + 1 : start_idx + 240] 
    for i, (ts, curr_row) in enumerate(future.iterrows()):
        mins_passed = i + 1
        t_current = (TARGET_EXPIRY_DAYS - (mins_passed / 390)) / 365
        curr_opt_price = get_bs_price(curr_row['Close'], strike, t_current, RISK_FREE_RATE, vol, opt_type)
        
        # Scaling
        moved_against = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * SCALE_IN_STEP_ATR))) or \
                        (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * SCALE_IN_STEP_ATR)))
        
        if moved_against and contracts < MAX_CONTRACTS:
            contracts += 1
            total_cost += curr_opt_price * 100
            
        # Exit: Reversion
        if (direction == 1 and curr_row['macd_z'] >= 0) or (direction == -1 and curr_row['macd_z'] <= 0):
            return 1 if (curr_opt_price * contracts * 100) > total_cost else 0
            
        # Exit: Stop Loss
        stop_hit = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * ATR_MULT))) or \
                   (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * ATR_MULT)))
        if stop_hit: return 0 
            
        # Exit: EOD
        if ts.hour == 15 and ts.minute >= 55:
            return 1 if (curr_opt_price * contracts * 100) > total_cost else 0
    return 0

def run_optimization():
    # 1. Fetch or Load Data
    if os.path.exists(CACHE_FILE):
        console.print(f"[bold green]Local cache found![/] Loading {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE, index_col=0)
        # Fix: Explicitly convert index to DatetimeIndex with UTC awareness, then convert to Eastern
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    else:
        console.print(f"[bold yellow]No cache found.[/] Fetching {LOOKBACK_DAYS} days from Polygon API...")
        all_chunks = []
        end_date = datetime.now()
        start_date_limit = end_date - timedelta(days=LOOKBACK_DAYS)
        curr = end_date
        
        while curr > start_date_limit:
            start = curr - timedelta(days=5)
            url = (f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/minute/"
                   f"{start.strftime('%Y-%m-%d')}/{curr.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    chunk = pd.DataFrame(data["results"])
                    chunk['t'] = pd.to_datetime(chunk['t'], unit='ms')
                    all_chunks.append(chunk)
            
            curr = start
            time.sleep(12.1) # Respecting free tier rate limits

        if not all_chunks:
            console.print("[bold red]Error:[/] Failed to fetch data.")
            return

        df = pd.concat(all_chunks).sort_values('t').drop_duplicates('t')
        df.set_index('t', inplace=True)
        # Handle conversion from UTC to EST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')
            
        df.rename(columns={'c':'Close', 'h':'High', 'l':'Low', 'o':'Open', 'v':'Volume'}, inplace=True)
        
        # Save to cache
        df.to_csv(CACHE_FILE)
        console.print(f"[bold green]Data saved to {CACHE_FILE} for future use.[/]")

    # 2. Pre-calculate Indicators
    console.print("[yellow]Calculating indicators...")
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['macd_z'] = (macd - macd.rolling(30).mean()) / macd.rolling(30).std()
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()
    tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Hour and minute access now works correctly on the DatetimeIndex
    df['mins_open'] = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)

    results = []

    # 3. Iterate through Parameter Grid
    grid_size = len(Z_CALL_RANGE) * len(Z_PUT_RANGE) * len(VOL_RANGE)
    console.print(f"[bold cyan]Running Grid Search on {grid_size} combinations...")
    
    for z_call in Z_CALL_RANGE:
        for z_put in Z_PUT_RANGE:
            for vol_z in VOL_RANGE:
                # Find signals for this specific threshold combo
                sigs = df[
                    ((df['macd_z'] < -z_call) | (df['macd_z'] > z_put)) & 
                    (df['vol_z'] > vol_z) & 
                    (df['mins_open'] >= 15) & (df['mins_open'] < 240)
                ].index

                if len(sigs) < 20: continue # Higher minimum sample for reliability

                outcomes = []
                for idx in sigs:
                    loc = df.index.get_loc(idx)
                    if loc < len(df) - 241:
                        outcomes.append(simulate_campaign(df, loc))
                
                if outcomes:
                    win_rate = np.mean(outcomes)
                    wins = sum(outcomes)
                    losses = len(outcomes) - wins
                    profit_factor = wins / losses if losses > 0 else wins
                    
                    results.append({
                        'z_call': z_call,
                        'z_put': z_put,
                        'vol': vol_z,
                        'win_rate': win_rate,
                        'pf': profit_factor,
                        'count': len(outcomes),
                        'score': win_rate * profit_factor * np.log10(len(outcomes))
                    })

    # 4. Present Results
    if not results:
        console.print("[bold red]No successful combinations found.[/]")
        return
        
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)
    
    res_table = Table(title=f"Top 10 Optimized Thresholds for {SYMBOL}")
    res_table.add_column("Z Call", style="cyan")
    res_table.add_column("Z Put", style="cyan")
    res_table.add_column("Vol Z", style="cyan")
    res_table.add_column("Win Rate", style="green")
    res_table.add_column("Profit Factor", style="magenta")
    res_table.add_column("Sample Size", style="white")

    for _, row in results_df.head(10).iterrows():
        res_table.add_row(
            f"{row['z_call']}", f"{row['z_put']}", f"{row['vol']}", 
            f"{row['win_rate']:.1%}", f"{row['pf']:.2f}", f"{int(row['count'])}"
        )

    console.print(res_table)
    
    best = results_df.iloc[0]
    
    # --- SAVE TO CONFIG FILE ---
    config_data = {
        "SYMBOL": SYMBOL,
        "Z_THRESHOLD_CALL": float(best['z_call']),
        "Z_THRESHOLD_PUT": float(best['z_put']),
        "VOL_THRESHOLD": float(best['vol']),
        "OPTIMIZED_ON": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "EXPECTED_WIN_RATE": float(best['win_rate'])
    }
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)
        
    console.print(Panel(
        f"Recommended Configuration for Max Stability:\n\n"
        f"Z_THRESHOLD_CALL = {best['z_call']}\n"
        f"Z_THRESHOLD_PUT = {best['z_put']}\n"
        f"VOL_THRESHOLD = {best['vol']}\n\n"
        f"Expected Win Rate: {best['win_rate']:.1%}\n"
        f"Historical Sample: {int(best['count'])} trades\n\n"
        f"[bold green]Saved to {CONFIG_FILE}[/]",
        title="Optimization Success", border_style="green"
    ))

if __name__ == "__main__":
    run_optimization()