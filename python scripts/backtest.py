import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETTINGS & PATHS ---
SYMBOL = "AMD"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
RISK_FREE_RATE = 0.045
TARGET_EXPIRY_DAYS = 7
LOOKBACK_DAYS = 180
CACHE_FILE = f"{SYMBOL}_historical_data_cache.csv"
CONFIG_FILE = "strategy_params.json"

# --- DEFAULT PARAMETERS (Fallback if JSON is missing) ---
Z_THRESHOLD_CALL = 2.3
Z_THRESHOLD_PUT = 2.8
VOL_THRESHOLD = 1.8
START_TIME_MIN = 15          # 9:45 AM EST
ENTRY_CUTOFF_MIN = 270       # 2:00 PM EST
ATR_MULT = 5.5              
SCALE_IN_STEP_ATR = 1.25    
MAX_CONTRACTS = 10
INITIAL_SIZE = 1             

# --- DYNAMIC CONFIG LOADING ---
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Thresholds
            Z_THRESHOLD_CALL = config.get("Z_THRESHOLD_CALL", Z_THRESHOLD_CALL)
            Z_THRESHOLD_PUT = config.get("Z_THRESHOLD_PUT", Z_THRESHOLD_PUT)
            VOL_THRESHOLD = config.get("VOL_THRESHOLD", VOL_THRESHOLD)
            
            # Market Window
            START_TIME_MIN = config.get("START_TIME_MIN", START_TIME_MIN)
            ENTRY_CUTOFF_MIN = config.get("ENTRY_CUTOFF_MIN", ENTRY_CUTOFF_MIN)
            
            # Risk Logic
            ATR_MULT = config.get("ATR_MULT", ATR_MULT)
            SCALE_IN_STEP_ATR = config.get("SCALE_IN_STEP_ATR", SCALE_IN_STEP_ATR)
            
            print(f"\n[CONFIG] Loaded optimized parameters from {CONFIG_FILE}")
            print(f" > Call Z: {Z_THRESHOLD_CALL} | Put Z: {Z_THRESHOLD_PUT} | Vol Z: {VOL_THRESHOLD}")
            print(f" > Window: {START_TIME_MIN}m to {ENTRY_CUTOFF_MIN}m | ATR Mult: {ATR_MULT}\n")
    except Exception as e:
        print(f"[CONFIG] Error loading {CONFIG_FILE}: {e}. Using defaults.")

# --- 2. BLACK-SCHOLES PRICING ENGINE ---
def get_bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- 3. CAMPAIGN SIMULATOR (Updated for P&L) ---
def simulate_campaign(df, start_idx):
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
    
    # Look ahead 4 hours (240 mins)
    future = df.iloc[start_idx + 1 : start_idx + 240] 
    for i, (ts, curr_row) in enumerate(future.iterrows()):
        mins_passed = i + 1
        t_current = (TARGET_EXPIRY_DAYS - (mins_passed / 390)) / 365
        curr_opt_price = get_bs_price(curr_row['Close'], strike, t_current, RISK_FREE_RATE, vol, opt_type)
        
        # Scaling Logic: Buy 1 at a time when price moves against us
        moved_against = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * SCALE_IN_STEP_ATR))) or \
                        (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * SCALE_IN_STEP_ATR)))
        
        if moved_against and contracts < MAX_CONTRACTS:
            contracts += 1
            total_cost += curr_opt_price * 1 * 100
            
        current_value = curr_opt_price * contracts * 100
        pnl = current_value - total_cost

        # Exit Condition: Reversion to Mean (Z crosses 0)
        if (direction == 1 and curr_row['macd_z'] >= 0) or (direction == -1 and curr_row['macd_z'] <= 0):
            if current_value > (total_cost * 1.01): 
                return pnl
            elif i > 120: # If stuck for 2 hours, exit
                return pnl
            
        # Exit: Stop Loss
        stop_hit = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * ATR_MULT))) or \
                   (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * ATR_MULT)))
        if stop_hit: 
            return pnl
            
        # Exit: EOD (3:55 PM)
        if ts.hour == 15 and ts.minute >= 55:
            return pnl

    # If loop finishes without exit (rare), force close
    return (curr_opt_price * contracts * 100) - total_cost

# --- 4. DATA FETCHING & CACHING ---
def get_polygon_data(days_back=180):
    if os.path.exists(CACHE_FILE):
        print(f"Loading data from local cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
        return df
    
    print(f"No cache found. Fetching {days_back} days from Polygon...")
    all_chunks = []
    end_date = datetime.now()
    start_date_limit = end_date - timedelta(days=days_back)
    current_end = end_date
    
    while current_end > start_date_limit:
        current_start = current_end - timedelta(days=5)
        url = (f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/minute/"
               f"{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                chunk_df = pd.DataFrame(data["results"])
                chunk_df['timestamp'] = pd.to_datetime(chunk_df['t'], unit='ms')
                chunk_df.set_index('timestamp', inplace=True)
                chunk_df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                all_chunks.append(chunk_df)
        
        current_end = current_start
        time.sleep(12.1)

    if not all_chunks: return pd.DataFrame()
    
    full_df = pd.concat(all_chunks).sort_index().drop_duplicates()
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        full_df.index = full_df.index.tz_convert('US/Eastern')
        
    full_df.to_csv(CACHE_FILE)
    return full_df

# --- 5. MAIN PROCESSING ---
def run_backtest():
    print(f"Starting Backtest for {SYMBOL}...")
    df = get_polygon_data(LOOKBACK_DAYS)
    if df.empty:
        print("Error: No data available.")
        return
    
    # Indicators
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['macd_z'] = (macd - macd.rolling(30).mean()) / macd.rolling(30).std()
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()
    tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    df['mins_open'] = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    
    # --- SIGNAL LOGIC ---
    df['signal'] = 0
    df.loc[(df['macd_z'] < -Z_THRESHOLD_CALL) & (df['vol_z'] > VOL_THRESHOLD), 'signal'] = 1
    df.loc[(df['macd_z'] > Z_THRESHOLD_PUT) & (df['vol_z'] > VOL_THRESHOLD), 'signal'] = 1
    
    signals = df[(df['signal'] == 1) & 
                (df['mins_open'] >= START_TIME_MIN) & 
                (df['mins_open'] < ENTRY_CUTOFF_MIN)].index
    
    print(f"Found {len(signals)} signals matching optimized thresholds.")
    
    training_rows = []
    for idx in signals:
        loc = df.index.get_loc(idx)
        if loc > len(df) - 241: continue
        
        pnl = simulate_campaign(df, loc)
        vol = df['Close'].pct_change().rolling(30).std().iloc[loc] * np.sqrt(252 * 390)
        
        # We store 'pnl' for financial analysis AND 'target' for the ML classifier
        training_rows.append({
            'atr': df.at[idx,'atr'], 
            'mins_open': df.at[idx,'mins_open'], 
            'macd_z': df.at[idx,'macd_z'], 
            'vol_z': df.at[idx,'vol_z'], 
            'implied_vol': vol, 
            'pnl': round(pnl, 2),
            'target': 1 if pnl > 0 else 0
        })
        
    pd.DataFrame(training_rows).to_csv("historical_training_data.csv", index=False)
    print(f"Backtest Complete. Exported {len(training_rows)} campaigns to historical_training_data.csv")

if __name__ == "__main__":
    run_backtest()