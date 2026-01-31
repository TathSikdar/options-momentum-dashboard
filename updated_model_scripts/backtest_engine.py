import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from rich.console import Console
from data_manager import DataManager

console = Console()

# --- CONFIG ---
RISK_FREE_RATE = 0.045
TARGET_EXPIRY_DAYS = 7
INITIAL_CAPITAL = 10000

# SOTA THRESHOLDS (Can be optimized later, but these are research baselines)
HURST_THRESHOLD = 0.45    # Strict mean reversion requirement
THETA_THRESHOLD = 0.05    # Minimum reversion speed
SIGMA_ENTRY = 2.0         # Entry deviation (Z-score equivalent)

class BacktestEngine:
    def __init__(self, df):
        self.df = df
        self.results = []
        
    def get_bs_price(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def run_simulation(self):
        console.print("[bold cyan]Running SOTA Simulation (Hurst + OU Logic)...[/]")
        
        trades = []
        
        # Iterate through data
        # We need at least 60 mins of history, which DataManager already handles by dropping NaNs
        
        for i in range(len(self.df) - 240): # Stop before end of data
            row = self.df.iloc[i]
            
            # --- 1. REGIME FILTER (The Gatekeeper) ---
            # If Hurst > Threshold, market is becoming Random or Trending. DONT TRADE.
            if row['hurst'] > HURST_THRESHOLD:
                continue
                
            # --- 2. SPEED FILTER ---
            # If Theta is low, the rubber band is loose. Price won't snap back.
            if row['ou_theta'] < THETA_THRESHOLD:
                continue
                
            # --- 3. ENTRY SIGNAL (OU Divergence) ---
            # Calc dynamic Z-score based on recent volatility
            # Simple proxy: (Price - Mu) / (Price * Volatility)
            # A more robust way: use the pre-calculated ou_divergence
            
            # Dynamic volatility estimate for Z-calc
            local_vol = self.df['Close'].iloc[i-30:i].std()
            if local_vol == 0: continue
            
            z_score = row['ou_divergence'] / local_vol
            
            direction = None
            if z_score < -SIGMA_ENTRY: direction = 'Call'  # Price is way below Mu
            elif z_score > SIGMA_ENTRY: direction = 'Put'  # Price is way above Mu
            
            if direction:
                trade_res = self.simulate_trade(i, direction, local_vol)
                
                # Log Training Data
                trades.append({
                    'hurst': row['hurst'],
                    'theta': row['ou_theta'],
                    'z_score': z_score,
                    'volatility': row['volatility'],
                    'hour': row.name.hour,
                    'target': trade_res # 1 = Win, 0 = Loss
                })
        
        res_df = pd.DataFrame(trades)
        res_df.to_csv("sota_training_data.csv", index=False)
        
        win_rate = res_df['target'].mean()
        console.print(f"[green]Backtest Complete.[/] Generated {len(res_df)} samples. Win Rate: {win_rate:.1%}")

    def simulate_trade(self, entry_idx, direction, local_vol):
        entry_row = self.df.iloc[entry_idx]
        strike = round(entry_row['Close'])
        t_start = TARGET_EXPIRY_DAYS / 365
        bs_vol = entry_row['volatility']
        
        opt_type = direction.lower()
        entry_opt_price = self.get_bs_price(entry_row['Close'], strike, t_start, RISK_FREE_RATE, bs_vol, opt_type)
        
        # Max hold 4 hours
        future = self.df.iloc[entry_idx+1 : entry_idx+240]
        
        for j, (ts, curr_row) in enumerate(future.iterrows()):
            mins_passed = j + 1
            t_curr = (TARGET_EXPIRY_DAYS - (mins_passed/390)) / 365
            curr_opt_price = self.get_bs_price(curr_row['Close'], strike, t_curr, RISK_FREE_RATE, bs_vol, opt_type)
            
            # EXIT 1: Mean Reversion Accomplished
            # Using OU Mu from the CURRENT moment (dynamic target)
            curr_divergence = curr_row['Close'] - curr_row['ou_mu']
            
            if (direction == 'Call' and curr_divergence >= 0) or \
               (direction == 'Put' and curr_divergence <= 0):
                return 1 if curr_opt_price > entry_opt_price else 0
                
            # EXIT 2: Stop Loss (Physics Based)
            # If Hurst spikes > 0.6, the regime has broken into a trend against us. BAIL.
            if curr_row['hurst'] > 0.6:
                return 0
                
            # EXIT 3: Hard Stop (Price moves 3 sigma against us)
            if (direction == 'Call' and (entry_row['Close'] - curr_row['Close']) > 3*local_vol) or \
               (direction == 'Put' and (curr_row['Close'] - entry_row['Close']) > 3*local_vol):
                return 0
                
            # EXIT 4: EOD
            if ts.hour == 15 and ts.minute >= 55:
                 return 1 if curr_opt_price > entry_opt_price else 0
                 
        return 0

if __name__ == "__main__":
    dm = DataManager()
    df = dm.fetch_polygon_data(days=120)
    bt = BacktestEngine(df)
    bt.run_simulation()