import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from rich.console import Console
from data_manager import DataManager

console = Console()

class BacktestEngine:
    def __init__(self, df, config_override=None):
        self.df = df
        
        # Load default config from file
        try:
            with open("strategy_config.json", "r") as f:
                self.config = json.load(f)
        except:
            self.config = {
                "RISK_FREE_RATE": 0.045, 
                "HURST_THRESHOLD": 0.45,
                "THETA_THRESHOLD": 0.05, 
                "SIGMA_ENTRY": 2.0,
                "MAX_CONTRACTS": 10, 
                "INITIAL_SIZE": 1,
                "SCALE_IN_ATR": 1.25, 
                "STOP_LOSS_ATR": 2.0,
                "TARGET_EXPIRY_DAYS": 4,
                "SLIPPAGE_PER_CONTRACT": 0.0
            }
            
        # Apply Override (for Optimization)
        if config_override:
            self.config.update(config_override)
        
    def get_bs_price(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def run_simulation(self, silent=False):
        if not silent:
            console.print("[bold cyan]Running SOTA + Inventory Simulation...[/]")
        
        # Defensive Check
        required = ['atr', 'macd_z', 'rsi']
        if not all(col in self.df.columns for col in required):
            if not silent: console.print(f"[bold red]CRITICAL ERROR: Columns {required} missing.[/]")
            return 0.0 # Return 0 PnL on error

        self.df['vol_baseline'] = self.df['volatility'].rolling(window=1950).mean()

        trades = []
        HURST_MAX = self.config.get('HURST_THRESHOLD', 0.45)
        THETA_MIN = self.config.get('THETA_THRESHOLD', 0.05)
        SIGMA_ENTRY = self.config.get('SIGMA_ENTRY', 2.0)
        
        skipped_toxic = 0

        for i in range(len(self.df) - 240): 
            row = self.df.iloc[i]
            ts = row.name
            
            # --- 1. SOTA TIMING FILTERS ---
            is_event = row['volatility'] > (1.8 * row['vol_baseline'])
            is_friday_pm = (ts.dayofweek == 4 and ts.hour >= 12)
            is_late = (ts.hour >= 14)
            
            if is_event or is_friday_pm or is_late:
                skipped_toxic += 1
                continue

            # --- 2. PHYSICS FILTERS ---
            if row['hurst'] > HURST_MAX: continue
            if row['ou_theta'] < THETA_MIN: continue
                
            # --- 3. SIGNAL ---
            local_vol = self.df['Close'].iloc[i-30:i].std()
            if local_vol == 0: continue
            
            z_score = row['ou_divergence'] / local_vol
            
            direction = None
            if z_score < -SIGMA_ENTRY: direction = 'Call'
            elif z_score > SIGMA_ENTRY: direction = 'Put'
            
            if direction:
                pnl = self.simulate_campaign(i, direction)
                trades.append({
                    'hurst': row['hurst'],
                    'theta': row['ou_theta'],
                    'z_score': z_score,
                    'macd_z': row['macd_z'], 
                    'rsi': row['rsi'],      
                    'volatility': row['volatility'],
                    'atr': row['atr'],
                    'hour': ts.hour,
                    'day_of_week': ts.dayofweek,
                    'pnl': pnl,
                    'target': 1 if pnl > 0 else 0
                })

        if not trades:
            if not silent: console.print("[red]No trades generated.[/]")
            return 0.0

        res_df = pd.DataFrame(trades)
        
        # Only save CSV if not running silent optimization
        if not silent:
            res_df.to_csv("sota_training_data.csv", index=False)
        
        total_pnl = res_df['pnl'].sum()
        win_rate = res_df['target'].mean()
        
        if not silent:
            console.print(f"[green]Backtest Complete.[/]")
            console.print(f"Total P&L: [bold]${total_pnl:.2f}[/]")
            console.print(f"Win Rate: [bold]{win_rate:.1%}[/]")
            
        return total_pnl

    def simulate_campaign(self, idx, direction):
        entry_row = self.df.iloc[idx]
        strike = round(entry_row['Close'])
        target_days = self.config.get('TARGET_EXPIRY_DAYS', 4)
        t_start = target_days / 365
        rf = self.config.get('RISK_FREE_RATE', 0.045)
        bs_vol = entry_row['volatility']
        atr = entry_row['atr']
        
        MAX_CONTRACTS = self.config.get('MAX_CONTRACTS', 10)
        SCALE_STEP = self.config.get('SCALE_IN_ATR', 1.25) * atr
        STOP_DIST = self.config.get('STOP_LOSS_ATR', 2.0) * atr
        
        contracts = self.config.get('INITIAL_SIZE', 1)
        opt_type = direction.lower()
        
        entry_opt_price = self.get_bs_price(entry_row['Close'], strike, t_start, rf, bs_vol, opt_type)
        
        total_cost = (entry_opt_price * contracts * 100)
        avg_entry_stock = entry_row['Close']
        breakeven_triggered = False
        
        future = self.df.iloc[idx+1 : idx+240]
        
        for j, (ts, curr_row) in enumerate(future.iterrows()):
            t_curr = (target_days - ((j+1)/390)) / 365
            curr_opt_price = self.get_bs_price(curr_row['Close'], strike, t_curr, rf, bs_vol, opt_type)
            current_net_value = (curr_opt_price * contracts * 100)
            
            # --- SCALING ---
            moved_against = False
            if direction == 'Call' and curr_row['Close'] < (avg_entry_stock - SCALE_STEP): moved_against = True
            elif direction == 'Put' and curr_row['Close'] > (avg_entry_stock + SCALE_STEP): moved_against = True
            
            if moved_against and contracts < MAX_CONTRACTS:
                contracts += 1
                total_cost += (curr_opt_price * 100)
                avg_entry_stock = (avg_entry_stock * (contracts-1) + curr_row['Close']) / contracts
                
            # --- BREAKEVEN ---
            moved_favor = False
            if direction == 'Call' and curr_row['Close'] > (avg_entry_stock + (1.5 * atr)): moved_favor = True
            elif direction == 'Put' and curr_row['Close'] < (avg_entry_stock - (1.5 * atr)): moved_favor = True
            if moved_favor: breakeven_triggered = True

            # --- EXIT LOGIC ---
            curr_div = curr_row['Close'] - curr_row['ou_mu']
            reverted = (direction == 'Call' and curr_div >= 0) or (direction == 'Put' and curr_div <= 0)
            
            stop_hit = False
            if breakeven_triggered:
                stop_price = avg_entry_stock
                if direction == 'Call' and curr_row['Close'] < stop_price: stop_hit = True
                if direction == 'Put' and curr_row['Close'] > stop_price: stop_hit = True
            else:
                if direction == 'Call' and curr_row['Close'] < (avg_entry_stock - STOP_DIST): stop_hit = True
                if direction == 'Put' and curr_row['Close'] > (avg_entry_stock + STOP_DIST): stop_hit = True
            
            if reverted or stop_hit or curr_row['hurst'] > 0.6:
                return current_net_value - total_cost
                
            if ts.hour == 15 and ts.minute >= 55:
                return current_net_value - total_cost
                 
        return current_net_value - total_cost

if __name__ == "__main__":
    dm = DataManager()
    df = dm.fetch_polygon_data()
    bt = BacktestEngine(df)
    bt.run_simulation()