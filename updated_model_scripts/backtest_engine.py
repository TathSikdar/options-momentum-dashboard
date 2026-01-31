import pandas as pd
import numpy as np
import json
from rich.console import Console
from data_manager import DataManager
from market_physics import MarketPhysics

console = Console()

class BacktestEngine:
    def __init__(self, df, config_override=None):
        self.df = df
        
        try:
            with open("strategy_config.json", "r") as f:
                self.config = json.load(f)
        except:
            self.config = {
                "RISK_FREE_RATE": 0.045, 
                "HURST_THRESHOLD": 0.45,
                "THETA_THRESHOLD": 0.05, 
                "SIGMA_ENTRY": 2.0,
                "STOP_LOSS_ATR": 3.0,     # Widened for Scaling room
                "SCALE_IN_ATR": 1.0,      # Step size for scaling
                "MAX_CONTRACTS": 5,       # Cap scaling at 5 (Safety)
                "TARGET_EXPIRY_DAYS": 4,
                "INITIAL_SIZE": 1
            }
            
        if config_override:
            self.config.update(config_override)

    def run_simulation(self, silent=False):
        if not silent:
            console.print("[bold cyan]Running SOTA 'Smart Scaling' Simulation...[/]")
        
        required = ['atr', 'macd_z', 'rsi', 'dist_vwap', 'adx']
        if not all(col in self.df.columns for col in required):
            if not silent: console.print(f"[bold red]CRITICAL ERROR: Columns {required} missing.[/]")
            return 0.0

        # Pre-calcs
        self.df['vol_baseline'] = self.df['volatility'].rolling(window=1950).mean()
        if 'Volume' in self.df.columns:
            vol_mean = self.df['Volume'].rolling(window=20).mean()
            vol_std = self.df['Volume'].rolling(window=20).std()
            self.df['vol_z'] = (self.df['Volume'] - vol_mean) / vol_std
            self.df['vol_z'] = self.df['vol_z'].fillna(0)
        else:
            self.df['vol_z'] = 0.0

        trades = []
        HURST_MAX = self.config.get('HURST_THRESHOLD', 0.45)
        THETA_MIN = self.config.get('THETA_THRESHOLD', 0.05)
        SIGMA_ENTRY = self.config.get('SIGMA_ENTRY', 2.0)
        
        for i in range(len(self.df) - 240): 
            row = self.df.iloc[i]
            ts = row.name
            
            # --- 1. FILTERS ---
            is_event = row['volatility'] > (1.8 * row['vol_baseline'])
            is_friday_pm = (ts.dayofweek == 4 and ts.hour >= 12)
            is_late = (ts.hour >= 14) # No new campaigns after 2 PM
            
            if is_event or is_friday_pm or is_late: continue
            if row['hurst'] > HURST_MAX: continue
            if row['ou_theta'] < THETA_MIN: continue
                
            # --- 2. SIGNAL ---
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
                    'dist_vwap': row['dist_vwap'],
                    'adx': row['adx'],
                    'vol_z': row['vol_z'], 
                    'volatility': row['volatility'],
                    'atr': row['atr'],
                    'hour': ts.hour,
                    'pnl': pnl,
                    'target': 1 if pnl > 0 else 0
                })

        if not trades:
            if not silent: console.print("[red]No trades generated.[/]")
            return 0.0

        res_df = pd.DataFrame(trades)
        
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
        """
        Executes a trade using 'Regime-Gated Scaling'.
        Allows averaging down IF AND ONLY IF Physics says market is still reverting.
        """
        entry_row = self.df.iloc[idx]
        strike = round(entry_row['Close'])
        target_days = self.config.get('TARGET_EXPIRY_DAYS', 4)
        t_start = target_days / 365
        rf = self.config.get('RISK_FREE_RATE', 0.045)
        bs_vol = entry_row['volatility']
        atr = entry_row['atr']
        
        # Scaling Params
        MAX_CONTRACTS = self.config.get('MAX_CONTRACTS', 5)
        SCALE_STEP_DIST = self.config.get('SCALE_IN_ATR', 1.0) * atr
        STOP_DIST = self.config.get('STOP_LOSS_ATR', 3.0) * atr # Hard stop relative to AVG entry
        
        contracts = self.config.get('INITIAL_SIZE', 1)
        opt_type = direction.lower()
        
        entry_opt_price = MarketPhysics.get_bs_price(entry_row['Close'], strike, t_start, rf, bs_vol, opt_type)
        
        total_cost = (entry_opt_price * contracts * 100)
        avg_stock_entry = entry_row['Close']
        
        # Max Hold: 4 Hours (Give it time to revert)
        future = self.df.iloc[idx+1 : idx+240] 
        
        for j, (ts, curr_row) in enumerate(future.iterrows()):
            t_curr = (target_days - ((j+1)/390)) / 365
            curr_opt_price = MarketPhysics.get_bs_price(curr_row['Close'], strike, t_curr, rf, bs_vol, opt_type)
            current_net_value = (curr_opt_price * contracts * 100)
            
            # --- SCALING LOGIC (The "Old Model" Magic) ---
            moved_against = False
            if direction == 'Call' and curr_row['Close'] < (avg_stock_entry - SCALE_STEP_DIST): moved_against = True
            elif direction == 'Put' and curr_row['Close'] > (avg_stock_entry + SCALE_STEP_DIST): moved_against = True
            
            if moved_against and contracts < MAX_CONTRACTS:
                # SOTA GATE: Only scale if Hurst < 0.55 (Market is NOT trending)
                if curr_row['hurst'] < 0.55:
                    contracts += 1
                    total_cost += (curr_opt_price * 100)
                    # Update average entry price
                    avg_stock_entry = (avg_stock_entry * (contracts-1) + curr_row['Close']) / contracts
            
            # --- EXIT 1: Mean Reversion (Profit) ---
            curr_div = curr_row['Close'] - curr_row['ou_mu']
            reverted = (direction == 'Call' and curr_div >= 0) or (direction == 'Put' and curr_div <= 0)
            
            if reverted:
                # If profitable, exit. If scaling trapped us, hold until scratch or stop.
                if current_net_value > total_cost:
                    return current_net_value - total_cost

            # --- EXIT 2: Hard Stop (Relative to Average Entry) ---
            stop_hit = False
            if direction == 'Call' and curr_row['Close'] < (avg_stock_entry - STOP_DIST): stop_hit = True
            if direction == 'Put' and curr_row['Close'] > (avg_stock_entry + STOP_DIST): stop_hit = True
            if stop_hit: return current_net_value - total_cost
            
            # --- EXIT 3: Regime Break (The "New Model" Safety) ---
            # If Hurst spikes massively (> 0.65), the rubber band snapped. Get out.
            if curr_row['hurst'] > 0.65: return current_net_value - total_cost
            
            # --- EXIT 4: EOD ---
            if ts.hour == 15 and ts.minute >= 55: return current_net_value - total_cost

        return current_net_value - total_cost

if __name__ == "__main__":
    dm = DataManager()
    df = dm.fetch_polygon_data()
    bt = BacktestEngine(df)
    bt.run_simulation()