import pandas as pd
import numpy as np
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from data_manager import DataManager
from market_physics import MarketPhysics

console = Console()

class BacktestEngine:
    def __init__(self, df, config_override=None):
        self.df = df
        self.results = None # Store results for external access
        
        try:
            with open("strategy_config.json", "r") as f:
                self.config = json.load(f)
        except:
            self.config = {
                "RISK_FREE_RATE": 0.045, 
                "HURST_THRESHOLD": 0.45,
                "THETA_THRESHOLD": 0.05, 
                "SIGMA_ENTRY": 2.0,
                "STOP_LOSS_ATR": 3.0,
                "SCALE_IN_ATR": 1.0,
                "MAX_CONTRACTS": 5,
                "TARGET_EXPIRY_DAYS": 4, # Used as a fallback preference
                "INITIAL_SIZE": 1
            }
            
        if config_override:
            self.config.update(config_override)

    def run_simulation(self, silent=False):
        if not silent:
            console.print("[bold cyan]Running SOTA 'HMM Regime' Simulation (Numpy Accelerated)...[/]")
        
        required = ['atr', 'macd_z', 'rsi', 'dist_vwap', 'adx', 'regime']
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
        
        # Main Loop
        for i in range(len(self.df) - 240): 
            row = self.df.iloc[i]
            
            # --- FILTERS ---
            if row['regime'] == 2: continue # HMM Crash Filter
            if row['hurst'] > HURST_MAX: continue
            if row['ou_theta'] < THETA_MIN: continue
            
            # Event Filter (Vol Baseline)
            if row['volatility'] > (1.8 * row['vol_baseline']): continue
            
            # Time Filters
            ts = row.name
            if (ts.dayofweek == 4 and ts.hour >= 12) or (ts.hour >= 14): continue

            # --- SIGNAL ---
            local_vol = self.df['Close'].iloc[i-30:i].std()
            if local_vol == 0: continue
            
            z_score = row['ou_divergence'] / local_vol
            
            direction = None
            if z_score < -SIGMA_ENTRY: direction = 'Call'
            elif z_score > SIGMA_ENTRY: direction = 'Put'
            
            if direction:
                pnl = self.simulate_campaign(i, direction)
                trades.append({
                    'entry_time': ts,
                    'regime': row['regime'],
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
        self.results = res_df # Save for external access
        
        if not silent:
            res_df.to_csv("sota_training_data.csv", index=False)
            self.print_advanced_stats(res_df)
        
        return res_df['pnl'].sum()

    def print_advanced_stats(self, df):
        # 1. Basic P&L
        total_pnl = df['pnl'].sum()
        win_rate = df['target'].mean()
        
        # 2. Risk Metrics
        df['cum_pnl'] = df['pnl'].cumsum()
        df['peak'] = df['cum_pnl'].cummax()
        df['drawdown'] = df['cum_pnl'] - df['peak']
        max_drawdown = df['drawdown'].min()
        
        # 3. Trade Quality
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] <= 0]['pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
        
        # 4. SQN (System Quality Number)
        # SQN = (Expectancy / StdDev) * Sqrt(N)
        n_trades = len(df)
        if n_trades > 1:
            pnl_std = df['pnl'].std()
            avg_pnl = df['pnl'].mean()
            sqn = (avg_pnl / pnl_std) * np.sqrt(n_trades) if pnl_std != 0 else 0
        else:
            sqn = 0

        # Display
        table = Table(title="SOTA Strategy Performance Audit", border_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        table.add_column("Verdict", style="italic")

        table.add_row("Total P&L", f"${total_pnl:,.2f}", "[green]PROFITABLE[/]" if total_pnl > 0 else "[red]LOSS[/]")
        table.add_row("Win Rate", f"{win_rate:.1%}", "Target > 55%")
        table.add_row("Profit Factor", f"{profit_factor:.2f}", "[green]EXCELLENT[/]" if profit_factor > 1.5 else "NEEDS WORK")
        table.add_row("Max Drawdown", f"[red]${max_drawdown:,.2f}[/]", "Risk Exposure")
        table.add_row("Avg Win", f"${avg_win:.2f}", "")
        table.add_row("Avg Loss", f"[red]${avg_loss:.2f}[/]", "")
        table.add_row("Risk/Reward Ratio", f"{abs(avg_win/avg_loss):.2f}", "Target > 1.5")
        table.add_row("SQN Score", f"{sqn:.2f}", "Easier to trade > 2.0")
        table.add_row("Total Trades", f"{n_trades}", "Sample Size")

        console.print(table)

    def simulate_campaign(self, idx, direction):
        """
        Executes a trade using Numpy Optimization for speed.
        Forcefully simulates Wealthsimple constraints (Next Friday Expiry).
        """
        entry_row = self.df.iloc[idx]
        strike = round(entry_row['Close'])
        rf = self.config.get('RISK_FREE_RATE', 0.045)
        bs_vol = entry_row['volatility']
        atr = entry_row['atr']
        
        # --- WEALTHSIMPLE CONSTRAINT LOGIC ---
        # Calculate actual DTE to next Friday
        entry_date = entry_row.name
        # Weekday: Mon=0, Fri=4
        days_ahead = 4 - entry_date.weekday()
        if days_ahead <= 0: # If today is Friday or Weekend, push to next week
            days_ahead += 7
        
        # Actual Time to Expiry (in Years) for BS Model
        # We start with 'days_ahead' as our rigid contract duration
        actual_dte_days = days_ahead
        t_start = actual_dte_days / 365.0
        
        # Scaling Params
        MAX_CONTRACTS = self.config.get('MAX_CONTRACTS', 5)
        SCALE_STEP_DIST = self.config.get('SCALE_IN_ATR', 1.0) * atr
        STOP_DIST = self.config.get('STOP_LOSS_ATR', 3.0) * atr
        
        contracts = self.config.get('INITIAL_SIZE', 1)
        opt_type = direction.lower()
        
        entry_opt_price = MarketPhysics.get_bs_price(entry_row['Close'], strike, t_start, rf, bs_vol, opt_type)
        
        total_cost = (entry_opt_price * contracts * 100)
        avg_stock_entry = entry_row['Close']
        
        # --- NUMPY ACCELERATION ---
        # Extract the future window as a numpy array
        # Columns: [Close, hurst, ou_mu, regime]
        max_hold = 240
        future_slice = self.df.iloc[idx+1 : idx+max_hold]
        future_data = future_slice[['Close', 'hurst', 'ou_mu', 'regime']].values
        future_times = future_slice.index
        
        # Column Indices for fast access
        CLOSE, HURST, OU_MU, REGIME = 0, 1, 2, 3
        
        for j, (ts, row) in enumerate(zip(future_times, future_data)):
            curr_close = row[CLOSE]
            
            # Calculate Option Price based on DECAY of the FRIDAY contract
            # We are j+1 minutes into the trade
            # t_curr = (Original Days - (Minutes Elapsed / MinsInDay)) / 365
            t_curr = (actual_dte_days - ((j+1)/390.0)) / 365.0
            
            if t_curr <= 0: # Expiry Hit
                return (max(0, curr_close - strike) if opt_type == 'call' else max(0, strike - curr_close)) * contracts * 100 - total_cost

            curr_opt_price = MarketPhysics.get_bs_price(curr_close, strike, t_curr, rf, bs_vol, opt_type)
            current_net_value = (curr_opt_price * contracts * 100)
            
            # --- SCALING LOGIC ---
            moved_against = False
            if direction == 'Call' and curr_close < (avg_stock_entry - SCALE_STEP_DIST): moved_against = True
            elif direction == 'Put' and curr_close > (avg_stock_entry + SCALE_STEP_DIST): moved_against = True
            
            if moved_against and contracts < MAX_CONTRACTS:
                if row[REGIME] == 0 and row[HURST] < 0.55:
                    contracts += 1
                    total_cost += (curr_opt_price * 100)
                    avg_stock_entry = (avg_stock_entry * (contracts-1) + curr_close) / contracts
            
            # --- EXIT 1: Mean Reversion ---
            curr_div = curr_close - row[OU_MU]
            reverted = (direction == 'Call' and curr_div >= 0) or (direction == 'Put' and curr_div <= 0)
            if reverted:
                if current_net_value > total_cost:
                    return current_net_value - total_cost

            # --- EXIT 2: Hard Stop ---
            stop_hit = False
            if direction == 'Call' and curr_close < (avg_stock_entry - STOP_DIST): stop_hit = True
            if direction == 'Put' and curr_close > (avg_stock_entry + STOP_DIST): stop_hit = True
            if stop_hit: return current_net_value - total_cost
            
            # --- EXIT 3: Regime Break ---
            if row[REGIME] == 2 or row[HURST] > 0.65:
                return current_net_value - total_cost
            
            # --- EXIT 4: EOD ---
            if ts.hour == 15 and ts.minute >= 55: return current_net_value - total_cost

        return current_net_value - total_cost

if __name__ == "__main__":
    dm = DataManager()
    df = dm.fetch_polygon_data()
    bt = BacktestEngine(df)
    bt.run_simulation()