import time
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from market_physics import MarketPhysics

# CONFIG
SYMBOL = "AMD"
MODEL_FILE = "sota_brain.pkl"

class LiveTrader:
    def __init__(self):
        self.model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
        self.status = "Initializing..."
        
    def get_live_physics(self):
        # We need last 60 candles for physics
        df = yf.download(SYMBOL, period="5d", interval="1m", progress=False)
        if len(df) < 60: return None
        
        # Strip MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        last_60 = df['Close'].values[-60:]
        
        hurst = MarketPhysics.get_hurst_exponent(last_60)
        theta, mu, sigma = MarketPhysics.fit_ornstein_uhlenbeck(last_60)
        
        current_price = last_60[-1]
        divergence = current_price - mu
        # Approximate local vol
        local_vol = np.std(last_60)
        z_score = divergence / local_vol if local_vol > 0 else 0
        
        vol_annual = df['Close'].pct_change().std() * np.sqrt(252*390)
        
        return {
            'price': current_price,
            'hurst': hurst,
            'theta': theta,
            'mu': mu,
            'z_score': z_score,
            'vol': vol_annual,
            'hour': datetime.now().hour
        }

    def predict_trade(self, physics):
        if not self.model: return 0.0
        
        # Match feature order from train_brain.py
        # ['hurst', 'theta', 'z_score', 'volatility', 'hour']
        features = pd.DataFrame([[
            physics['hurst'],
            physics['theta'],
            physics['z_score'],
            physics['vol'],
            physics['hour']
        ]], columns=['hurst', 'theta', 'z_score', 'volatility', 'hour'])
        
        return self.model.predict_proba(features)[0][1]

    def run(self):
        with Live(refresh_per_second=1) as live:
            while True:
                try:
                    data = self.get_live_physics()
                    if not data:
                        live.update(Panel("Waiting for data..."))
                        time.sleep(5)
                        continue
                        
                    prob = self.predict_trade(data)
                    
                    # Dashboard
                    table = Table(title=f"[bold cyan]{SYMBOL} Physics Engine[/]")
                    table.add_column("Metric")
                    table.add_column("Value")
                    table.add_column("Interpretation")
                    
                    # Hurst (Regime)
                    h_color = "green" if data['hurst'] < 0.45 else "red"
                    h_msg = "MEAN REVERTING" if data['hurst'] < 0.45 else "TRENDING/RANDOM"
                    table.add_row("Hurst Exponent", f"[{h_color}]{data['hurst']:.3f}[/]", h_msg)
                    
                    # Theta (Speed)
                    t_color = "green" if data['theta'] > 0.05 else "yellow"
                    table.add_row("OU Theta (Speed)", f"[{t_color}]{data['theta']:.4f}[/]", "High = Fast Snapback")
                    
                    # Divergence
                    div_color = "magenta" if abs(data['z_score']) > 2.0 else "white"
                    table.add_row("OU Z-Score", f"[{div_color}]{data['z_score']:.2f}[/]", "Deviation from Mu")
                    
                    # ML Brain
                    ml_color = "green" if prob > 0.75 else "white"
                    table.add_row("ML Confidence", f"[{ml_color}]{prob:.1%}[/]", "Prob of Win")
                    
                    live.update(Panel(table))
                    time.sleep(5)
                    
                except Exception as e:
                    live.update(Panel(f"[red]Error: {e}[/]"))
                    time.sleep(5)

from datetime import datetime
if __name__ == "__main__":
    lt = LiveTrader()
    lt.run()