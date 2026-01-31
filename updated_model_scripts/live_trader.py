import time
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import json
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from market_physics import MarketPhysics
from datetime import datetime

class LiveTrader:
    def __init__(self):
        with open("strategy_config.json", "r") as f:
            self.config = json.load(f)
        self.model = joblib.load("sota_brain.pkl") if os.path.exists("sota_brain.pkl") else None
        
    def get_live_physics(self):
        df = yf.download(self.config['SYMBOL'], period="5d", interval="1m", progress=False)
        if len(df) < 60: return None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        last_60 = df['Close'].values[-60:]
        
        hurst = MarketPhysics.get_hurst_exponent(last_60)
        theta, mu, _ = MarketPhysics.fit_ornstein_uhlenbeck(last_60)
        
        curr_price = last_60[-1]
        local_vol = np.std(last_60)
        z_score = (curr_price - mu) / local_vol if local_vol > 0 else 0
        
        vol_annual = df['Close'].pct_change().std() * np.sqrt(252*390)
        
        return {
            'price': curr_price, 'hurst': hurst, 'theta': theta,
            'z_score': z_score, 'vol': vol_annual, 'hour': datetime.now().hour
        }

    def predict(self, data):
        if not self.model: return 0.0
        feat = pd.DataFrame([[data['hurst'], data['theta'], data['z_score'], data['vol'], data['hour']]], 
                            columns=['hurst', 'theta', 'z_score', 'volatility', 'hour'])
        return self.model.predict_proba(feat)[0][1]

    def run(self):
        with Live(refresh_per_second=1) as live:
            while True:
                try:
                    data = self.get_live_physics()
                    if not data:
                        live.update(Panel("Waiting for data..."))
                        time.sleep(5)
                        continue
                        
                    prob = self.predict(data)
                    
                    table = Table(title=f"[bold cyan]{self.config['SYMBOL']} Physics Engine (SOTA)[/]")
                    table.add_column("Sensor")
                    table.add_column("Value")
                    table.add_column("State")
                    
                    # Hurst
                    h_lim = self.config['HURST_THRESHOLD']
                    h_state = "[green]MEAN REVERTING[/]" if data['hurst'] < h_lim else "[red]TRENDING[/]"
                    table.add_row("Regime (Hurst)", f"{data['hurst']:.3f}", h_state)
                    
                    # Theta
                    t_lim = self.config['THETA_THRESHOLD']
                    t_state = "[green]FAST[/]" if data['theta'] > t_lim else "[yellow]SLUGGISH[/]"
                    table.add_row("Reversion Speed", f"{data['theta']:.4f}", t_state)
                    
                    # Z-Score
                    z_color = "magenta" if abs(data['z_score']) > 2 else "white"
                    table.add_row("Deviation (Z)", f"[{z_color}]{data['z_score']:.2f}[/]", "Entry > 2.0")
                    
                    # AI
                    ai_state = "[bold green]HIGH CONFIDENCE[/]" if prob > 0.8 else "WAIT"
                    table.add_row("AI Confidence", f"{prob:.1%}", ai_state)
                    
                    live.update(Panel(table))
                    time.sleep(5)
                except Exception as e:
                    live.update(Panel(f"[red]Error: {e}[/]"))
                    time.sleep(5)

if __name__ == "__main__":
    lt = LiveTrader()
    lt.run()