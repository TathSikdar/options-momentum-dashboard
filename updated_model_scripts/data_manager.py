import pandas as pd
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.console import Console
from market_physics import MarketPhysics

load_dotenv()
console = Console()

class DataManager:
    def __init__(self):
        self.load_config()
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.cache_file = f"{self.config['SYMBOL']}_physics_data.csv"

    def load_config(self):
        default_config = {"SYMBOL": "AMD", "BACKTEST_WINDOW_DAYS": 180}
        if os.path.exists("strategy_config.json"):
            with open("strategy_config.json", "r") as f:
                self.config = json.load(f)
        else:
            self.config = default_config

    def fetch_polygon_data(self):
        days = self.config.get("BACKTEST_WINDOW_DAYS", 180)
        
        if os.path.exists(self.cache_file):
            console.print(f"[green]Loading cached data: {self.cache_file}[/]")
            try:
                df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                
                # CHECK FOR STALE CACHE (Missing new Confluence Metrics)
                required = ['atr', 'hurst', 'ou_theta', 'macd_z', 'rsi']
                missing = [c for c in required if c not in df.columns]
                
                if missing:
                    console.print(f"[yellow]Cache is missing columns {missing}. Regenerating features...[/]")
                    if {'Close', 'High', 'Low'}.issubset(df.columns):
                        df = self.apply_physics_features(df)
                        df.to_csv(self.cache_file)
                    else:
                        return self._force_fetch(days)
                
                cutoff = pd.Timestamp.now(tz='US/Eastern') - pd.Timedelta(days=days)
                df = df[df.index >= cutoff]
                return df
                
            except Exception as e:
                console.print(f"[red]Error reading cache: {e}. Re-fetching...[/]")
        
        return self._force_fetch(days)

    def _force_fetch(self, days):
        console.print(f"[yellow]Fetching {days} days from Polygon...[/]")
        all_chunks = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        curr = end_date
        
        while curr > start_date:
            start_chunk = curr - timedelta(days=5)
            url = (f"https://api.polygon.io/v2/aggs/ticker/{self.config['SYMBOL']}/range/1/minute/"
                   f"{start_chunk.strftime('%Y-%m-%d')}/{curr.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            try:
                resp = requests.get(url).json()
                if "results" in resp:
                    chunk = pd.DataFrame(resp["results"])
                    chunk['t'] = pd.to_datetime(chunk['t'], unit='ms')
                    all_chunks.append(chunk)
                time.sleep(12.1)
            except Exception as e:
                console.print(f"[red]Fetch Error: {e}[/]")
            
            curr = start_chunk

        if not all_chunks: return pd.DataFrame()
        
        df = pd.concat(all_chunks).sort_values('t').drop_duplicates('t')
        df.set_index('t', inplace=True)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')
        df.rename(columns={'c':'Close', 'h':'High', 'l':'Low', 'o':'Open', 'v':'Volume'}, inplace=True)
        
        df = self.apply_physics_features(df)
        df.to_csv(self.cache_file)
        return df

    def apply_physics_features(self, df):
        console.print("[cyan]Calculating Physics, ATR & Technicals...[/]")
        window = 60
        
        # 1. Volatility & ATR
        df['pct_change'] = df['Close'].pct_change()
        df['volatility'] = df['pct_change'].rolling(30).std() * np.sqrt(252 * 390)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # 2. Classic Technicals (MACD & RSI) for Confluence
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        # Z-Score of MACD to normalize it
        df['macd_z'] = (macd - macd.rolling(60).mean()) / macd.rolling(60).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 3. Physics (Hurst/OU)
        closes = df['Close'].values
        hursts, thetas, mus = [0.5]*window, [0.0]*window, [closes[0]]*window
        
        for i in range(window, len(df)):
            slice_data = closes[i-window : i]
            hursts.append(MarketPhysics.get_hurst_exponent(slice_data))
            theta, mu, _ = MarketPhysics.fit_ornstein_uhlenbeck(slice_data)
            thetas.append(theta)
            mus.append(mu)
            
        df['hurst'] = hursts
        df['ou_theta'] = thetas
        df['ou_mu'] = mus
        df['ou_divergence'] = df['Close'] - df['ou_mu']
        
        return df.dropna()

if __name__ == "__main__":
    dm = DataManager()
    dm.fetch_polygon_data()