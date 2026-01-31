import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.console import Console
from market_physics import MarketPhysics

load_dotenv()
console = Console()

class DataManager:
    def __init__(self, symbol="AMD", api_key=None):
        self.symbol = symbol
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.cache_file = f"{symbol}_physics_data.csv"

    def fetch_polygon_data(self, days=180):
        if os.path.exists(self.cache_file):
            console.print(f"[green]Loading cached data: {self.cache_file}[/]")
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            return df

        console.print(f"[yellow]Fetching {days} days from Polygon...[/]")
        all_chunks = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        curr = end_date
        
        while curr > start_date:
            start_chunk = curr - timedelta(days=5)
            url = (f"https://api.polygon.io/v2/aggs/ticker/{self.symbol}/range/1/minute/"
                   f"{start_chunk.strftime('%Y-%m-%d')}/{curr.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            try:
                resp = requests.get(url).json()
                if "results" in resp:
                    chunk = pd.DataFrame(resp["results"])
                    chunk['t'] = pd.to_datetime(chunk['t'], unit='ms')
                    all_chunks.append(chunk)
            except Exception as e:
                console.print(f"[red]Error fetching: {e}[/]")
            
            curr = start_chunk
            time.sleep(12.5) # Rate limit safe

        if not all_chunks: return pd.DataFrame()
        
        df = pd.concat(all_chunks).sort_values('t').drop_duplicates('t')
        df.set_index('t', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        df.rename(columns={'c':'Close', 'h':'High', 'l':'Low', 'o':'Open', 'v':'Volume'}, inplace=True)
        
        # Pre-process Physics Features
        df = self.apply_physics_features(df)
        
        df.to_csv(self.cache_file)
        return df

    def apply_physics_features(self, df):
        console.print("[cyan]Calculating Physics Metrics (Hurst, Lambda, OU)... This may take a moment.[/]")
        
        # Rolling windows for physics
        # 60 minutes is standard for intraday regime detection
        window = 60
        
        # 1. Standard Indicators
        df['pct_change'] = df['Close'].pct_change()
        df['volatility'] = df['pct_change'].rolling(30).std() * np.sqrt(252 * 390)
        
        # 2. Physics Metrics (Rolling Apply)
        # Note: Rolling apply is slow, optimizing with simple iteration for clarity/robustness
        hursts = []
        thetas = []
        mus = []
        
        closes = df['Close'].values
        
        # Pre-fill initial NaN
        hursts = [0.5] * window
        thetas = [0.0] * window
        mus = [closes[0]] * window
        
        for i in range(window, len(df)):
            slice_data = closes[i-window : i]
            
            # Hurst
            h = MarketPhysics.get_hurst_exponent(slice_data)
            hursts.append(h)
            
            # OU Process
            theta, mu, _ = MarketPhysics.fit_ornstein_uhlenbeck(slice_data)
            thetas.append(theta)
            mus.append(mu)
            
        df['hurst'] = hursts
        df['ou_theta'] = thetas
        df['ou_mu'] = mus
        df['ou_divergence'] = df['Close'] - df['ou_mu'] # Positive = Overbought, Negative = Oversold
        
        return df.dropna()

if __name__ == "__main__":
    dm = DataManager()
    df = dm.fetch_polygon_data(days=60) # Test run
    print(df.tail())