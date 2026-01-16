import os
import time
import json
import asyncio
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.core.engine import get_bs_price, calculate_indicators
from app.schemas.profile import StrategyProfile

class ThresholdOptimizer:
    def __init__(self, profile: StrategyProfile, api_key: str):
        self.profile = profile
        self.api_key = api_key
        self.symbol = profile.symbol
        self.cache_dir = "backend/app/data/cache"
        self.cache_file = os.path.join(self.cache_dir, f"{self.symbol}_historical.csv")
        
        # Ensure directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Search Space (Directly from optimize_thresholds.py)
        self.z_call_range = [2.0, 2.3, 2.5, 2.8, 3.0]
        self.z_put_range = [2.3, 2.5, 2.8, 3.0, 3.2, 3.5]
        self.vol_range = [1.2, 1.5, 1.8, 2.0, 2.2]

    def _fetch_historical_data(self) -> pd.DataFrame:
        """Fetches data from Polygon with rate limiting."""
        if os.path.exists(self.cache_file):
            df = pd.read_csv(self.cache_file, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
            return df

        all_chunks = []
        end_date = datetime.now()
        start_date_limit = end_date - timedelta(days=self.profile.lookback_days)
        curr = end_date
        
        while curr > start_date_limit:
            start = curr - timedelta(days=5)
            url = (f"https://api.polygon.io/v2/aggs/ticker/{self.symbol}/range/1/minute/"
                   f"{start.strftime('%Y-%m-%d')}/{curr.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    chunk = pd.DataFrame(data["results"])
                    chunk['t'] = pd.to_datetime(chunk['t'], unit='ms')
                    all_chunks.append(chunk)
            
            curr = start
            # Polygon Free Tier: 5 calls per minute
            time.sleep(12.1) 

        if not all_chunks:
            return pd.DataFrame()

        df = pd.concat(all_chunks).sort_values('t').drop_duplicates('t')
        df.set_index('t', inplace=True)
        
        # Normalize Index
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')
            
        df.rename(columns={'c':'Close', 'h':'High', 'l':'Low', 'o':'Open', 'v':'Volume'}, inplace=True)
        df.to_csv(self.cache_file)
        return df

    def simulate_campaign(self, df: pd.DataFrame, start_idx: int) -> int:
        """Ported exactly from optimize_thresholds.py."""
        entry_row = df.iloc[start_idx]
        atr = entry_row['atr']
        z_score = entry_row['macd_z']
        
        opt_type = 'call' if z_score < 0 else 'put'
        direction = 1 if z_score < 0 else -1
        strike = round(entry_row['Close']) 
        t_start = self.profile.target_expiry_days / 365 
        vol = entry_row['implied_vol']
        
        contracts = self.profile.initial_size
        initial_opt_price = get_bs_price(entry_row['Close'], strike, t_start, self.profile.risk_free_rate, vol, opt_type)
        total_cost = initial_opt_price * contracts * 100
        
        # Look ahead window (4 hours)
        future = df.iloc[start_idx + 1 : start_idx + 240] 
        for i, (ts, curr_row) in enumerate(future.iterrows()):
            mins_passed = i + 1
            t_current = (self.profile.target_expiry_days - (mins_passed / 390)) / 365
            curr_opt_price = get_bs_price(curr_row['Close'], strike, t_current, self.profile.risk_free_rate, vol, opt_type)
            
            # Scaling
            moved_against = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * self.profile.scale_in_step_atr))) or \
                            (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * self.profile.scale_in_step_atr)))
            
            if moved_against and contracts < self.profile.max_contracts:
                contracts += 1
                total_cost += curr_opt_price * 100
                
            # Exit: Reversion
            if (direction == 1 and curr_row['macd_z'] >= 0) or (direction == -1 and curr_row['macd_z'] <= 0):
                return 1 if (curr_opt_price * contracts * 100) > total_cost else 0
                
            # Exit: Stop Loss
            stop_hit = (direction == 1 and curr_row['Close'] < (entry_row['Close'] - (atr * self.profile.atr_mult))) or \
                       (direction == -1 and curr_row['Close'] > (entry_row['Close'] + (atr * self.profile.atr_mult)))
            if stop_hit: return 0 
                
            # Exit: EOD
            if ts.hour == 15 and ts.minute >= 55:
                return 1 if (curr_opt_price * contracts * 100) > total_cost else 0
        return 0

    async def run_grid_search(self) -> List[Dict[str, Any]]:
        """Runs the optimization grid search."""
        df = self._fetch_historical_data()
        if df.empty:
            return []

        df = calculate_indicators(df)
        results = []

        for z_call in self.z_call_range:
            for z_put in self.z_put_range:
                for vol_z in self.vol_range:
                    # Filter signals
                    sigs = df[
                        ((df['macd_z'] < -z_call) | (df['macd_z'] > z_put)) & 
                        (df['vol_z'] > vol_z) & 
                        (df['mins_open'] >= self.profile.start_time_min) & 
                        (df['mins_open'] < self.profile.entry_cutoff_min)
                    ].index

                    if len(sigs) < 20: 
                        continue

                    outcomes = []
                    for idx in sigs:
                        loc = df.index.get_loc(idx)
                        if loc < len(df) - 241:
                            # We yield control to event loop occasionally
                            if len(outcomes) % 50 == 0:
                                await asyncio.sleep(0)
                            outcomes.append(self.simulate_campaign(df, loc))
                    
                    if outcomes:
                        win_rate = np.mean(outcomes)
                        wins = sum(outcomes)
                        losses = len(outcomes) - wins
                        pf = wins / losses if losses > 0 else wins
                        
                        results.append({
                            'z_threshold_call': float(z_call),
                            'z_threshold_put': float(z_put),
                            'vol_threshold': float(vol_z),
                            'win_rate': float(win_rate),
                            'profit_factor': float(pf),
                            'trade_count': len(outcomes),
                            'score': float(win_rate * pf * np.log10(len(outcomes)))
                        })

        return sorted(results, key=lambda x: x['score'], reverse=True)