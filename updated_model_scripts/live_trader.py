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
from rich.console import Console
from market_physics import MarketPhysics
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner dashboard
warnings.filterwarnings('ignore')
console = Console()

class LiveTrader:
    def __init__(self):
        # Load Config
        try:
            with open("strategy_config.json", "r") as f:
                self.config = json.load(f)
        except:
            self.config = {
                "SYMBOL": "AMD", 
                "THETA_THRESHOLD": 0.05,
                "HURST_THRESHOLD": 0.5,
                "SIGMA_ENTRY": 2.0,
                "MAX_CONTRACTS": 5,
                "TARGET_EXPIRY_DAYS": 4,
                "RISK_FREE_RATE": 0.045
            }

        # Load Brain
        if os.path.exists("sota_brain.pkl"):
            self.model = joblib.load("sota_brain.pkl")
            console.print("[green]SOTA Brain loaded successfully.[/]")
        else:
            self.model = None
            console.print("[bold red]WARNING: 'sota_brain.pkl' not found. AI predictions disabled.[/]")
            
        # EXACT feature list from train_brain.py
        self.features = ['regime', 'hurst', 'theta', 'z_score', 'macd_z', 'rsi', 
                         'dist_vwap', 'adx', 'vol_z', 'volatility', 'atr', 'hour']
                         
        # Virtual Inventory State
        self.inventory = 0
        self.avg_price = 0.0
        self.stop_loss = 0.0

    def get_live_data(self):
        """
        Fetches recent 1-minute data to calculate rolling indicators.
        We need ~5 days (approx 2000 bars) to ensure HMM and EMAs stabilize.
        """
        try:
            # Fetch 5 days of 1m data
            df = yf.download(self.config['SYMBOL'], period="5d", interval="1m", progress=False)
            
            if len(df) < 200: 
                return None
            
            # Handle YFinance MultiIndex (if present)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Rename columns to match DataManager expectations
            df.rename(columns={'Close': 'Close', 'High': 'High', 'Low': 'Low', 'Open': 'Open', 'Volume': 'Volume'}, inplace=True)
            
            return df
        except Exception as e:
            return None

    def calculate_live_features(self, df):
        """
        Replicates the feature engineering from DataManager for a single live snapshot.
        """
        # 1. Physics (HMM Regimes) - Apply to whole window
        df = MarketPhysics.get_market_regime(df)
        
        # 2. Volatility & ATR
        df['pct_change'] = df['Close'].pct_change()
        df['volatility'] = df['pct_change'].rolling(30).std() * np.sqrt(252 * 390)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # 3. Technicals (MACD, RSI)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        # Z-score normalization for MACD (to make it stationary for ML)
        df['macd_z'] = (macd - macd.rolling(60).mean()) / macd.rolling(60).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 4. Advanced (VWAP, ADX)
        df['vwap'] = MarketPhysics.calculate_vwap(df)
        df['dist_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
        df['adx'] = MarketPhysics.calculate_adx(df)
        
        # 5. Volume Z-Score
        vol_mean = df['Volume'].rolling(20).mean()
        vol_std = df['Volume'].rolling(20).std()
        df['vol_z'] = (df['Volume'] - vol_mean) / vol_std
        
        # 6. Hurst & OU (Calculate for the most recent window)
        window = 60
        last_slice = df['Close'].values[-window:]
        
        hurst = MarketPhysics.get_hurst_exponent(last_slice)
        theta, mu, _ = MarketPhysics.fit_ornstein_uhlenbeck(last_slice)
        
        # OU Z-Score (Mean Reversion Signal Strength)
        local_vol = np.std(last_slice)
        z_score = (last_slice[-1] - mu) / local_vol if local_vol > 0 else 0
        
        # --- Construct Result Row ---
        # We take the values from the very last candle
        last_idx = df.index[-1]
        
        result = df.iloc[-1].copy()
        result['hurst'] = hurst
        result['theta'] = theta
        result['z_score'] = z_score
        result['hour'] = last_idx.hour
        
        # Fill NaNs (important for ML safety)
        result = result.fillna(0)
        
        return result

    def get_preview_data(self, current_price, volatility, z_score):
        """
        Calculates the live preview contract (Strike, Expiry, Price) for the dashboard.
        Prices the option using Black-Scholes.
        """
        # 1. Determine Direction (Preview Only)
        # If Z-Score is positive -> Revert Down -> PUT
        # If Z-Score is negative -> Revert Up -> CALL
        direction = "PUT" if z_score > 0 else "CALL"
        opt_type = direction.lower()
        
        # 2. Wealthsimple Friday Logic
        target_days = self.config.get('TARGET_EXPIRY_DAYS', 4)
        future_date = datetime.now() + timedelta(days=target_days)
        days_ahead = 4 - future_date.weekday()
        expiry_date = future_date + timedelta(days=days_ahead)
        
        if expiry_date.date() <= datetime.now().date():
            expiry_date += timedelta(days=7)
            
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        
        # Calculate Time to Expiry (Years)
        dte_days = (expiry_date - datetime.now()).days
        t_years = dte_days / 365.0
        
        # 3. Strike (ATM)
        strike = round(current_price)
        
        # 4. Black-Scholes Pricing
        rf = self.config.get('RISK_FREE_RATE', 0.045)
        # Assuming volatility is annual, if it came from calculate_live_features it is already annualized
        bs_price = MarketPhysics.get_bs_price(current_price, strike, t_years, rf, volatility, opt_type)
        
        return {
            "type": direction,
            "strike": strike,
            "expiry": expiry_str,
            "price": bs_price
        }

    def run(self):
        console.print(Panel("[bold yellow]Initializing Live Physics Engine...[/]", border_style="yellow"))
        
        with Live(refresh_per_second=4) as live:
            while True:
                try:
                    # 1. Fetch & Process
                    df = self.get_live_data()
                    if df is None:
                        live.update(Panel("[yellow]Waiting for data stream...[/]"))
                        time.sleep(2)
                        continue
                    
                    latest = self.calculate_live_features(df)
                    current_price = latest['Close']
                    volatility = latest['volatility']
                    z_val = latest['z_score']
                    last_updated = datetime.now().strftime("%H:%M:%S")
                    
                    # 2. AI Inference
                    prob = 0.0
                    ai_msg = "[dim]OFFLINE[/]"
                    
                    if self.model:
                        input_vector = pd.DataFrame([latest[self.features].values], columns=self.features)
                        prob = self.model.predict_proba(input_vector)[0][1]
                        
                        if prob >= 0.80: ai_msg = "[bold green]HIGH CONFIDENCE BUY[/]"
                        elif prob >= 0.60: ai_msg = "[green]MODERATE BUY[/]"
                        elif prob <= 0.20: ai_msg = "[red]AVOID / SHORT[/]"
                        else: ai_msg = "[yellow]WAITING[/]"

                    # 3. Build Option Preview
                    preview = self.get_preview_data(current_price, volatility, z_val)
                    type_color = "red" if preview['type'] == "PUT" else "green"
                    type_str = f"[bold {type_color}]{preview['type']}[/]"
                    
                    # 4. Build Dashboard
                    table = Table(title=f"[bold cyan]{self.config['SYMBOL']} Live Physics (SOTA)[/] [dim]Updated: {last_updated}[/]")
                    table.add_column("Sensor", style="cyan")
                    table.add_column("Value", style="bold white")
                    table.add_column("Interpretation", style="italic")

                    # -- OPTION PREVIEW ROW --
                    # Format: Option Contract │ PREVIEW: $237.5 PUT @ $13.00 │ Exp: 2026-02-13
                    opt_preview_str = f"PREVIEW: ${preview['strike']} {type_str} @ ${preview['price']:.2f}"
                    table.add_row("Option Contract", opt_preview_str, f"Exp: {preview['expiry']}")
                    
                    # -- AVG PRICE / STOP ROW --
                    # Format: Avg Price / Stop │ 0.00 / 0.00 │ Dist: 236.78
                    # We use 'Spot' for the third column to represent the current underlying price clearly
                    pos_str = f"{self.avg_price:.2f} / {self.stop_loss:.2f}"
                    table.add_row("Avg Price / Stop", pos_str, f"Spot: {current_price:.2f}")

                    # -- INVENTORY (Merged into status or separate?)
                    # Keeping it separate for clarity on contract count
                    max_contracts = self.config.get('MAX_CONTRACTS', 5)
                    inv_col = "white" if self.inventory == 0 else "yellow"
                    table.add_row("Current Inventory", f"[{inv_col}]{self.inventory} / {max_contracts}[/]", "Simulated Position")

                    # -- REGIME (HMM) --
                    regime_map = {0: "Calm (Mean Rev)", 1: "Volatile", 2: "EXTREME (Crash)"}
                    regime_val = int(latest['regime'])
                    regime_str = regime_map.get(regime_val, "Unknown")
                    regime_col = "green" if regime_val == 0 else "red"
                    table.add_row("Market Regime", f"[{regime_col}]{regime_str}[/]", "Target: Calm (0)")

                    # -- PHYSICS --
                    hurst_val = latest['hurst']
                    hurst_thresh = self.config.get('HURST_THRESHOLD', 0.5)
                    h_col = "green" if hurst_val < hurst_thresh else "red"
                    table.add_row("Hurst Exponent", f"[{h_col}]{hurst_val:.3f}[/]", f"< {hurst_thresh:.2f} = Reverting")

                    theta_val = latest['theta']
                    theta_thresh = self.config.get('THETA_THRESHOLD', 0.05)
                    t_col = "green" if theta_val > theta_thresh else "yellow"
                    table.add_row("OU Theta", f"[{t_col}]{theta_val:.4f}[/]", f"Threshold > {theta_thresh:.2f}")

                    # -- TECHNICALS --
                    z_thresh = self.config.get('SIGMA_ENTRY', 2.0)
                    z_col = "magenta" if abs(z_val) > z_thresh else "white"
                    table.add_row("OU Z-Score", f"[{z_col}]{z_val:.2f}[/]", f"Entry Signal @ > {z_thresh:.2f}")

                    rsi_val = latest['rsi']
                    rsi_col = "green" if rsi_val < 30 or rsi_val > 70 else "white"
                    table.add_row("RSI (14)", f"[{rsi_col}]{rsi_val:.1f}[/]", "Oversold < 30")

                    # -- AI VERDICT --
                    prob_col = "green" if prob > 0.6 else "white"
                    table.add_row("ML Confidence", f"[{prob_col}]{prob:.1%}[/]", ai_msg)

                    live.update(Panel(table))
                    time.sleep(2)

                except KeyboardInterrupt:
                    console.print("[yellow]Stopping Live Trader...[/]")
                    break
                except Exception as e:
                    live.update(Panel(f"[bold red]CRASH: {e}[/]\nRebooting sensor..."))
                    time.sleep(5)

if __name__ == "__main__":
    lt = LiveTrader()
    lt.run()