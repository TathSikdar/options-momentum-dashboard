import numpy as np
import pandas as pd
from scipy.stats import norm

# --- BLACK-SCHOLES PRICING ---
def get_bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculates Black-Scholes option price. Matches optimize_thresholds.py logic.
    """
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    # Avoid division by zero in sigma or T
    if sigma <= 0 or T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- INDICATOR CALCULATION ---
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends strategy indicators. Logic matches: optimize_thresholds.py & main.py.
    """
    if df.empty or len(df) < 35:
        return df

    # 1. MACD (12, 26)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    
    # 2. Z-Scores (30-period rolling)
    df['macd_z'] = (macd - macd.rolling(30).mean()) / macd.rolling(30).std()
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()
    
    # 3. ATR (14)
    tr = pd.concat([
        (df['High'] - df['Low']), 
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # 4. Volatility Proxy (for Black-Scholes)
    # Annualized volatility: std * sqrt(days * mins)
    df['implied_vol'] = df['Close'].pct_change().rolling(30).std() * np.sqrt(252 * 390)
    
    # 5. Market Timing (Minutes from Open)
    if isinstance(df.index, pd.DatetimeIndex):
        df['mins_open'] = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    
    df['raw_macd'] = macd
    
    return df