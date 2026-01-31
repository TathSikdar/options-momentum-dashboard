import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

class MarketPhysics:
    """
    SOTA Mean Reversion Mathematics.
    Implements Ornstein-Uhlenbeck (OU), Black-Scholes, and Technical Physics.
    """

    @staticmethod
    def get_hurst_exponent(time_series, max_lag=20):
        if len(time_series) < max_lag + 1: return 0.5
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    @staticmethod
    def fit_ornstein_uhlenbeck(data, dt=1/390):
        x = np.array(data)
        if len(x) < 2: return 0.0, 0.0, 0.0
        xx = x[:-1]
        xy = x[1:]
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, xy, rcond=None)[0]
        if m <= 0 or m >= 1: return 0.0, np.mean(x), 0.0 
        theta = -np.log(m) / dt
        mu = c / (1 - m)
        residuals = xy - (m * xx + c)
        sigma = np.std(residuals) / np.sqrt(dt)
        return theta, mu, sigma

    @staticmethod
    def get_bs_price(S, K, T, r, sigma, option_type='call'):
        if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def calculate_vwap(df):
        """
        Calculates Intraday VWAP. Resets at the start of each trading day.
        Formula: Cumulative(Price * Volume) / Cumulative(Volume)
        """
        df = df.copy()
        # Group by Date to reset VWAP daily
        df['date'] = df.index.date
        
        def calc_daily_vwap(group):
            cum_vol = group['Volume'].cumsum()
            cum_pv = (group['Close'] * group['Volume']).cumsum()
            return cum_pv / cum_vol
            
        return df.groupby('date').apply(calc_daily_vwap).reset_index(level=0, drop=True)

    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculates Average Directional Index (ADX) to measure trend strength.
        """
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low'] - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        return adx.fillna(0)