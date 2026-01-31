import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM

class MarketPhysics:
    """
    SOTA Mean Reversion Mathematics.
    Implements Ornstein-Uhlenbeck (OU), Black-Scholes, Technical Physics, and HMM Regime Detection.
    """

    @staticmethod
    def get_market_regime(df):
        """
        Fits a Gaussian HMM to detect market regimes.
        Returns the dataframe with a 'regime' column (0, 1, 2).
        """
        # Prepare Feature Vector: [Returns, Volatility]
        # We use Log Returns for statistical stability
        df = df.copy()
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vol_60'] = df['log_ret'].rolling(window=60).std()
        
        # Drop NaNs created by rolling windows
        clean_df = df.dropna().copy()
        
        if len(clean_df) < 100:
            return df # Not enough data
            
        X = clean_df[['log_ret', 'vol_60']].values
        
        # Fit HMM with 3 components (Calm, Volatile, Extreme)
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        try:
            model.fit(X)
            hidden_states = model.predict(X)
            
            # Map states to logic (We want State 0 to ALWAYS be the 'Low Vol' state)
            # We calculate the variance of each state to sort them.
            # State with lowest variance = "Calm/Reverting"
            variances = [np.diag(model.covars_[i]).mean() for i in range(model.n_components)]
            state_map = np.argsort(variances) # e.g. [2, 0, 1] -> State 2 is actually lowest vol
            
            # Re-map the predictions so 0 is always Low Vol, 2 is always High Vol
            remapped_states = [np.where(state_map == s)[0][0] for s in hidden_states]
            
            # Align indices
            df.loc[clean_df.index, 'regime'] = remapped_states
            df['regime'] = df['regime'].fillna(1) # Default to 'Volatile' if unsure
            
        except Exception:
            df['regime'] = 1 # Fallback
            
        return df

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
        df = df.copy()
        df['date'] = df.index.date
        def calc_daily_vwap(group):
            cum_vol = group['Volume'].cumsum()
            cum_pv = (group['Close'] * group['Volume']).cumsum()
            return cum_pv / cum_vol
        return df.groupby('date').apply(calc_daily_vwap).reset_index(level=0, drop=True)

    @staticmethod
    def calculate_adx(df, period=14):
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