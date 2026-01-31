import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class MarketPhysics:
    """
    SOTA Mean Reversion Mathematics.
    Implements Ornstein-Uhlenbeck (OU) process fitting and Stationarity tests.
    """

    @staticmethod
    def get_hurst_exponent(time_series, max_lag=20):
        """
        Calculates the Hurst Exponent (H).
        H < 0.5: Mean Reverting (Safe to trade)
        H > 0.5: Trending (Do NOT trade mean reversion)
        """
        if len(time_series) < max_lag + 1:
            return 0.5 # Default to random if insufficient data
            
        lags = range(2, max_lag)
        # Calculate standard deviation of differences for various lags
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        
        # Use linear regression to find the slope
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    @staticmethod
    def fit_ornstein_uhlenbeck(data, dt=1/390):
        """
        Fits the Ornstein-Uhlenbeck process: dXt = theta * (mu - Xt) * dt + sigma * dWt
        Returns:
            theta (Speed of Reversion): Higher is better.
            mu (Equilibrium Level): The 'True Mean'.
            sigma (Volatility): Noise level.
        """
        x = np.array(data)
        if len(x) < 2: return 0.0, 0.0, 0.0
        
        xx = x[:-1] # x(t)
        xy = x[1:]  # x(t+1)
        
        # Linear Regression: xy = a*xx + b
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, xy, rcond=None)[0]
        
        if m <= 0 or m >= 1:
            return 0.0, np.mean(x), 0.0 
            
        theta = -np.log(m) / dt
        mu = c / (1 - m)
        
        residuals = xy - (m * xx + c)
        sigma_epsilon = np.std(residuals)
        sigma = sigma_epsilon / np.sqrt(dt)
        
        return theta, mu, sigma