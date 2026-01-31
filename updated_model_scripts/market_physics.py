import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class MarketPhysics:
    """
    SOTA Mean Reversion Mathematics.
    Implements Ornstein-Uhlenbeck (OU) process fitting and Stationarity tests
    to detect 'Tradeable Regimes' vs 'Trending Regimes'.
    """

    @staticmethod
    def get_hurst_exponent(time_series, max_lag=20):
        """
        Calculates the Hurst Exponent (H) to classify the time series.
        H < 0.5: Mean Reverting (Safe to trade)
        H = 0.5: Random Walk (Geometric Brownian Motion)
        H > 0.5: Trending (Do NOT trade mean reversion)
        """
        lags = range(2, max_lag)
        # Calculate standard deviation of differences for various lags
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        
        # Use linear regression to find the slope (Hurst)
        # log(std_dev) = H * log(time_lag) + c
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    @staticmethod
    def adf_test(time_series):
        """
        Augmented Dickey-Fuller test for stationarity.
        Returns p-value.
        p-value < 0.05 implies the series is stationary (Good for Mean Reversion).
        """
        try:
            result = adfuller(time_series)
            return result[1] # p-value
        except:
            return 1.0

    @staticmethod
    def fit_ornstein_uhlenbeck(data, dt=1/390):
        """
        Fits the Ornstein-Uhlenbeck process: dXt = theta * (mu - Xt) * dt + sigma * dWt
        
        Uses Linear Regression logic on the discretized version:
        x(t+1) = a * x(t) + b + epsilon
        
        Returns:
            theta (Speed of Reversion): Higher is better.
            mu (Equilibrium Level): The 'True Mean' price is trying to reach.
            sigma (Volatility): Noise level.
        """
        x = np.array(data)
        xx = x[:-1] # x(t)
        xy = x[1:]  # x(t+1)
        
        # Linear Regression: xy = a*xx + b
        # Using numpy's efficient linear algebra solver
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, xy, rcond=None)[0]
        
        # Recover OU parameters from regression coefficients
        # a = exp(-theta * dt) => theta = -ln(a) / dt
        # b = mu * (1 - a)     => mu = b / (1 - a)
        
        # Safety check for 'a' to prevent log of negative numbers or division by zero
        if m <= 0 or m >= 1:
            return 0.0, np.mean(x), 0.0 # Fallback: No reversion detected
            
        theta = -np.log(m) / dt
        mu = c / (1 - m)
        
        # Calculate residuals for sigma
        residuals = xy - (m * xx + c)
        sigma_epsilon = np.std(residuals)
        
        # sigma_ou = sigma_epsilon / sqrt((1 - exp(-2*theta*dt)) / (2*theta))
        # For small dt, approx: sigma_ou = sigma_epsilon / sqrt(dt)
        sigma = sigma_epsilon / np.sqrt(dt)
        
        return theta, mu, sigma

    @staticmethod
    def calculate_half_life(theta):
        """
        Time required for the price to revert halfway back to the mean.
        Half-Life = ln(2) / theta
        Lower is better.
        """
        if theta <= 1e-5: return float('inf')
        return np.log(2) / theta