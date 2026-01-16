from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class StrategyProfile(BaseModel):
    """
    Schema for Ticker Strategy Profiles.
    Matches parameters in optimize_thresholds.py and backtest.py.
    """
    name: str = Field(..., example="AMD_Standard")
    symbol: str = Field(..., example="AMD")
    
    # Optimization & Entry Thresholds
    z_threshold_call: float = Field(default=2.3)
    z_threshold_put: float = Field(default=2.8)
    vol_threshold: float = Field(default=1.8)
    
    # Market Hours Logic
    start_time_min: int = Field(default=15, description="Mins from 9:30 AM")
    entry_cutoff_min: int = Field(default=270, description="Mins from 9:30 AM")
    hard_exit_min: int = Field(default=385, description="Mins from 9:30 AM")
    
    # Risk & Sizing
    max_contracts: int = Field(default=10)
    initial_size: int = Field(default=1)
    scale_in_step_atr: float = Field(default=1.25)
    atr_mult: float = Field(default=5.5, description="Stop Loss Multiplier")
    risk_free_rate: float = Field(default=0.045)
    target_expiry_days: int = Field(default=7)
    
    # Meta
    lookback_days: int = Field(default=480)
    last_optimized: Optional[datetime] = None

class ProfileUpdate(BaseModel):
    """Used for partial updates to a profile."""
    z_threshold_call: Optional[float] = None
    z_threshold_put: Optional[float] = None
    vol_threshold: Optional[float] = None
    max_contracts: Optional[int] = None
    atr_mult: Optional[float] = None