from pydantic import BaseModel
from typing import Optional

class LiveMetrics(BaseModel):
    """
    Schema for the 2-second dashboard updates.
    Matches the data structure in main.py update_dashboard.
    """
    timestamp: str
    price: float
    raw_macd: float
    m_z: float
    v_z: float
    atr: float
    implied_vol: float
    ml_confidence: float
    mins_open: int
    
    # Current Trade Status
    in_position: bool
    trade_direction: Optional[str] = None
    contracts: int
    avg_price: float
    stop_level: float
    last_event: str
    
    # Option Data
    contract_symbol: Optional[str] = None
    contract_price: Optional[float] = None
    contract_strike: Optional[float] = None
    contract_expiry: Optional[str] = None