import asyncio
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.engine import calculate_indicators, get_bs_price
from app.core.optimizer import ThresholdOptimizer
from app.core.backtester import StrategyBacktester
from app.core.trainer import ModelTrainer
from app.schemas.dashboard import LiveMetrics
from app.schemas.profile import StrategyProfile, ProfileUpdate

load_dotenv()

app = FastAPI(title="Options Momentum Dashboard API")

# Enable CORS for frontend hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demonstration (Replace with Firestore/DB later)
profiles: Dict[str, StrategyProfile] = {}
active_connections: List[WebSocket] = []

# --- APP STATE ---
class AppState:
    def __init__(self):
        self.monitoring_active = False
        self.current_metrics: LiveMetrics = None
        self.active_profile_id: str = None
        self.loaded_model = None
        self.loaded_model_id = None

state = AppState()

# --- API ROUTES ---

@app.get("/profiles", response_model=List[StrategyProfile])
async def get_all_profiles():
    return list(profiles.values())

@app.post("/profiles", response_model=StrategyProfile)
async def create_profile(profile: StrategyProfile):
    profiles[profile.name] = profile
    return profile

@app.patch("/profiles/{name}", response_model=StrategyProfile)
async def update_profile(name: str, update: ProfileUpdate):
    if name not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    current_data = profiles[name].dict()
    update_data = update.dict(exclude_unset=True)
    updated_profile = StrategyProfile(**{**current_data, **update_data})
    profiles[name] = updated_profile
    return updated_profile

# --- PIPELINE ROUTES ---

@app.post("/optimize/{name}")
async def run_optimization(name: str):
    """Triggers the grid search for best thresholds."""
    if name not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Polygon API Key not configured in .env")
        
    optimizer = ThresholdOptimizer(profiles[name], api_key)
    results = await optimizer.run_grid_search()
    
    if not results:
        return {"status": "error", "message": "No profitable combinations found with current settings."}
    
    return {"status": "success", "results": results[:10]}

@app.post("/backtest/{name}")
async def run_backtest(name: str):
    """Triggers a historical backtest and generates training data."""
    if name not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    backtester = StrategyBacktester(profiles[name])
    try:
        results = await backtester.run_backtest()
        return {"status": "success", "data": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/{name}")
async def train_profile_model(name: str):
    """Trains the XGBoost brain using data from the backtester."""
    if name not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    trainer = ModelTrainer(profiles[name])
    results = trainer.train_model()
    
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
        
    return results

# --- DASHBOARD LOGIC ---

async def monitor_market():
    """
    Background task that simulates the live execution loop.
    Updates the global state every 2 seconds.
    """
    while state.monitoring_active:
        if not state.active_profile_id or state.active_profile_id not in profiles:
            await asyncio.sleep(2)
            continue

        profile = profiles[state.active_profile_id]
        symbol = profile.symbol

        # Dynamically load/swap the model if the active profile changes
        if state.loaded_model_id != state.active_profile_id:
            model_path = f"backend/app/data/brains/{symbol}_model.pkl"
            if os.path.exists(model_path):
                state.loaded_model = joblib.load(model_path)
                state.loaded_model_id = state.active_profile_id
            else:
                state.loaded_model = None

        try:
            # 1. Fetch live data (matching main.py get_live_metrics)
            df = yf.download(symbol, period="2d", interval="1m", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                df = calculate_indicators(df)
                latest = df.iloc[-1]
                
                now = datetime.now()
                mins_open = (now.hour * 60 + now.minute) - (9 * 60 + 30)
                
                # 2. ML Inference (Incorporate the Brain)
                prob = 0.0
                if state.loaded_model:
                    # Prepare feature vector for XGBoost
                    feat = pd.DataFrame([[
                        float(latest['atr']), 
                        int(mins_open), 
                        float(latest['macd_z']), 
                        float(latest['vol_z']), 
                        float(latest['implied_vol'])
                    ]], columns=['atr', 'mins_open', 'macd_z', 'vol_z', 'implied_vol'])
                    prob = float(state.loaded_model.predict_proba(feat)[0][1])

                metrics = LiveMetrics(
                    timestamp=now.strftime("%I:%M:%S %p"),
                    price=float(latest['Close']),
                    raw_macd=float(latest['raw_macd']),
                    m_z=float(latest['macd_z']),
                    v_z=float(latest['vol_z']),
                    atr=float(latest['atr']),
                    implied_vol=float(latest['implied_vol']),
                    ml_confidence=prob,
                    mins_open=mins_open,
                    in_position=False, # Position logic tracking would happen here
                    contracts=0,
                    avg_price=0.0,
                    stop_level=0.0,
                    last_event="Monitoring - No Model" if not state.loaded_model else "Monitoring with XGBoost Brain",
                    contract_symbol=None,
                    contract_price=None,
                    contract_strike=None,
                    contract_expiry=None
                )
                
                state.current_metrics = metrics
                
                # 3. Push to all connected WebSockets
                for connection in active_connections:
                    try:
                        await connection.send_json(metrics.dict())
                    except:
                        pass # Connection cleanup handled by WebSocketDisconnect

        except Exception as e:
            print(f"Monitor Error: {e}")

        await asyncio.sleep(2)

# --- WEBSOCKETS ---

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive; messages are pushed from the monitor task
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

# --- CONTROL ROUTES ---

@app.post("/start/{profile_name}")
async def start_monitoring(profile_name: str, background_tasks: BackgroundTasks):
    if profile_name not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    state.active_profile_id = profile_name
    if not state.monitoring_active:
        state.monitoring_active = True
        background_tasks.add_task(monitor_market)
    
    return {"status": "Monitoring started", "profile": profile_name}

@app.post("/stop")
async def stop_monitoring():
    state.monitoring_active = False
    return {"status": "Monitoring stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)