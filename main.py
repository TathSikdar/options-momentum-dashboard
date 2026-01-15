import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import joblib
import json
from datetime import datetime, timedelta
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group

# --- SOUND NOTIFICATION SYSTEM ---
try:
    import winsound
    def play_alert(alert_type):
        """Plays distinct frequencies based on the event type."""
        if alert_type == "CALL_ENTRY":
            winsound.Beep(1000, 600)  # High-mid tone for Calls
        elif alert_type == "PUT_ENTRY":
            winsound.Beep(1500, 600)  # High sharp tone for Puts
        elif alert_type == "SCALE":
            winsound.Beep(800, 300)   # Short blip for scaling
        elif alert_type == "EXIT":
            winsound.Beep(400, 800)   # Low long tone for closing
except ImportError:
    def play_alert(alert_type):
        """Fallback for non-Windows systems (Terminal Bell)."""
        print("\a", end="", flush=True)

# --- 1. SETTINGS & AUTO-CONFIG ---
SYMBOL = "AMD"
MAX_CONTRACTS = 10
INITIAL_SIZE = 1          
SCALE_IN_STEP_ATR = 1.25  
START_TIME_MIN = 15       # 9:45 AM EST
ENTRY_CUTOFF_MIN = 270    # 2:00 PM EST 
HARD_EXIT_MIN = 385       # 3:55 PM EST 
MODEL_FILE = "amd_model.pkl"
CONFIG_FILE = "strategy_params.json"
TARGET_EXPIRY_DAYS = 14

# Default thresholds (Fallback if config is missing)
Z_THRESHOLD_CALL = 2.3    
Z_THRESHOLD_PUT = 2.8     
VOL_THRESHOLD = 1.8       

# Load optimized parameters from the Threshold Optimizer
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            Z_THRESHOLD_CALL = float(config.get("Z_THRESHOLD_CALL", Z_THRESHOLD_CALL))
            Z_THRESHOLD_PUT = float(config.get("Z_THRESHOLD_PUT", Z_THRESHOLD_PUT))
            VOL_THRESHOLD = float(config.get("VOL_THRESHOLD", VOL_THRESHOLD))
    except Exception:
        pass # Fall back to defaults on error

class TradeManager:
    def __init__(self):
        self.in_position = False
        self.contracts = 0
        self.total_cost = 0
        self.avg_price = 0
        self.stop_level = 0
        self.last_event = "System Ready - Monitoring for Signals"
        self.current_contract = "None"
        self.current_strike = 0.0
        self.contract_price = 0.0
        self.trade_direction = None 
    
    def reset(self):
        self.__init__()

tm = TradeManager()
console = Console()

# Load the Brain
model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

def get_best_option_contract(symbol, stock_price, direction):
    """Fetches live ATM contracts closest to 14-day expiry."""
    try:
        tk = yf.Ticker(symbol)
        all_expiries = tk.options
        target_date = datetime.now() + timedelta(days=TARGET_EXPIRY_DAYS)
        best_expiry = min(all_expiries, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
        
        opt_chain = tk.option_chain(best_expiry)
        chain = opt_chain.calls if direction == 'Call' else opt_chain.puts
        
        # Find ATM
        atm_row = chain.iloc[(chain['strike'] - stock_price).abs().argsort()[:1]]
        
        return {
            "symbol": atm_row['contractSymbol'].values[0],
            "price": atm_row['lastPrice'].values[0],
            "expiry": best_expiry,
            "strike": atm_row['strike'].values[0]
        }
    except Exception:
        return {"symbol": "CHAIN_ERROR", "price": 0.0, "expiry": "N/A", "strike": 0}

def get_live_metrics():
    """Calculates live sensors for the XGBoost model with stabilized indicators."""
    try:
        # Fetching 5 days to provide EMA "warm-up"
        df = yf.download(SYMBOL, period="5d", interval="1m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # EMA calculations
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        
        # Z-Score and ATR calculations
        m_z = (macd - macd.rolling(30).mean()) / macd.rolling(30).std()
        v_z = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()
        
        tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        vol_proxy = df['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252 * 390)
        
        # Ensure we get the latest non-zero volume (yfinance 1m can lag)
        raw_vol = df['Volume'].replace(0, np.nan).ffill().iloc[-1]
        if pd.isna(raw_vol): raw_vol = 0
        
        return {
            "price": float(df['Close'].iloc[-1]), 
            "raw_macd": float(macd.iloc[-1]),
            "raw_vol": float(raw_vol),
            "m_z": float(m_z.iloc[-1]), 
            "v_z": float(v_z.iloc[-1]), 
            "atr": float(atr), 
            "implied_vol": float(vol_proxy)
        }
    except Exception as e:
        return None

def update_dashboard(data, prob, mins_open, opt_info=None):
    # Time formatting for 12-hour clock
    now = datetime.now()
    timestamp = now.strftime("%I:%M:%S %p")
    
    # Determine color and status message based on market hours
    if mins_open < START_TIME_MIN:
        time_color = "red"
        status_msg = "Market Opens Next: 9:30 AM" if mins_open < 0 else "Market Warm-up (Wait until 9:45 AM)"
    elif START_TIME_MIN <= mins_open < ENTRY_CUTOFF_MIN:
        time_color = "green"
        if mins_open >= (ENTRY_CUTOFF_MIN - 20):
            status_msg = "WARNING: Entry Window Closing Soon"
        else:
            status_msg = "Market Open: Entry Window Active"
    elif ENTRY_CUTOFF_MIN <= mins_open < HARD_EXIT_MIN:
        time_color = "yellow"
        if mins_open >= (HARD_EXIT_MIN - 15):
            status_msg = "WARNING: Hard Exit Approaching"
        else:
            status_msg = "Monitoring Exits Only (No New Entries)"
    else:
        time_color = "red"
        status_msg = "Market Closed - No Active Monitoring"

    # Create the unified table with dynamic title
    title_str = f"[bold cyan]AMD Options Control Center[/] | [{time_color}]Last Update: {timestamp}[/] | [white]{status_msg}[/]"
    table = Table(title=title_str, expand=True)
    table.add_column("Sensor/Action", style="white")
    table.add_column("Current Value", style="bold magenta")
    table.add_column("Logic Note", style="italic white")
    
    # 1. ML Row
    window_status = "[green]WINDOW OPEN[/]" if START_TIME_MIN <= mins_open < ENTRY_CUTOFF_MIN else "[red]WINDOW CLOSED[/]"
    table.add_row("ML Confidence", f"{prob:.1%}", "Entry > 80.0%" if prob > 0 else window_status)
    
    # 2. Momentum & Volume Row (Consolidated)
    z_val = data['m_z']
    raw_m = data['raw_macd']
    vol_z = data.get('v_z', 0)
    
    if tm.in_position:
        direction_val = f"[bold green]ACTIVE {tm.trade_direction}[/]"
        logic_note = f"Exit when Z reaches 0.0"
    else:
        direction_val = f"{'Call' if z_val < 0 else 'Put'} Watch"
        logic_note = f"Z: {z_val:+.2f} | MACD: {raw_m:+.3f} | VZ: {vol_z:+.2f}"
    
    table.add_row("Execution Target", direction_val, logic_note)
    
    # 3. Option Contract Row
    if tm.in_position:
        contract_val = f"${tm.current_strike} {tm.trade_direction} @ ${tm.contract_price:.2f}"
        expiry_val = f"Exp: {opt_info['expiry'] if opt_info else 'N/A'}"
    elif opt_info and opt_info['symbol'] != "CHAIN_ERROR":
        c_type = "PUT" if "P" in opt_info['symbol'][-9:] else "CALL"
        contract_val = f"[yellow]PREVIEW:[/] ${opt_info['strike']} {c_type} @ ${opt_info['price']:.2f}"
        expiry_val = f"Exp: {opt_info['expiry']}"
    else:
        contract_val = "Scanning Chain..."
        expiry_val = "N/A"
    table.add_row("Option Contract", contract_val, expiry_val)
    
    # 4. Risk/Inventory Row
    pos_color = "green" if tm.in_position else "white"
    v_target = float(VOL_THRESHOLD)
    table.add_row("Inventory Size", f"[{pos_color}]{tm.contracts} / {MAX_CONTRACTS}[/]", f"Thresh: C{Z_THRESHOLD_CALL:.1f}/P{Z_THRESHOLD_PUT:.1f}/V{v_target:.1f}")
    table.add_row("Avg Price / Stop", f"{tm.avg_price:.2f} / [red]{tm.stop_level:.2f}[/]", f"Dist: {abs(data['price']-tm.stop_level):.2f}")
    
    return Panel(table, subtitle=f"Last Event: {tm.last_event}", border_style="cyan")

# --- EXECUTION LOOP ---
with Live(Panel("Initializing Sensors & Brain..."), refresh_per_second=1) as live:
    opt_info = None
    while True:
        data = get_live_metrics()
        if data:
            now = datetime.now()
            mins_open = (now.hour * 60 + now.minute) - (9 * 60 + 30)
            
            # XGBOOST INFERENCE
            prob = 0
            if model:
                feat = pd.DataFrame([[data['atr'], mins_open, data['m_z'], data['v_z'], data['implied_vol']]], 
                                    columns=['atr', 'mins_open', 'macd_z', 'vol_z', 'implied_vol'])
                prob = model.predict_proba(feat)[0][1]

            # PREVIEW LOGIC
            if not tm.in_position:
                preview_dir = 'Call' if data['m_z'] < 0 else 'Put'
                opt_info = get_best_option_contract(SYMBOL, data['price'], preview_dir)

            # ENTRY & SCALING
            if START_TIME_MIN <= mins_open < ENTRY_CUTOFF_MIN:
                is_call = data['m_z'] < -Z_THRESHOLD_CALL
                is_put = data['m_z'] > Z_THRESHOLD_PUT
                direction = 'Call' if is_call else 'Put' if is_put else None
                
                if not tm.in_position and direction and data['v_z'] > VOL_THRESHOLD and prob > 0.80:
                    opt_info = get_best_option_contract(SYMBOL, data['price'], direction)
                    tm.trade_direction = direction
                    tm.current_contract = opt_info['symbol']
                    tm.current_strike = opt_info['strike']
                    tm.contract_price = opt_info['price']
                    tm.in_position = True
                    tm.contracts = INITIAL_SIZE
                    tm.avg_price = data['price']
                    
                    mult = 6.0
                    tm.stop_level = data['price'] - (data['atr'] * mult) if direction == 'Call' else data['price'] + (data['atr'] * mult)
                    tm.last_event = f"BOUGHT {direction} ${tm.current_strike} | Prob: {prob:.1%}"
                    play_alert("CALL_ENTRY" if direction == "Call" else "PUT_ENTRY")

                elif tm.in_position and tm.contracts < MAX_CONTRACTS:
                    against_call = tm.trade_direction == 'Call' and data['price'] < (tm.avg_price - (data['atr'] * SCALE_IN_STEP_ATR))
                    against_put = tm.trade_direction == 'Put' and data['price'] > (tm.avg_price + (data['atr'] * SCALE_IN_STEP_ATR))
                    
                    if against_call or against_put:
                        opt_info = get_best_option_contract(SYMBOL, data['price'], tm.trade_direction)
                        tm.contracts += 1  
                        tm.current_strike = opt_info['strike']
                        tm.contract_price = opt_info['price']
                        tm.avg_price = (tm.avg_price + data['price']) / 2
                        tm.last_event = f"SCALING: Added {tm.trade_direction} ${tm.current_strike} | New Count: {tm.contracts}"
                        play_alert("SCALE")

            # EXIT LOGIC
            if tm.in_position:
                eod_exit = mins_open >= HARD_EXIT_MIN
                reversion = (tm.trade_direction == 'Call' and data['m_z'] >= 0) or \
                            (tm.trade_direction == 'Put' and data['m_z'] <= 0)
                stop_hit = (tm.trade_direction == 'Call' and data['price'] <= tm.stop_level) or \
                           (tm.trade_direction == 'Put' and data['price'] >= tm.stop_level)
                
                if eod_exit or reversion or stop_hit:
                    reason = "EOD FLATTEN" if eod_exit else "REVERSION" if reversion else "STOP LOSS"
                    tm.last_event = f"EXIT: {reason} at {data['price']:.2f}"
                    tm.reset()
                    play_alert("EXIT")

            live.update(update_dashboard(data, prob, mins_open, opt_info))
        
        time.sleep(2)