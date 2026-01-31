import optuna
import json
import pandas as pd
import os
import joblib
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from backtest_engine import BacktestEngine
from data_manager import DataManager

# Suppress Optuna logging to keep console clean for our progress bar
optuna.logging.set_verbosity(optuna.logging.ERROR)
console = Console()

# --- WORKER DATA LOADER ---
WORKER_DATA = None

def load_data_fast():
    """
    Fastest way to load data in a worker process.
    Reads binary pickle instead of parsing CSV text.
    """
    global WORKER_DATA
    if WORKER_DATA is not None:
        return WORKER_DATA
    
    try:
        # Load the binary cache created by the main process
        WORKER_DATA = pd.read_pickle("temp_fast_data.pkl")
        return WORKER_DATA
    except Exception:
        return None

def objective(trial):
    """
    Optuna Objective Function.
    """
    # 1. Suggest Parameters
    params = {
        "HURST_THRESHOLD": trial.suggest_float("HURST_THRESHOLD", 0.40, 0.65),
        "SIGMA_ENTRY": trial.suggest_float("SIGMA_ENTRY", 2.0, 3.5),
        "STOP_LOSS_ATR": trial.suggest_float("STOP_LOSS_ATR", 2.0, 6.0),
        "SCALE_IN_ATR": trial.suggest_float("SCALE_IN_ATR", 0.5, 2.0),
        "TARGET_EXPIRY_DAYS": trial.suggest_int("TARGET_EXPIRY_DAYS", 3, 5),
        "MAX_CONTRACTS": trial.suggest_int("MAX_CONTRACTS", 3, 10),
        
        # Fixed Constants
        "THETA_THRESHOLD": 0.05,
        "INITIAL_SIZE": 1,
        "RISK_FREE_RATE": 0.045
    }
    
    # 2. Load Data (Binary Fast Load)
    df = load_data_fast()
    
    if df is None or len(df) == 0:
        return -999999.0

    # 3. Run Simulation
    try:
        engine = BacktestEngine(df, config_override=params)
        total_pnl = engine.run_simulation(silent=True)
        return total_pnl
    except Exception:
        return -999999.0

def run_optimization():
    console.print(Panel("[bold cyan]🚀 Enterprise Optimization Engine (Auto-Save Enabled)[/]", border_style="cyan"))
    
    # --- LOAD CONFIGURATION ---
    # We read the N_TRIALS from the JSON file now
    try:
        with open("strategy_config.json", "r") as f:
            existing_config = json.load(f)
            n_trials = existing_config.get("OPTIMIZATION_TRIALS", 100) # Default to 100 if missing
    except FileNotFoundError:
        n_trials = 100
        existing_config = {}

    # 1. Prepare Data (Main Thread)
    console.print("[yellow]Generating binary cache for high-speed workers...[/]")
    dm = DataManager()
    # Fetch/Load standard data
    df = dm.fetch_polygon_data()
    # Save as Pickle (Binary) for instant worker access
    df.to_pickle("temp_fast_data.pkl")
    
    # 2. Setup or Resume Study
    study_file = "optuna_study.pkl"
    if os.path.exists(study_file):
        console.print(f"[green]Resuming from {study_file}...[/]")
        study = joblib.load(study_file)
    else:
        # Create new in-memory study
        study = optuna.create_study(direction="maximize", storage=None)
    
    n_jobs = -1 # All Cores
    
    console.print(Panel(
        f"Configuration:\n"
        f" > Target Trials: [bold]{n_trials}[/]\n"
        f" > Parallel Jobs: [bold]ALL CORES[/]\n"
        f" > Safety: [bold]Auto-Save Enabled[/] (optuna_study.pkl)",
        title="Job Spec", border_style="green"
    ))
    
    # 3. Optimization Loop with Progress Bar
    current_trials = len(study.trials)
    remaining = n_trials - current_trials
    
    if remaining > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=False
        ) as progress:
            
            task = progress.add_task("Optimizing...", total=n_trials, completed=current_trials)
            
            def callback(study, trial):
                progress.update(task, advance=1)
                if trial.number % 10 == 0:
                    joblib.dump(study, study_file)
            
            study.optimize(objective, n_trials=remaining, n_jobs=n_jobs, callbacks=[callback])
    else:
        console.print("[green]Target trials already reached. Showing results...[/]")
            
    # Final Save
    joblib.dump(study, study_file)
    
    # 4. Results
    best_params = study.best_params
    best_pnl = study.best_value
    
    console.print(Panel(
        f"[bold green]Optimization Complete![/]\n\n"
        f"🏆 Best P&L Found: [bold]${best_pnl:,.2f}[/]\n"
        f"Trials Completed: {len(study.trials)}\n\n"
        f"Optimal Settings:\n"
        f" > Hurst Threshold: {best_params['HURST_THRESHOLD']:.3f}\n"
        f" > Entry Z-Score: {best_params['SIGMA_ENTRY']:.2f}\n"
        f" > Stop Loss (ATR): {best_params['STOP_LOSS_ATR']:.2f}\n"
        f" > Scale Step (ATR): {best_params['SCALE_IN_ATR']:.2f}\n"
        f" > Max Contracts: {best_params['MAX_CONTRACTS']}\n"
        f" > DTE: {best_params['TARGET_EXPIRY_DAYS']}",
        title="Winner Circle", border_style="green"
    ))
    
    # 5. Save Config
    final_config = {
        "SYMBOL": "AMD",
        "RISK_FREE_RATE": 0.045,
        "BACKTEST_WINDOW_DAYS": 180,
        "MAX_HOLD_MINUTES": 240,
        "THETA_THRESHOLD": 0.05,
        "INITIAL_SIZE": 1,
        "OPTIMIZATION_TRIALS": n_trials, # Persist the setting
        **best_params
    }
    
    with open("strategy_config.json", "w") as f:
        json.dump(final_config, f, indent=4)
        console.print("[green]Saved optimal parameters to strategy_config.json[/]")
        
    # Cleanup temp file
    if os.path.exists("temp_fast_data.pkl"):
        os.remove("temp_fast_data.pkl")

if __name__ == "__main__":
    run_optimization()