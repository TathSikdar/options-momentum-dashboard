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

# Quiet Mode
optuna.logging.set_verbosity(optuna.logging.ERROR)
console = Console()

# --- WORKER DATA LOADER ---
WORKER_DATA = None

def load_data_fast():
    global WORKER_DATA
    if WORKER_DATA is not None: return WORKER_DATA
    try:
        WORKER_DATA = pd.read_pickle("temp_fast_data.pkl")
        return WORKER_DATA
    except: return None

# Helper for Parallel Batching
def run_batch_trial(trial_params):
    """
    Independent worker function.
    """
    df = load_data_fast()
    if df is None: return -999999.0
    
    try:
        engine = BacktestEngine(df, config_override=trial_params)
        return engine.run_simulation(silent=True)
    except:
        return -999999.0

def run_optimization():
    console.print(Panel("[bold cyan]🚀 Turbo-Batch Optimization (100% CPU)[/]", border_style="cyan"))
    
    # Load Config
    try:
        with open("strategy_config.json", "r") as f:
            cfg = json.load(f)
            n_trials = cfg.get("OPTIMIZATION_TRIALS", 100)
    except:
        n_trials = 100

    # 1. Prepare Data
    console.print("[yellow]Caching data for workers...[/]")
    dm = DataManager()
    df = dm.fetch_polygon_data()
    df.to_pickle("temp_fast_data.pkl")
    
    # 2. Setup Study
    study_file = "optuna_study.pkl"
    if os.path.exists(study_file):
        study = joblib.load(study_file)
    else:
        study = optuna.create_study(direction="maximize", storage=None)
    
    # 3. Batch Configuration
    n_cores = joblib.cpu_count()
    batch_size = n_cores * 2 # Keep the queue full
    
    console.print(Panel(
        f"Specs:\n"
        f" > Trials: {n_trials}\n"
        f" > Batch Size: {batch_size} (Parallel Futures)\n"
        f" > Engine: Numpy Accelerated",
        title="Performance", border_style="green"
    ))
    
    # 4. Optimization Loop
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(), transient=False
    ) as progress:
        
        task = progress.add_task("Optimizing...", total=n_trials, completed=len(study.trials))
        
        while len(study.trials) < n_trials:
            try:
                # A. Ask for N suggestions
                trials_to_run = []
                for _ in range(batch_size):
                    if len(study.trials) + len(trials_to_run) >= n_trials: break
                    
                    # Create a trial object
                    trial = study.ask()
                    
                    # Define search space
                    params = {
                        "HURST_THRESHOLD": trial.suggest_float("HURST_THRESHOLD", 0.40, 0.65),
                        "SIGMA_ENTRY": trial.suggest_float("SIGMA_ENTRY", 2.0, 3.5),
                        "STOP_LOSS_ATR": trial.suggest_float("STOP_LOSS_ATR", 2.0, 6.0),
                        "SCALE_IN_ATR": trial.suggest_float("SCALE_IN_ATR", 0.5, 2.0),
                        "TARGET_EXPIRY_DAYS": trial.suggest_int("TARGET_EXPIRY_DAYS", 3, 5),
                        "MAX_CONTRACTS": trial.suggest_int("MAX_CONTRACTS", 3, 10),
                        # Fixed
                        "THETA_THRESHOLD": 0.05,
                        "INITIAL_SIZE": 1,
                        "RISK_FREE_RATE": 0.045
                    }
                    trials_to_run.append((trial, params))
                
                if not trials_to_run: break
                
                # B. Run Batch in Parallel
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(run_batch_trial)(params) 
                    for _, params in trials_to_run
                )
                
                # C. Report Results
                for (trial, _), result in zip(trials_to_run, results):
                    study.tell(trial, result)
                    progress.update(task, advance=1)
                
                # Auto-save
                joblib.dump(study, study_file)
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Optimization interrupted by user. Saving progress...[/]")
                joblib.dump(study, study_file)
                console.print(f"[green]Progress saved to {study_file}. You can resume later.[/]")
                sys.exit(0)
            
    # 5. Results & Cleanup
    best_params = study.best_params
    best_pnl = study.best_value
    
    console.print(Panel(
        f"🏆 Best P&L: [bold]${best_pnl:,.2f}[/]",
        title="Optimization Complete", border_style="green"
    ))
    
    final_config = {
        "SYMBOL": "AMD",
        "RISK_FREE_RATE": 0.045,
        "BACKTEST_WINDOW_DAYS": 180,
        "MAX_HOLD_MINUTES": 240,
        "THETA_THRESHOLD": 0.05,
        "INITIAL_SIZE": 1,
        "OPTIMIZATION_TRIALS": n_trials,
        **best_params
    }
    
    with open("strategy_config.json", "w") as f:
        json.dump(final_config, f, indent=4)
        
    if os.path.exists("temp_fast_data.pkl"):
        os.remove("temp_fast_data.pkl")

if __name__ == "__main__":
    run_optimization()