import optuna
import json
import pandas as pd
import os
from rich.console import Console
from rich.panel import Panel
from backtest_engine import BacktestEngine
from data_manager import DataManager

console = Console()

# Global Data Cache
DATA_CACHE = None

def get_data():
    global DATA_CACHE
    if DATA_CACHE is None:
        dm = DataManager()
        DATA_CACHE = dm.fetch_polygon_data()
    return DATA_CACHE

def objective(trial):
    # Optimize Scaling & Safety Thresholds
    params = {
        "HURST_THRESHOLD": trial.suggest_float("HURST_THRESHOLD", 0.45, 0.60),
        "SIGMA_ENTRY": trial.suggest_float("SIGMA_ENTRY", 2.0, 3.0),
        "STOP_LOSS_ATR": trial.suggest_float("STOP_LOSS_ATR", 2.5, 5.0), # Wide stops for scaling
        "SCALE_IN_ATR": trial.suggest_float("SCALE_IN_ATR", 0.5, 1.5),   # Aggressive vs Conservative scaling
        "TARGET_EXPIRY_DAYS": trial.suggest_int("TARGET_EXPIRY_DAYS", 3, 5),
        "MAX_CONTRACTS": trial.suggest_int("MAX_CONTRACTS", 3, 8),
        
        # Fixed
        "THETA_THRESHOLD": 0.05,
        "INITIAL_SIZE": 1,
        "RISK_FREE_RATE": 0.045
    }
    
    df = get_data()
    engine = BacktestEngine(df, config_override=params)
    total_pnl = engine.run_simulation(silent=True)
    return total_pnl

def run_optimization():
    console.print(Panel("[bold cyan]Starting 'Hybrid Scaling' Optimization...[/]", border_style="cyan"))
    get_data()
    
    study = optuna.create_study(direction="maximize")
    console.print("[yellow]Running 50 Trials...[/]")
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    best_pnl = study.best_value
    
    console.print(Panel(
        f"[bold green]Optimization Complete![/]\n"
        f"Best P&L: [bold]${best_pnl:,.2f}[/]\n"
        f"Max Contracts: {best_params['MAX_CONTRACTS']} | "
        f"Scale Step: {best_params['SCALE_IN_ATR']:.2f} ATR",
        title="Results", border_style="green"
    ))
    
    final_config = {
        "SYMBOL": "AMD",
        "RISK_FREE_RATE": 0.045,
        "BACKTEST_WINDOW_DAYS": 180,
        "MAX_HOLD_MINUTES": 240,
        "THETA_THRESHOLD": 0.05,
        "INITIAL_SIZE": 1,
        **best_params
    }
    
    with open("strategy_config.json", "w") as f:
        json.dump(final_config, f, indent=4)
        console.print("[green]Saved to strategy_config.json[/]")

if __name__ == "__main__":
    run_optimization()