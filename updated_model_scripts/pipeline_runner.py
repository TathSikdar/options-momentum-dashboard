import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

# Get the directory where this script (pipeline_runner.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    console.print(f"[bold yellow]Running {script_name}...[/]")
    
    # We set cwd=BASE_DIR so the scripts can find 'strategy_config.json' and imports like 'market_physics'
    try:
        subprocess.run([sys.executable, script_path], check=True, cwd=BASE_DIR)
        console.print(f"[bold green]Done: {script_name}[/]\n")
    except subprocess.CalledProcessError:
        console.print(f"[bold red]Failed to run {script_name}[/]")
        sys.exit(1)
    except FileNotFoundError:
        console.print(f"[bold red]Could not find file: {script_path}[/]")
        sys.exit(1)

def main():
    console.print(Panel("[bold cyan]AMD SOTA System (Rolling Window Manager)[/]", border_style="cyan"))
    
    # 1. Fetch Data 
    # (Data Manager automatically truncates data older than BACKTEST_WINDOW_DAYS)
    run("data_manager.py")
    
    # 2. Hyperparameter Optimization (NEW STEP)
    # (Finds the best thresholds for the current regime via Bayesian Search)
    run("optimize_hyperparameters.py")
    
    # 3. Simulate Strategy
    # (Generates training samples using the newly OPTIMIZED config)
    run("backtest_engine.py")
    
    # 4. Retrain Brain
    # (Fits the XGBoost model to the new volatility regime)
    run("train_brain.py")
    
    console.print(Panel(
        "[bold green]System Update Complete![/]\n\n"
        "1. Data Refreshed (Last 180 Days)\n"
        "2. Strategy Thresholds Optimized (Bayesian Search)\n"
        "3. Backtest Simulation Run\n"
        "4. ML Brain Re-Trained\n\n"
        "[yellow]You are ready to run 'live_trader.py'[/]",
        border_style="green"
    ))

if __name__ == "__main__":
    main()