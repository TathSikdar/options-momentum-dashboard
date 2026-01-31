import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_step(script):
    console.print(f"[bold yellow]Running {script}...[/]")
    res = subprocess.run([sys.executable, script], check=False)
    if res.returncode != 0:
        console.print(f"[bold red]Failed: {script}[/]")
        sys.exit(1)
    console.print(f"[bold green]Success: {script}[/]\n")

def main():
    console.print(Panel("[bold cyan]SOTA System Initialization[/]", border_style="cyan"))
    
    # 1. Data & Physics
    if not os.path.exists("AMD_physics_data.csv"):
        run_step("data_manager.py")
    
    # 2. Backtest & Generation
    run_step("backtest_engine.py")
    
    # 3. Train
    run_step("train_brain.py")
    
    console.print(Panel("[bold green]System Ready![/] Run 'live_trader.py' to start monitoring.", border_style="green"))

if __name__ == "__main__":
    main()