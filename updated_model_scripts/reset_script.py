import os
from rich.console import Console

console = Console()

files_to_wipe = [
    "amd_optimization.db",
    "optuna_study.pkl",
    "sota_training_data.csv",
    "sota_brain.pkl",
    "strategy_config.json" # Optional: Delete this if you want to reset parameters to defaults
]

console.print("[bold red]Wiping stale data for System Reset...[/]")

for f in files_to_wipe:
    if os.path.exists(f):
        try:
            os.remove(f)
            console.print(f" > Deleted [yellow]{f}[/]")
        except Exception as e:
            console.print(f" > Error deleting {f}: {e}")
    else:
        console.print(f" > [dim]{f} not found (clean)[/]")

console.print("\n[bold green]System Reset Complete.[/]")
console.print("Run 'python pipeline_runner.py' to start fresh.")