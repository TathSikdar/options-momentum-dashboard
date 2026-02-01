import subprocess
import sys
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_script(script_name):
    """Executes a python script and monitors for errors."""
    console.print(Panel(f"[bold yellow]Executing:[/] {script_name}", expand=False))
    
    try:
        # Run the script and wait for it to complete
        # We use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            console.print(f"[bold green]✓ Finished:[/] {script_name}\n")
            return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]✗ Error in {script_name}:[/] Return code {e.returncode}")
        return False
    except Exception as e:
        console.print(f"[bold red]✗ Unexpected error running {script_name}:[/] {e}")
        return False

def main():
    """
    Orchestrates the AMD Options Strategy Pipeline.
    Excludes main.py (Live Execution).
    """
    pipeline = [
        "optimize_thresholds.py", # Step 1: Find best Z-Score/Vol constants
        "backtest.py",            # Step 2: Generate historical campaigns using optimized constants
        "train_model.py",         # Step 3: Train the XGBoost 'Brain' on the generated data
        "backtest_results.py"     # Step 4: Audit the results and directional performance
    ]

    console.print(Panel.fit(
        "[bold cyan]AMD Strategy Pipeline Runner[/]\n"
        "[white]Sequential processing: Optimization -> Backtest -> Train -> Audit[/]",
        border_style="cyan"
    ))

    for script in pipeline:
        if not os.path.exists(script):
            console.print(f"[bold red]Critical Error:[/] File '{script}' not found. Aborting pipeline.")
            break
            
        success = run_script(script)
        
        if not success:
            console.print(Panel("[bold red]Pipeline Halted[/]\nResolve the error in the script above before restarting.", 
                                title="Execution Failed", border_style="red"))
            sys.exit(1)

    console.print(Panel(
        "[bold green]Full Pipeline Successful![/]\n"
        "1. Thresholds Optimized\n"
        "2. Training Data Exported\n"
        "3. XGBoost Model Calibrated\n"
        "4. Performance Audit Generated\n\n"
        "[yellow]You are now ready to run 'main.py' for live monitoring.[/]",
        title="Complete", border_style="green"
    ))

if __name__ == "__main__":
    main()