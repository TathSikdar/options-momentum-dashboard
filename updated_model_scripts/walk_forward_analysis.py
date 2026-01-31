import pandas as pd
import numpy as np
import optuna
import json
import sys
import logging
from datetime import timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from backtest_engine import BacktestEngine
from data_manager import DataManager

# Suppress noisy logs
optuna.logging.set_verbosity(optuna.logging.ERROR)
console = Console()

class WalkForwardAnalyzer:
    def __init__(self):
        self.dm = DataManager()
        self.full_df = self.dm.fetch_polygon_data() # Load all available data
        
        # WFA Configuration (The "Standard" Quarterly Roll)
        self.TRAIN_WINDOW_DAYS = 90
        self.TEST_WINDOW_DAYS = 30
        self.TRIALS_PER_FOLD = 30  # Number of optimization runs per window
        
    def optimize_slice(self, df_train):
        """
        Runs a mini-Bayesian Optimization on the training slice.
        Returns the best parameters found for this specific historical period.
        """
        def objective(trial):
            params = {
                "HURST_THRESHOLD": trial.suggest_float("HURST_THRESHOLD", 0.40, 0.65),
                "SIGMA_ENTRY": trial.suggest_float("SIGMA_ENTRY", 2.0, 3.5),
                "STOP_LOSS_ATR": trial.suggest_float("STOP_LOSS_ATR", 2.0, 5.0),
                "SCALE_IN_ATR": trial.suggest_float("SCALE_IN_ATR", 0.5, 2.0),
                "TARGET_EXPIRY_DAYS": trial.suggest_int("TARGET_EXPIRY_DAYS", 3, 5),
                "MAX_CONTRACTS": trial.suggest_int("MAX_CONTRACTS", 3, 8),
                # Constants
                "THETA_THRESHOLD": 0.05,
                "INITIAL_SIZE": 1,
                "RISK_FREE_RATE": 0.045
            }
            
            # Run Simulation
            engine = BacktestEngine(df_train, config_override=params)
            return engine.run_simulation(silent=True)

        # Run Optimization (Sequential for stability inside WFA loop)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.TRIALS_PER_FOLD, n_jobs=1)
        return study.best_params, study.best_value

    def run_analysis(self):
        console.print(Panel("[bold cyan]🚀 Starting Walk-Forward Analysis (WFA)[/]\n"
                            "[white]Stress-testing strategy robustness on unseen data...[/]", border_style="cyan"))

        # Ensure we have enough data
        min_days = self.TRAIN_WINDOW_DAYS + self.TEST_WINDOW_DAYS
        date_range = (self.full_df.index.max() - self.full_df.index.min()).days
        
        if date_range < min_days:
            console.print(f"[bold red]Not enough data![/] Need {min_days} days, have {date_range}.")
            return

        # Prepare Windows
        start_date = self.full_df.index.min()
        end_date = self.full_df.index.max()
        
        results = []
        current_train_start = start_date
        
        # Determine number of folds for progress bar
        folds = []
        while True:
            train_end = current_train_start + timedelta(days=self.TRAIN_WINDOW_DAYS)
            test_end = train_end + timedelta(days=self.TEST_WINDOW_DAYS)
            if test_end > end_date: break
            folds.append((current_train_start, train_end, test_end))
            current_train_start += timedelta(days=self.TEST_WINDOW_DAYS)

        console.print(f"[yellow]Identified {len(folds)} Walk-Forward Folds.[/]")

        # --- THE WALK FORWARD LOOP ---
        for i, (train_start, train_end, test_end) in enumerate(folds):
            console.print(f"\n[bold white]--- Fold {i+1}/{len(folds)} ---[/]")
            console.print(f"Train: [cyan]{train_start.date()} -> {train_end.date()}[/]")
            console.print(f"Test:  [magenta]{train_end.date()} -> {test_end.date()}[/] (Unseen Data)")
            
            # 1. Slice Data
            mask_train = (self.full_df.index >= train_start) & (self.full_df.index < train_end)
            mask_test = (self.full_df.index >= train_end) & (self.full_df.index < test_end)
            
            df_train = self.full_df.loc[mask_train].copy()
            df_test = self.full_df.loc[mask_test].copy()
            
            if len(df_train) == 0 or len(df_test) == 0:
                console.print("[red]Skipping empty fold.[/]")
                continue

            # 2. Optimize on In-Sample (Train) Data
            console.print(" > Optimizing parameters on training set...")
            best_params, is_pnl = self.optimize_slice(df_train)
            
            # 3. Validate on Out-of-Sample (Test) Data
            console.print(" > Testing parameters on unseen future data...")
            engine = BacktestEngine(df_test, config_override=best_params)
            oos_pnl = engine.run_simulation(silent=True)
            
            # Record Results
            results.append({
                "Period": f"{train_end.strftime('%Y-%m-%d')} -> {test_end.strftime('%m-%d')}",
                "IS_PnL": is_pnl,       # Predicted P&L (Expectation)
                "OOS_PnL": oos_pnl,     # Actual P&L (Reality)
                "Robustness": oos_pnl / is_pnl if is_pnl > 0 else 0.0
            })
            
            color = "green" if oos_pnl > 0 else "red"
            console.print(f" > Result: In-Sample P&L: ${is_pnl:.0f} | Out-of-Sample P&L: [{color}]${oos_pnl:.0f}[/]")

        # --- FINAL REPORT ---
        res_df = pd.DataFrame(results)
        
        table = Table(title="Walk-Forward Analysis Report")
        table.add_column("Test Period", style="cyan")
        table.add_column("Predicted P&L (IS)", justify="right")
        table.add_column("Actual P&L (OOS)", justify="right", style="bold")
        table.add_column("Robustness Ratio", justify="right")
        
        total_oos_pnl = 0
        total_is_pnl = 0
        
        for _, row in res_df.iterrows():
            total_oos_pnl += row['OOS_PnL']
            total_is_pnl += row['IS_PnL']
            r_color = "green" if row['OOS_PnL'] > 0 else "red"
            rob_color = "green" if row['Robustness'] > 0.5 else "yellow" if row['Robustness'] > 0 else "red"
            
            table.add_row(
                row['Period'],
                f"${row['IS_PnL']:,.0f}",
                f"[{r_color}]${row['OOS_PnL']:,.0f}[/]",
                f"[{rob_color}]{row['Robustness']:.2f}[/]"
            )
            
        console.print("\n")
        console.print(table)
        
        # Conclusion
        console.print(Panel(
            f"Total Predicted P&L: ${total_is_pnl:,.0f}\n"
            f"Total Actual P&L: [bold]${total_oos_pnl:,.0f}[/]\n\n"
            f"Overall Robustness Score: {total_oos_pnl / total_is_pnl if total_is_pnl else 0:.2f}",
            title="Validation Verdict",
            border_style="green" if total_oos_pnl > 0 else "red"
        ))
        
        if total_oos_pnl > 0:
            console.print("[bold green]PASS:[/]. The strategy is profitable on unseen data.")
        else:
            console.print("[bold red]FAIL:[/]. The strategy is overfit. It finds patterns in the past that do not persist in the future.")

if __name__ == "__main__":
    wfa = WalkForwardAnalyzer()
    wfa.run_analysis()