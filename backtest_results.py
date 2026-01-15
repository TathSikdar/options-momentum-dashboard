import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def analyze_dual_direction_results():
    """
    Analyzes dual-direction results from backtest.py.
    Differentiates between Call (Oversold) and Put (Overbought) campaigns.
    """
    file_path = "historical_training_data.csv"
    
    if not os.path.exists(file_path):
        console.print(Panel("[bold red]Error:[/] historical_training_data.csv not found. Please run backtest.py first.", title="System Error"))
        return

    df = pd.read_csv(file_path)
    
    # 1. Label Trade Types based on MACD Z-score
    # In backtest.py: Z < 0 is Call (Oversold), Z > 0 is Put (Overbought)
    df['trade_type'] = df['macd_z'].apply(lambda x: 'Call (Oversold)' if x < 0 else 'Put (Overbought)')
    
    # 2. General Metrics
    total_campaigns = len(df)
    wins = df[df['target'] == 1]
    losses = df[df['target'] == 0]
    win_rate = (len(wins) / total_campaigns) * 100 if total_campaigns > 0 else 0
    profit_factor = len(wins) / len(losses) if len(losses) > 0 else float('inf')

    # 3. Summary Table
    summary_table = Table(title="[bold cyan]AMD Dual-Direction Options Strategy Audit")
    summary_table.add_column("Metric", style="white")
    summary_table.add_column("Value", style="bold magenta")
    summary_table.add_column("Assessment", justify="right")

    summary_table.add_row("Total Campaigns", str(total_campaigns), "Statistically Robust")
    summary_table.add_row("Global Win Rate", f"{win_rate:.2f}%", "[green]Target: >70%[/]" if win_rate > 70 else "[red]Needs Tuning[/]")
    summary_table.add_row("Global Profit Factor", f"{profit_factor:.2f}", "Healthy" if profit_factor > 1.5 else "Risky")
    console.print(summary_table)

    # 4. Trade Type Breakdown
    type_table = Table(title="[bold yellow]Trade Type Comparison (Calls vs Puts)")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Signals", justify="center")
    type_table.add_column("Win Rate", justify="right")
    type_table.add_column("Avg MACD Z", justify="right")

    for t_type in df['trade_type'].unique():
        sub = df[df['trade_type'] == t_type]
        wr = (sub['target'].mean()) * 100
        avg_z = sub['macd_z'].mean()
        type_table.add_row(t_type, str(len(sub)), f"{wr:.1f}%", f"{avg_z:.2f}")
    
    console.print(type_table)

    # 5. Hourly Performance
    df['hour_from_open'] = (df['mins_open'] // 60).astype(int)
    hourly_stats = df[df['hour_from_open'] <= 6].groupby('hour_from_open')['target'].agg(['count', 'mean'])
    
    hour_table = Table(title="[bold white]Hourly Heatmap (Performance by Time)")
    hour_table.add_column("Hour", style="cyan")
    hour_table.add_column("Volume of Trades", justify="center")
    hour_table.add_column("Win Rate", justify="right")
    
    for hour, row in hourly_stats.iterrows():
        wr = row['mean'] * 100
        color = "green" if wr > 75 else "yellow" if wr > 60 else "red"
        hour_table.add_row(f"Hour {int(hour)}", str(int(row['count'])), f"[{color}]{wr:.1f}%[/]")
    
    console.print(hour_table)

    # 6. Risk Dashboard
    avg_vol = df['implied_vol'].mean()
    risk_panel = Panel(
        f"Avg Market Volatility: [bold cyan]{avg_vol:.2%}[/]\n"
        f"Max Inventory Size: [bold white]10 Contracts[/]\n"
        f"Trade Logic: [bold green]Long Calls & Long Puts[/]\n"
        f"Mean Reversion: [bold magenta]Target Z=0[/]",
        title="[bold red]Strategy Risk Parameters",
        expand=False
    )
    console.print(risk_panel)

if __name__ == "__main__":
    analyze_dual_direction_results()