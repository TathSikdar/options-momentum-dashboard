import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def color_pnl(value):
    """Returns a colored string for P&L values."""
    if value > 0:
        return f"[bold green]+${value:,.2f}[/]"
    elif value < 0:
        return f"[bold red]-${abs(value):,.2f}[/]"
    else:
        return f"[white]${value:,.2f}[/]"

def analyze_financial_results():
    """
    Analyzes backtest results with a focus on P&L (Profit & Loss).
    """
    file_path = "historical_training_data.csv"
    
    if not os.path.exists(file_path):
        console.print(Panel("[bold red]Error:[/] historical_training_data.csv not found. Please run backtest.py first.", title="System Error"))
        return

    df = pd.read_csv(file_path)
    
    # 1. Label Trade Types
    df['trade_type'] = df['macd_z'].apply(lambda x: 'Call (Oversold)' if x < 0 else 'Put (Overbought)')
    
    # 2. Financial Metrics Calculation
    total_trades = len(df)
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    
    # Profit Factor (Gross Profit / Gross Loss)
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown Calculation (Simple Peak-to-Valley)
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()

    # 3. Main Financial Summary Table
    summary_table = Table(title=f"[bold cyan]AMD Options Strategy: Financial Audit[/]")
    summary_table.add_column("Metric", style="white")
    summary_table.add_column("Value", justify="right", style="bold magenta")
    summary_table.add_column("Context", style="italic white")

    summary_table.add_row("Total P&L", color_pnl(total_pnl), "Net Profit across all trades")
    summary_table.add_row("Total Trades", str(total_trades), "Sample Size")
    summary_table.add_row("Win Rate", f"{win_rate:.1f}%", f"{len(wins)} Wins / {len(losses)} Losses")
    summary_table.add_row("Avg P&L per Trade", color_pnl(avg_pnl), "Expectancy")
    summary_table.add_row("Avg Win", color_pnl(avg_win), "Avg gain on winning trades")
    summary_table.add_row("Avg Loss", color_pnl(avg_loss), "Avg loss on losing trades")
    summary_table.add_row("Profit Factor", f"{profit_factor:.2f}", "> 1.5 is ideal")
    summary_table.add_row("Max Drawdown", color_pnl(max_drawdown), "Worst decline from equity peak")

    console.print(summary_table)

    # 4. Breakdown by Trade Type (Call vs Put)
    type_table = Table(title="[bold yellow]P&L Breakdown by Trade Direction[/]")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="center")
    type_table.add_column("Win Rate", justify="right")
    type_table.add_column("Total P&L", justify="right")
    type_table.add_column("Avg P&L", justify="right")

    for t_type in df['trade_type'].unique():
        sub = df[df['trade_type'] == t_type]
        sub_wr = (len(sub[sub['pnl'] > 0]) / len(sub)) * 100
        sub_total = sub['pnl'].sum()
        sub_avg = sub['pnl'].mean()
        
        type_table.add_row(
            t_type, 
            str(len(sub)), 
            f"{sub_wr:.1f}%", 
            color_pnl(sub_total), 
            color_pnl(sub_avg)
        )
    
    console.print(type_table)

    # 5. Top Winners and Losers
    top_wins = df.nlargest(3, 'pnl')
    top_losses = df.nsmallest(3, 'pnl')
    
    ext_table = Table(title="[bold white]Extreme Outliers (Best & Worst)[/]")
    ext_table.add_column("Rank", style="white")
    ext_table.add_column("Type", style="cyan")
    ext_table.add_column("P&L", justify="right")
    ext_table.add_column("Z-Score", justify="right")

    for i, (_, row) in enumerate(top_wins.iterrows()):
        ext_table.add_row(f"Best #{i+1}", row['trade_type'].split()[0], color_pnl(row['pnl']), f"{row['macd_z']:.2f}")
    
    ext_table.add_row("---", "---", "---", "---") # Separator

    for i, (_, row) in enumerate(top_losses.iterrows()):
        ext_table.add_row(f"Worst #{i+1}", row['trade_type'].split()[0], color_pnl(row['pnl']), f"{row['macd_z']:.2f}")

    console.print(ext_table)

    # 6. Conclusion Panel
    final_equity = df['cumulative_pnl'].iloc[-1] if not df.empty else 0
    console.print(Panel(
        f"Final Simulated Equity Change: {color_pnl(final_equity)}\n"
        f"Based on fixed contract scaling (Max 10).",
        title="Account Simulation", border_style="green" if final_equity > 0 else "red"
    ))

if __name__ == "__main__":
    analyze_financial_results()