import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def train():
    if not os.path.exists("sota_training_data.csv"):
        console.print("[red]No training data found.[/]")
        return

    df = pd.read_csv("sota_training_data.csv")
    if len(df) < 10:
        console.print("[red]Not enough data to train.[/]")
        return
    
    # UPDATED FEATURES: Added 'regime'
    features = ['regime', 'hurst', 'theta', 'z_score', 'macd_z', 'rsi', 'dist_vwap', 'adx', 'vol_z', 'volatility', 'atr', 'hour']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1]))
    )
    
    console.print(f"[yellow]Training on {len(X_train)} campaigns...[/]")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # --- METRICS ---
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    
    # Financial Impact Analysis
    test_indices = y_test.index
    test_pnl = df.loc[test_indices, 'pnl']
    
    # What if we only took trades the model predicted as 1 (Win)?
    model_trades = test_pnl[preds == 1]
    model_total_pnl = model_trades.sum()
    avg_trade_val = model_trades.mean() if len(model_trades) > 0 else 0
    
    # Human Baseline (Take every trade)
    human_total_pnl = test_pnl.sum()
    
    # --- REPORTING ---
    console.print(Panel("[bold cyan]XGBoost Brain Diagnosis[/]", border_style="cyan"))
    
    # 1. Classification Metrics
    table = Table(title="ML Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold magenta")
    table.add_column("Meaning", style="dim white")
    
    table.add_row("Accuracy", f"{acc:.1%}", "Correct Guesses (Win & Loss)")
    table.add_row("Precision", f"{prec:.1%}", "Trustworthiness (If AI says Buy, is it a win?)")
    table.add_row("Recall", f"{rec:.1%}", "Opportunity Capture (Did we miss wins?)")
    table.add_row("F1-Score", f"{f1:.1%}", "Balance of Precision/Recall")
    console.print(table)
    
    # 2. Confusion Matrix
    tn, fp, fn, tp = cm.ravel()
    console.print(f"\n[bold white]Confusion Matrix:[/]")
    console.print(f"       [green]Pred Win[/]   [red]Pred Loss[/]")
    console.print(f"[green]Act Win[/]   {tp:<8}   {fn}")
    console.print(f"[red]Act Loss[/]  {fp:<8}   {tn}")
    console.print("[italic dim](FP = Bad trades taken, FN = Good trades missed)[/]\n")
    
    # 3. Financial Audit
    fin_table = Table(title="Financial Impact Audit (Test Set)")
    fin_table.add_column("Strategy", style="cyan")
    fin_table.add_column("Total P&L", style="bold green")
    fin_table.add_column("Avg P&L", style="bold white")
    
    fin_table.add_row("Raw Strategy (Take All)", f"${human_total_pnl:,.2f}", f"${test_pnl.mean():.2f}")
    fin_table.add_row("AI Filtered (Take Pred=1)", f"${model_total_pnl:,.2f}", f"${avg_trade_val:.2f}")
    
    console.print(fin_table)
    
    console.print(f"Top Features:\n{pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(3)}")
    
    joblib.dump(model, "sota_brain.pkl")

if __name__ == "__main__":
    train()