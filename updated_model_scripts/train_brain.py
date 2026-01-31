import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
from rich.console import Console
from rich.table import Table

console = Console()

def train():
    if not os.path.exists("sota_training_data.csv"):
        console.print("[red]No training data found.[/]")
        return

    df = pd.read_csv("sota_training_data.csv")
    if len(df) < 10:
        console.print("[red]Not enough data to train.[/]")
        return
    
    # UPDATED FEATURES: Added MACD_Z and RSI
    features = ['hurst', 'theta', 'z_score', 'macd_z', 'rsi', 'volatility', 'atr', 'hour']
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
    
    console.print(f"[yellow]Training on {len(X_train)} campaigns with Ensemble Features...[/]")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    # Financial Analysis
    test_indices = y_test.index
    test_pnl = df.loc[test_indices, 'pnl']
    model_trades = test_pnl[preds == 1]
    model_total_pnl = model_trades.sum()
    avg_trade_val = model_trades.mean() if len(model_trades) > 0 else 0
    
    table = Table(title="Real-World Ensemble Brain Audit")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold magenta")
    
    table.add_row("Model Accuracy", f"{acc:.1%}")
    table.add_row("Precision", f"{prec:.1%}")
    table.add_row("Projected P&L", f"${model_total_pnl:,.2f}")
    table.add_row("Avg P&L per Trade", f"${avg_trade_val:.2f}")
    
    console.print(table)
    console.print(f"Feature Importance:\n{pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(5)}")
    
    joblib.dump(model, "sota_brain.pkl")

if __name__ == "__main__":
    train()