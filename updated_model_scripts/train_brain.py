import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from rich.console import Console

console = Console()

def train():
    if not os.path.exists("sota_training_data.csv"):
        console.print("[red]No data found. Run backtest_engine.py first.[/]")
        return

    df = pd.read_csv("sota_training_data.csv")
    
    # New Features based on Physics
    X = df[['hurst', 'theta', 'z_score', 'volatility', 'hour']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.01,
        eval_metric='logloss'
    )
    
    console.print("[yellow]Training XGBoost on Physics Features...[/]")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    console.print(f"[green]Model Trained. Accuracy: {acc:.1%}[/]")
    console.print(f"Feature Importance:\n{pd.Series(model.feature_importances_, index=X.columns)}")
    
    joblib.dump(model, "sota_brain.pkl")

if __name__ == "__main__":
    train()