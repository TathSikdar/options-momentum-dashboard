import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def train_options_model():
    """
    Trains an XGBoost classifier for Dual-Direction Mean Reversion.
    Learns to predict success for both Put (Overbought) and Call (Oversold) campaigns.
    """
    file_path = "historical_training_data.csv"
    
    if not os.path.exists(file_path):
        console.print(Panel("[bold red]Error:[/] Training data not found. Please run backtest.py first.", title="System Error"))
        return

    # 1. Load Data
    df = pd.read_csv(file_path)
    
    if len(df) < 5:
        console.print(Panel("[bold red]Error:[/] Not enough data points to train a model. Need at least 5 campaigns.", title="Data Scarcity"))
        return

    # 2. Define Features
    features = ['atr', 'mins_open', 'macd_z', 'vol_z', 'implied_vol']
    X = df[features]
    y = df['target']

    # 3. Dynamic Stratification Check
    # We check if we have at least 2 members of each class (0 and 1)
    # If not, stratify=y will fail, so we disable it.
    class_counts = y.value_counts()
    can_stratify = all(count >= 2 for count in class_counts) and len(class_counts) > 1

    if not can_stratify:
        console.print("[yellow]Warning: Low sample size or class imbalance detected. Disabling stratification.")
        stratify_param = None
    else:
        stratify_param = y

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # 5. Handle Class Imbalance
    # Avoid division by zero if there are no wins or no losses
    num_wins = sum(y == 1)
    num_losses = sum(y == 0)
    scale_pos_weight = num_losses / num_wins if num_wins > 0 else 1

    # 6. Model Configuration
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.02,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    console.print(f"[yellow]Training Dual-Direction Brain on {len(X_train)} campaigns...")
    model.fit(X_train, y_train)

    # 7. Performance Audit
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    # Use zero_division=0 to prevent errors if a class is missing in the test set
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Results UI
    audit_table = Table(title="[bold cyan]XGBoost Model Audit: Dual-Direction Scaling")
    audit_table.add_column("Metric", style="white")
    audit_table.add_column("Value", style="bold magenta")

    # Handle potentially missing keys in report
    target_key = '1' if '1' in report else ('1.0' if '1.0' in report else None)
    
    audit_table.add_row("Overall Accuracy", f"{accuracy:.2%}")
    if target_key:
        audit_table.add_row("Precision (Win Prediction Accuracy)", f"{report[target_key]['precision']:.2%}")
        audit_table.add_row("Recall (Wins Captured)", f"{report[target_key]['recall']:.2%}")
        audit_table.add_row("F1-Score", f"{report[target_key]['f1-score']:.4f}")
    else:
        audit_table.add_row("Stats Note", "Insufficient variety in test set for full metrics.")
    
    console.print(audit_table)

    # 8. Directional Sensitivity Analysis
    X_test_with_target = X_test.copy()
    X_test_with_target['actual'] = y_test
    X_test_with_target['pred'] = y_pred
    
    calls_mask = X_test_with_target['macd_z'] < 0
    puts_mask = X_test_with_target['macd_z'] > 0
    
    sense_table = Table(title="[bold yellow]Model Sensitivity by Trade Type")
    sense_table.add_column("Trade Type", style="cyan")
    sense_table.add_column("Accuracy", justify="right")
    
    if calls_mask.any():
        call_acc = accuracy_score(y_test[calls_mask], y_pred[calls_mask])
        sense_table.add_row("Calls (Oversold)", f"{call_acc:.2%}")
    if puts_mask.any():
        put_acc = accuracy_score(y_test[puts_mask], y_pred[puts_mask])
        sense_table.add_row("Puts (Overbought)", f"{put_acc:.2%}")
        
    console.print(sense_table)

    # 9. Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    imp_table = Table(title="[bold white]Feature Importance")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Weight", style="magenta")
    for feat, val in importance.items():
        imp_table.add_row(feat, f"{val:.4f}")
    console.print(imp_table)

    # 10. Save the Model
    joblib.dump(model, "amd_model.pkl")
    console.print(Panel(f"[bold green]Success![/] Model calibrated and saved as 'amd_model.pkl'\n"
                        f"Confusion Matrix: [ {cm[0][0]} {cm[0][1] if len(cm[0])>1 else 0} ] (TN, FP)\n"
                        f"                  [ {cm[1][0] if len(cm)>1 else 0} {cm[1][1] if len(cm)>1 else cm[0][0]} ] (FN, TP)", title="Model Export"))

if __name__ == "__main__":
    train_options_model()