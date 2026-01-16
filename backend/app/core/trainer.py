import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, List

from app.schemas.profile import StrategyProfile

class ModelTrainer:
    def __init__(self, profile: StrategyProfile):
        self.profile = profile
        self.symbol = profile.symbol
        self.data_dir = "backend/app/data/cache"
        self.model_dir = "backend/app/data/brains"
        
        self.input_file = os.path.join(self.data_dir, f"{self.symbol}_training_data.csv")
        self.model_file = os.path.join(self.model_dir, f"{self.symbol}_model.pkl")
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self) -> Dict[str, Any]:
        """
        Trains the XGBoost classifier. Logic matches train_model.py.
        """
        if not os.path.exists(self.input_file):
            return {"error": f"Training data for {self.symbol} not found. Run backtest first."}

        # 1. Load Data
        df = pd.read_csv(self.input_file)
        
        if len(df) < 10:
            return {"error": f"Insufficient data ({len(df)} campaigns). Minimum 10 required for training."}

        # 2. Define Features
        features = ['atr', 'mins_open', 'macd_z', 'vol_z', 'implied_vol']
        X = df[features]
        y = df['target']

        # 3. Handle Class Imbalance
        num_wins = int(sum(y == 1))
        num_losses = int(sum(y == 0))
        
        if num_wins == 0 or num_losses == 0:
            return {"error": "Training data must contain both wins and losses."}
            
        scale_pos_weight = num_losses / num_wins

        # 4. Split and Train
        # Stratify to ensure equal class distribution in test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.02,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # 6. Feature Importance
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).to_dict()

        # 7. Directional Sensitivity (Calls vs Puts)
        X_test_with_target = X_test.copy()
        X_test_with_target['actual'] = y_test
        X_test_with_target['pred'] = y_pred
        
        calls_mask = X_test_with_target['macd_z'] < 0
        puts_mask = X_test_with_target['macd_z'] > 0
        
        sensitivity = {}
        if calls_mask.any():
            sensitivity["calls"] = float(accuracy_score(y_test[calls_mask], y_pred[calls_mask]))
        if puts_mask.any():
            sensitivity["puts"] = float(accuracy_score(y_test[puts_mask], y_pred[puts_mask]))

        # 8. Save Model
        joblib.dump(model, self.model_file)

        return {
            "status": "success",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(report.get('1', report.get('1.0', {})).get('precision', 0)),
                "recall": float(report.get('1', report.get('1.0', {})).get('recall', 0)),
                "f1_score": float(report.get('1', report.get('1.0', {})).get('f1-score', 0)),
                "confusion_matrix": cm
            },
            "feature_importance": importance,
            "sensitivity": sensitivity,
            "sample_size": {
                "total": len(df),
                "wins": num_wins,
                "losses": num_losses
            },
            "model_path": self.model_file
        }