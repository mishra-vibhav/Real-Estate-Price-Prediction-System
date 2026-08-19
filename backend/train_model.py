import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from config import MODEL_PARAMS, MODELS_DIR, DATA_DIR
from data_processor import DataProcessor


class ModelTrainer:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        self.processor = DataProcessor(DATA_DIR)
        
    def train(self) -> dict:
        print("Loading data...")
        X, y, feature_names, description = self.processor.load_california_housing()
        self.feature_names = feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        
        print("Processing features...")
        X = self.processor.preprocess_features(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Scaling data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training model...")
        self.model = xgb.XGBRegressor(**MODEL_PARAMS)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        
        self.training_metrics = {
            'r2_score': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': X.shape[1],
            'features': list(X.columns),
        }
        
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        self.training_metrics['cv_r2_mean'] = float(cv_scores.mean())
        self.training_metrics['cv_r2_std'] = float(cv_scores.std())
        
        self._print_metrics()
        return self.training_metrics
    
    def _print_metrics(self):
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"R² Score (Test):     {self.training_metrics['r2_score']:.4f}")
        print(f"RMSE:                ${self.training_metrics['rmse']:,.0f}")
        print(f"MAE:                 ${self.training_metrics['mae']:,.0f}")
        print(f"Cross-Val R² (mean): {self.training_metrics['cv_r2_mean']:.4f} ± {self.training_metrics['cv_r2_std']:.4f}")
        print("="*50 + "\n")
    
    def save_model(self):
        MODELS_DIR.mkdir(exist_ok=True)
        
        model_path = MODELS_DIR / "xgboost_model.pkl"
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        features_path = MODELS_DIR / "feature_names.pkl"
        metrics_path = MODELS_DIR / "training_metrics.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.training_metrics, f)
        
        print(f"Models saved to {MODELS_DIR}")


def train_and_save():
    trainer = ModelTrainer()
    trainer.train()
    trainer.save_model()
    return trainer.training_metrics


if __name__ == "__main__":
    train_and_save()
