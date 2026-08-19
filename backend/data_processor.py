import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class DataProcessor:
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
    def load_california_housing(self) -> tuple:
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name="Price")
        y = y * 100000
        
        return X, y, housing.feature_names, housing.DESCR
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['rooms_per_household'] = X['AveRooms']
        X['bedrooms_per_household'] = X['AveBedrms']
        X['population_per_household'] = X['Population']
        X['coastal_proximity'] = (X['Latitude'] > 37.5).astype(int)
        
        return X
    
    def save_scaler(self, scaler: StandardScaler, filename: str):
        path = self.data_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
    
    def load_scaler(self, filename: str) -> StandardScaler:
        path = self.data_dir / filename
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_feature_statistics(X: pd.DataFrame) -> dict:
        stats = {
            'features': X.columns.tolist(),
            'means': X.mean().to_dict(),
            'stds': X.std().to_dict(),
            'mins': X.min().to_dict(),
            'maxs': X.max().to_dict(),
        }
        return stats
