from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from loguru import logger

from .moving_average import Strategy

class RandomForestStrategy(Strategy):
    def __init__(self, name: str, n_estimators: int, max_depth: int, output_dir: str = "models"):
        super().__init__(name, n_estimators=n_estimators, max_depth=max_depth, output_dir=output_dir)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.output_dir = output_dir

    def train(self, features: pd.DataFrame):
        logger.info(f"Training {self.name}")
        features = features.dropna()
        
        # Save just the model parameters
        self.feature_columns = [col for col in features.columns if col != "Date" and col != "returns"]
        X = features[self.feature_columns].values
        y = (features["returns"] > 0).astype(int).values
        
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.model.fit(X, y)
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(self.model, f"{self.output_dir}/model.pkl")
        logger.info(f"Model saved to {self.output_dir}/model.pkl")

    def generate_signals(self, features: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained")
            
        features = features.fillna(0)
        X = features[self.feature_columns].values
        predictions = self.model.predict(X)
        signals = pd.Series([-1 if x == 0 else 1 for x in predictions], index=features.index)

        prev = signals.shift(1, fill_value=0)
        long_entry = ((signals == 1) & (prev != 1)).astype(bool)
        long_exit = ((signals != 1) & (prev == 1)).astype(bool)
        short_entry = ((signals == -1) & (prev != -1)).astype(bool)
        short_exit = ((signals != -1) & (prev == -1)).astype(bool)

        # Ensure all outputs are Series with the same index as features
        long_entry = pd.Series(long_entry.values, index=features.index)
        long_exit = pd.Series(long_exit.values, index=features.index)
        short_entry = pd.Series(short_entry.values, index=features.index)
        short_exit = pd.Series(short_exit.values, index=features.index)

        return {
            'long_entry': long_entry,
            'long_exit': long_exit,
            'short_entry': short_entry,
            'short_exit': short_exit,
            'raw_signal': signals
        }