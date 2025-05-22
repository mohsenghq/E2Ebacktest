from datetime import datetime
import polars as pl
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

    def train(self, data: pl.DataFrame):
        logger.info(f"Training {self.name}")
        features = data.with_columns(
            pl.col("Close").pct_change().alias("returns"),
            pl.col("Close").rolling_mean(10).alias("ma10"),
            pl.col("Close").rolling_mean(30).alias("ma30")
        ).drop_nulls()
        X = features.select(["returns", "ma10", "ma30"]).to_numpy()
        y = (features["returns"] > 0).cast(int).to_numpy()
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.model.fit(X, y)
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(self.model, f"{self.output_dir}/model.pkl")
        logger.info(f"Model saved to {self.output_dir}/model.pkl")

    def generate_signals(self, data: pl.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained")
        features = data.with_columns(
            pl.col("Close").pct_change().alias("returns"),
            pl.col("Close").rolling_mean(10).alias("ma10"),
            pl.col("Close").rolling_mean(30).alias("ma30")
        ).select(["returns", "ma10", "ma30"]).to_numpy()
        predictions = self.model.predict(features)
        signals = pl.Series([1 if x == 1 else -1 for x in predictions])

        prev = signals.shift(1, fill_value=0)
        long_entry = (signals == 1) & (prev != 1)
        long_exit = (signals != 1) & (prev == 1)
        short_entry = (signals == -1) & (prev != -1)
        short_exit = (signals != -1) & (prev == -1)

        return {
            'long_entry': long_entry,
            'long_exit': long_exit,
            'short_entry': short_entry,
            'short_exit': short_exit,
            'raw_signal': signals
        }