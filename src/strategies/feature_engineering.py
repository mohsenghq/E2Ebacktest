import pandas as pd
from typing import Callable, Dict, List

class FeatureEngineer:
    def __init__(self):
        self.feature_funcs: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def register(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        """Register a new feature function."""
        self.feature_funcs[name] = func

    def generate(self, data: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """Generate selected features (or all if features=None) and return as DataFrame."""
        feats = features if features is not None else list(self.feature_funcs.keys())
        feature_data = {}
        for name in feats:
            if name not in self.feature_funcs:
                raise ValueError(f"Feature '{name}' not registered.")
            feature_data[name] = self.feature_funcs[name](data)
        return pd.DataFrame(feature_data, index=data.index)

# Example built-in features
def returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change()


# Generic moving average feature generator
def make_ma(window: int):
    def ma(df: pd.DataFrame) -> pd.Series:
        return df["Close"].rolling(window=window, min_periods=1).mean()
    ma.__name__ = f"ma{window}"
    return ma

# Helper to register multiple moving averages
def register_mas(feature_engineer, windows):
    for w in windows:
        feature_engineer.register(f"ma{w}", make_ma(w))

# Instantiate and register default features
feature_engineer = FeatureEngineer()
feature_engineer.register("returns", returns)
register_mas(feature_engineer, [10, 30, 50])  # Easily add more MAs here
