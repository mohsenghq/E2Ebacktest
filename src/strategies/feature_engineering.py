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

import numpy as np
import pandas as pd
import holidays

# Example built-in features
def returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change()

def volatility(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["Close"].pct_change().rolling(window=window, min_periods=1).std()

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def bband_upper(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = df["Close"].rolling(window=window, min_periods=1).mean()
    std = df["Close"].rolling(window=window, min_periods=1).std()
    return ma + num_std * std

def bband_lower(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = df["Close"].rolling(window=window, min_periods=1).mean()
    std = df["Close"].rolling(window=window, min_periods=1).std()
    return ma - num_std * std

def stocastic_k(df: pd.DataFrame, window: int = 14) -> pd.Series:
    low_min = df["Low"].rolling(window=window, min_periods=1).min()
    high_max = df["High"].rolling(window=window, min_periods=1).max()
    return 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window=window, min_periods=1).mean()
    md = tp.rolling(window=window, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def ema(df: pd.DataFrame, span: int = 20) -> pd.Series:
    return df["Close"].ewm(span=span, adjust=False).mean()

def lagged_price(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    return df["Close"].shift(lag)

def rolling_mean(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["Close"].rolling(window=window, min_periods=1).mean()

def rolling_std(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["Close"].rolling(window=window, min_periods=1).std()

def ewm_mean(df: pd.DataFrame, span: int = 10) -> pd.Series:
    return df["Close"].ewm(span=span, adjust=False).mean()

def day_of_week(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["Date"]).dt.dayofweek if "Date" in df else df.index.to_series().dt.dayofweek

def month(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["Date"]).dt.month if "Date" in df else df.index.to_series().dt.month

def quarter(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["Date"]).dt.quarter if "Date" in df else df.index.to_series().dt.quarter

def is_holiday(df: pd.DataFrame, country: str = "US") -> pd.Series:
    if "Date" in df:
        dates = pd.to_datetime(df["Date"])
    else:
        dates = df.index.to_series()
    cal = holidays.country_holidays(country)
    return dates.apply(lambda d: d in cal).astype(int)

def open_close_diff(df: pd.DataFrame) -> pd.Series:
    return df["Open"] - df["Close"]

def high_low_diff(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]

def open_close_pct(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] - df["Close"]) / (df["Close"] + 1e-10)

def high_low_pct(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]) / (df["Low"] + 1e-10)

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

# Add multiple lagged close price features

def make_lag_feature(lag: int):
    def lag_func(df: pd.DataFrame) -> pd.Series:
        return df["Close"].shift(lag)
    lag_func.__name__ = f"lag{lag}"
    return lag_func

# Instantiate and register default features
feature_engineer = FeatureEngineer()

# Register lag1 to lag100
for i in range(1, 101):
    feature_engineer.register(f"lag{i}", make_lag_feature(i))


feature_engineer.register("returns", returns)
feature_engineer.register("volatility", volatility)
feature_engineer.register("rsi", rsi)
feature_engineer.register("macd", macd)
feature_engineer.register("bband_upper", bband_upper)
feature_engineer.register("bband_lower", bband_lower)
feature_engineer.register("stocastic_k", stocastic_k)
feature_engineer.register("cci", cci)
feature_engineer.register("atr", atr)
feature_engineer.register("ema20", lambda df: ema(df, 20))
feature_engineer.register("lag1", lambda df: lagged_price(df, 1))
feature_engineer.register("rolling_mean10", lambda df: rolling_mean(df, 10))
feature_engineer.register("rolling_std10", lambda df: rolling_std(df, 10))
feature_engineer.register("ewm_mean10", lambda df: ewm_mean(df, 10))
feature_engineer.register("day_of_week", day_of_week)
feature_engineer.register("month", month)
feature_engineer.register("quarter", quarter)
feature_engineer.register("is_holiday", is_holiday)
feature_engineer.register("open_close_diff", open_close_diff)
feature_engineer.register("high_low_diff", high_low_diff)
feature_engineer.register("open_close_pct", open_close_pct)
feature_engineer.register("high_low_pct", high_low_pct)
register_mas(feature_engineer, [10, 30, 50])  # Easily add more MAs here
