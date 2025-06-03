import pandas as pd
from abc import ABC, abstractmethod
from loguru import logger

class Strategy(ABC):
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> dict:
        pass

    def get_params(self):
        return self.params

class MovingAverageStrategy(Strategy):
    def __init__(self, name: str, short_window: int, long_window: int):
        super().__init__(name, short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window    
        
    def generate_signals(self, data: pd.DataFrame):
        # Create signals using the DataFrame's index
        short_ma = data["Close"].rolling(window=self.short_window, min_periods=1).mean()
        long_ma = data["Close"].rolling(window=self.long_window, min_periods=1).mean()
        signals = pd.Series(index=data.index, dtype=int)
        signals = signals.fillna(0)
        signals = signals.where(short_ma <= long_ma, 1)  # Long signal
        signals = signals.where(short_ma >= long_ma, -1)  # Short signal
        logger.info(f"[DEBUG] Signal value counts (pandas): {signals.value_counts().to_dict()}")

        prev = signals.shift(1, fill_value=0)
        long_entry = ((signals == 1) & (prev != 1)).astype(bool)
        long_exit = ((signals != 1) & (prev == 1)).astype(bool)
        short_entry = ((signals == -1) & (prev != -1)).astype(bool)
        short_exit = ((signals != -1) & (prev == -1)).astype(bool)

        # Ensure all outputs are Series with the same index as data
        long_entry = pd.Series(long_entry.values, index=data.index)
        long_exit = pd.Series(long_exit.values, index=data.index)
        short_entry = pd.Series(short_entry.values, index=data.index)
        short_exit = pd.Series(short_exit.values, index=data.index)

        return {
            'long_entry': long_entry,
            'long_exit': long_exit,
            'short_entry': short_entry,
            'short_exit': short_exit,
            'raw_signal': signals
        }