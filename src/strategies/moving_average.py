import polars as pl
from abc import ABC, abstractmethod
from loguru import logger

class Strategy(ABC):
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> dict:
        pass

    def get_params(self):
        return self.params

class MovingAverageStrategy(Strategy):
    def __init__(self, name: str, short_window: int, long_window: int):
        super().__init__(name, short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pl.DataFrame):
        short_ma = data["Close"].rolling_mean(self.short_window)
        long_ma = data["Close"].rolling_mean(self.long_window)
        # 1 for long, -1 for short, 0 for flat
        signals = (short_ma > long_ma).cast(int) - (short_ma < long_ma).cast(int)
        signals = signals.fill_null(0)
        logger.info(f"[DEBUG] Signal value counts (polars): {signals.value_counts()}")

        # Entry/exit logic
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