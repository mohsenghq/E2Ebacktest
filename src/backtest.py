import pandas as pd
import vectorbt as vbt
from loguru import logger
import numpy as np

class Backtester:
    def __init__(self, initial_cash: float = 10000, transaction_cost: float = 0.001, train_size: float = 0.7):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.train_size = train_size

    def run(self, data: pd.DataFrame, strategy):
        data = data.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")

        if hasattr(strategy, "train"):
            n = len(data)
            train_end = int(n * self.train_size)
            train_data = data.iloc[:train_end]            
            test_data = data.iloc[train_end:].copy()
            test_data.set_index("Date", inplace=True)
            strategy.train(train_data)
            signals_dict = strategy.generate_signals(test_data)
            test_df = test_data
        else:
            n = len(data)
            train_end = int(n * self.train_size)            
            test_data = data.iloc[train_end:].copy()
            test_data.set_index("Date", inplace=True)
            signals_dict = strategy.generate_signals(test_data)
            test_df = test_data

        def to_pd(series):
            if isinstance(series, np.ndarray):
                return pd.Series(series, index=test_df.index)
            return series

        long_entry = to_pd(signals_dict['long_entry'])
        long_exit = to_pd(signals_dict['long_exit'])
        short_entry = to_pd(signals_dict['short_entry'])
        short_exit = to_pd(signals_dict['short_exit'])
        raw_signal = to_pd(signals_dict['raw_signal'])
        logger.info(f"[DEBUG] test_df length: {len(test_df)}, raw_signal length: {len(raw_signal)}")
        logger.info(f"[DEBUG] raw_signal value counts (pandas): {raw_signal.value_counts().to_dict() if hasattr(raw_signal, 'value_counts') else 'N/A'}")
        # Debug dtypes and shapes
        # logger.info(f"long_entry dtype: {long_entry.dtype}, shape: {long_entry.shape}")
        # logger.info(f"long_exit dtype: {long_exit.dtype}, shape: {long_exit.shape}")
        # logger.info(f"short_entry dtype: {short_entry.dtype}, shape: {short_entry.shape}")
        # logger.info(f"short_exit dtype: {short_exit.dtype}, shape: {short_exit.shape}")
        # logger.info(f"raw_signal dtype: {raw_signal.dtype}, shape: {raw_signal.shape}")
        # Fill NaNs with False for boolean signals
        long_entry = long_entry.fillna(False)
        long_exit = long_exit.fillna(False)
        short_entry = short_entry.fillna(False)
        short_exit = short_exit.fillna(False)
        # Ensure all inputs are pandas Series with the same index
        close = test_df["Close"]
        long_entry = pd.Series(long_entry, index=close.index)
        long_exit = pd.Series(long_exit, index=close.index)
        short_entry = pd.Series(short_entry, index=close.index)
        short_exit = pd.Series(short_exit, index=close.index)
        # Convert to numpy arrays of correct type
        long_entry = np.asarray(long_entry, dtype=bool)
        long_exit = np.asarray(long_exit, dtype=bool)
        short_entry = np.asarray(short_entry, dtype=bool)
        short_exit = np.asarray(short_exit, dtype=bool)
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=long_entry,
            exits=long_exit,
            short_entries=short_entry,
            short_exits=short_exit,
            init_cash=self.initial_cash,
            fees=self.transaction_cost,
            freq='1D',
            broadcast_kwargs=dict(require_kwargs=dict(requirements='W'))  # Less strict broadcasting
        )

        return portfolio, test_df