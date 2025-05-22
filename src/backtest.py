import polars as pl
import pandas as pd
import vectorbt as vbt
from loguru import logger
import numpy as np

class Backtester:
    def __init__(self, initial_cash: float = 10000, transaction_cost: float = 0.001, train_size: float = 0.7):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.train_size = train_size

    def run(self, data: pl.DataFrame, strategy):
        data = data.with_columns(pl.col("Date").cast(pl.Datetime))
        data = data.sort("Date")

        if hasattr(strategy, "train"):
            n = len(data)
            train_end = int(n * self.train_size)
            train_data = data[:train_end]
            test_data = data[train_end:]
            strategy.train(train_data)
            signals_dict = strategy.generate_signals(test_data)
            test_df = test_data.to_pandas().set_index("Date")
        else:
            n = len(data)
            train_end = int(n * self.train_size)
            test_data = data[train_end:]
            signals_dict = strategy.generate_signals(test_data)
            test_df = test_data.to_pandas().set_index("Date")

        # Convert signals to pandas Series with proper index
        def to_pd(series):
            if isinstance(series, pl.Series):
                return series.to_pandas()
            elif isinstance(series, np.ndarray):
                return pd.Series(series, index=test_df.index)
            return series

        long_entry = to_pd(signals_dict['long_entry'])
        long_exit = to_pd(signals_dict['long_exit'])
        short_entry = to_pd(signals_dict['short_entry'])
        short_exit = to_pd(signals_dict['short_exit'])
        raw_signal = to_pd(signals_dict['raw_signal'])
        logger.info(f"[DEBUG] test_df length: {len(test_df)}, raw_signal length: {len(raw_signal)}")
        logger.info(f"[DEBUG] raw_signal value counts (pandas): {raw_signal.value_counts().to_dict() if hasattr(raw_signal, 'value_counts') else 'N/A'}")

        # Run VectorBT portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=test_df["Close"],
            entries=long_entry,
            exits=long_exit,
            short_entries=short_entry,
            short_exits=short_exit,
            init_cash=self.initial_cash,
            fees=self.transaction_cost,
            freq='1D'
        )

        return portfolio, test_df