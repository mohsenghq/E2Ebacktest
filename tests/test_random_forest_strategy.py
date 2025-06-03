import pytest
import pandas as pd
from src.strategies.random_forest import RandomForestStrategy
import os

def make_test_data():
    data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=20, freq='D'),
        'Open': range(20),
        'High': range(1, 21),
        'Low': range(0, 20),
        'Close': range(2, 22),
        'Volume': [100]*20
    })
    data.set_index('Date', inplace=True)
    return data

def test_train_and_generate_signals(tmp_path):
    data = make_test_data()
    output_dir = tmp_path
    strategy = RandomForestStrategy(name="TestRF", n_estimators=10, max_depth=2, output_dir=str(output_dir))
    # Should not raise
    strategy.train(data)
    signals = strategy.generate_signals(data)
    assert isinstance(signals, dict)
    assert 'long_entry' in signals
    assert 'long_exit' in signals
