import pytest
import pandas as pd
from src.strategies.moving_average import MovingAverageStrategy

def make_test_data():
    data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'Open': range(10),
        'High': range(1, 11),
        'Low': range(0, 10),
        'Close': range(2, 12),
        'Volume': [100]*10
    })
    data.set_index('Date', inplace=True)
    return data

def test_generate_signals():
    data = make_test_data()
    strategy = MovingAverageStrategy(name="TestMA", short_window=2, long_window=3)
    signals = strategy.generate_signals(data)
    assert isinstance(signals, dict)
    assert 'long_entry' in signals
    assert 'long_exit' in signals
    assert hasattr(signals['long_entry'], 'index')
    assert hasattr(signals['long_exit'], 'index')
