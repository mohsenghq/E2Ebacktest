import pytest
import pandas as pd
from src.backtest import Backtester
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
    return data

def test_backtester_run_ma():
    data = make_test_data()
    backtester = Backtester(initial_cash=10000, transaction_cost=0.001, train_size=0.7)
    strategy = MovingAverageStrategy(name="TestMA", short_window=2, long_window=3)
    portfolio, test_df = backtester.run(data, strategy)
    # Portfolio should have trades and returns
    assert hasattr(portfolio, 'trades')
    assert hasattr(portfolio, 'returns')
    returns = portfolio.returns()
    assert isinstance(returns, pd.Series)
    assert not returns.isnull().all()
    assert isinstance(test_df, pd.DataFrame)
    assert 'Close' in test_df.columns
