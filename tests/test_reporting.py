import pytest
import pandas as pd
import os
from src.reporting import Reporting

def make_returns():
    idx = pd.date_range('2024-01-01', periods=5, freq='D')
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=idx)
    benchmark = pd.Series([0.0, 0.01, -0.01, 0.02, 0.0], index=idx)
    return returns, benchmark

def test_generate_report(tmp_path):
    reporting = Reporting()
    returns, benchmark = make_returns()
    out_dir = tmp_path
    reporting.generate_report(returns, benchmark, out_dir, strategy_name="Test")
    html_path = os.path.join(out_dir, "Test.html")
    assert os.path.exists(html_path)

def test_generate_multi_report(tmp_path):
    reporting = Reporting()
    returns, benchmark = make_returns()
    all_returns = {"A": returns, "B": benchmark}
    out_path = os.path.join(tmp_path, "multi.html")
    reporting.generate_multi_report(all_returns, out_path)
    assert os.path.exists(out_path)

def test_save_trades_to_excel(tmp_path):
    reporting = Reporting()
    trades_df = pd.DataFrame({"col1": [1,2], "col2": [3,4]})
    out_dir = tmp_path
    reporting.save_trades_to_excel(trades_df, out_dir, strategy_name="Test")
    xlsx_path = os.path.join(out_dir, "Test_trades.xlsx")
    assert os.path.exists(xlsx_path)
