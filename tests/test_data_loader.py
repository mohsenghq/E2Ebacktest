import pytest
import pandas as pd
from src.data_loader import DataLoader
import os

def test_load_csv_valid(tmp_path):
    # Create a valid CSV file
    csv_content = """Date,Open,High,Low,Close,Volume\n2024-01-01,1,2,0.5,1.5,100\n2024-01-02,1.5,2.5,1,2,200\n"""
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    loader = DataLoader()
    df = loader.load_csv(str(file_path))
    assert list(df.columns) == ["Date","Open","High","Low","Close","Volume"]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])

def test_load_csv_missing_columns(tmp_path):
    # Missing 'Close' column
    csv_content = """Date,Open,High,Low,Volume\n2024-01-01,1,2,0.5,100\n"""
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_csv(str(file_path))

def test_load_csv_invalid_path():
    loader = DataLoader()
    with pytest.raises(Exception):
        loader.load_csv("nonexistent.csv")
