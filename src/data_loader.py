import pandas as pd
import sqlite3
from loguru import logger

class DataLoader:
    def load_csv(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Date','Open','High','Low','Close','Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("CSV missing required OHLCV columns")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.sort_values("Date")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_sql(self, db_path: str, table_name: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            required_columns = ['Date','Open','High','Low','Close','Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("SQL table missing required OHLCV columns")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.sort_values("Date")
        except Exception as e:
            logger.error(f"Error loading SQL: {e}")
            raise