import polars as pl
import sqlite3
from loguru import logger

class DataLoader:
    def load_csv(self, file_path: str) -> pl.DataFrame:
        try:
            df = pl.read_csv(file_path)
            required_columns = ['Date','Open','High','Low','Close','Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("CSV missing required OHLCV columns")
            df = df.with_columns(
                pl.col("Date").str.to_datetime("%Y-%m-%d %H:%M:%S%z")
            )
            return df.sort("Date")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_sql(self, db_path: str, table_name: str) -> pl.DataFrame:
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            df = pl.read_database(query, conn)
            conn.close()
            required_columns = ['Date','Open','High','Low','Close','Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("SQL table missing required OHLCV columns")
            df = df.with_columns(pl.col("Date").cast(pl.Datetime))
            return df.sort("Date")
        except Exception as e:
            logger.error(f"Error loading SQL: {e}")
            raise