import quantstats as qs
from loguru import logger
import pandas as pd
import os

class Reporting:
    def _make_compare_df(self, returns_dict):
        """
        Create a DataFrame comparing cumulative returns of all strategies.
        :param returns_dict: dict of {strategy_name: returns_series}
        :return: DataFrame with cumulative returns for each strategy
        """
        df = pd.DataFrame(returns_dict)
        cum_df = (1 + df).cumprod() - 1
        return cum_df
    def generate_multi_report(self, returns_dict, output_path: str, title: str = "Multi-Strategy Backtest Report"):
        """
        Combine returns from multiple strategies and save a single QuantStats HTML report.
        :param returns_dict: dict of {strategy_name: returns_series}
        :param output_path: path to save the HTML report
        :param title: report title
        """
        try:
            # Combine all returns into a DataFrame
            df = pd.DataFrame(returns_dict)
            
            # Skip report if all returns are zero
            if df.empty or (df == 0).all().all():
                logger.warning("No trades executed for any strategy. Skipping multi-strategy report.")
                return
                
            # Replace any NaN values with 0
            df = df.fillna(0)
            
            qs.reports.html(
                df,
                title=title,
                output=output_path,
                periods_per_year=252,
                match_dates=True,
                download_filename=output_path,
            )
            logger.info(f"Multi-strategy report generated at {output_path}")
        except Exception as e:
            logger.error(f"Error generating multi-strategy report: {e}")
            raise

    def generate_report(self, returns, benchmark_returns, output_dir: str, strategy_name: str = "Strategy", benchmark_name: str = "Buy and Hold"):
        """
        Generate a performance report for a single strategy
        :param returns: Series of strategy returns
        :param benchmark_returns: Series of benchmark returns
        :param output_dir: Directory to save the report
        :param strategy_name: Name of the strategy
        :param benchmark_name: Name of the benchmark
        """
        try:
            # Check if we have any non-zero returns to report
            if returns.empty or (returns == 0).all():
                logger.warning(f"No trades executed for {strategy_name}. Skipping report generation.")
                return
                
            # Create report directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
                
            # Combine returns into a DataFrame, handling NaN values
            returns = returns.fillna(0)
            benchmark_returns = benchmark_returns.fillna(0)
            df = pd.DataFrame({strategy_name: returns, benchmark_name: benchmark_returns})
            
            # Generate the report
            output_file = os.path.join(output_dir, f"{strategy_name}.html")
            qs.reports.html(
                df,
                title=f"{strategy_name} Backtest Report",
                output=output_file,
                periods_per_year=252,
                match_dates=True,
                download_filename=output_file,
            )
            logger.info(f"Report generated at {output_file}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def save_trades_to_excel(self, trades_df, output_dir: str, strategy_name: str = "Strategy"):
        """
        Save all executed trades (from vectorbt) to an Excel file in the strategy folder.
        :param trades_df: DataFrame of executed trades (from vectorbt)
        :param output_dir: Directory to save the Excel file
        :param strategy_name: Name of the strategy
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            if trades_df is None or trades_df.empty:
                logger.warning(f"No trades to save for {strategy_name}.")
                return
            output_file = os.path.join(output_dir, f"{strategy_name}_trades.xlsx")
            trades_df.to_excel(output_file, index=False)
            logger.info(f"Trades saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving trades to Excel: {e}")
            raise
