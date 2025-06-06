import quantstats as qs
from loguru import logger
import pandas as pd
import os
import numpy as np

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
            trades_df = trades_df.rename(columns={'Entry Timestamp': 'Entry Time','Avg Entry Price': 'Entry Price', 'Exit Timestamp': 'Exit Time'})
            columns_to_keep = self.get_columns_to_keep(trades_df)
            trades_df = trades_df[columns_to_keep]
            
            # Convert timezone-aware datetime columns to timezone-naive
            if 'Entry Time' in trades_df.columns:
                trades_df['Entry Time'] = trades_df['Entry Time'].dt.tz_localize(None)
            if 'Exit Time' in trades_df.columns:
                trades_df['Exit Time'] = trades_df['Exit Time'].dt.tz_localize(None)
                
            trades_df.to_excel(output_file, index=False)
            logger.info(f"Trades saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving trades to Excel: {e}")
            raise

    def get_columns_to_keep(self, trades_df):
        """
        Get the columns to keep in the trades DataFrame.
        """
        good_columns = ['Entry Time', 'Size', 'Entry Price', 'Exit Time', 'PnL', 'Entry Fees', 'Exit Price', 'Return', 'Direction']
        return [col for col in trades_df.columns if col in good_columns]    
    
    def save_summary_report(self, all_returns: dict, output_dir: str):
        """
        Generate summary Excel file and bar chart comparing strategy performance across files.
        :param all_returns: dict of {file_name: {strategy_name: returns_series}}
        :param output_dir: Directory to save the summary reports
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create summary dataframe
            summary_data = []
            for file_name, returns_dict in all_returns.items():
                file_metrics = {'File': file_name}
                for strategy_name, returns in returns_dict.items():
                    if returns is not None and len(returns) > 0:
                        # Calculate key metrics
                        total_return = (1 + returns).cumprod().iloc[-1] - 1
                        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
                        # Calculate max drawdown
                        cum_returns = (1 + returns).cumprod()
                        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
                        # Calculate win rate and sortino ratio
                        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
                        neg_returns = returns[returns < 0]
                        sortino = np.sqrt(252) * returns.mean() / neg_returns.std() if len(neg_returns) > 0 and neg_returns.std() != 0 else 0
                          # Store only Return and WinRate metrics
                        file_metrics[f'{strategy_name}_Return'] = total_return
                        file_metrics[f'{strategy_name}_WinRate'] = win_rate
                summary_data.append(file_metrics)
            summary_df = pd.DataFrame(summary_data)
            
            # Save to Excel with formatting
            excel_path = os.path.join(output_dir, "strategy_summary.xlsx")
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Performance Summary', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Performance Summary']
                
                # Add formatting
                percent_format = workbook.add_format({'num_format': '0.00%'})
                decimal_format = workbook.add_format({'num_format': '0.00'})
                
                # Apply formatting to columns
                for col_idx, col in enumerate(summary_df.columns):
                    if '_Return' in col or '_MaxDD' in col or '_WinRate' in col:
                        worksheet.set_column(col_idx+1, col_idx+1, 12, percent_format)
                    elif '_Sharpe' in col or '_Sortino' in col:
                        worksheet.set_column(col_idx+1, col_idx+1, 12, decimal_format)
                
                # Adjust first column width for file names
                worksheet.set_column(0, 0, 20)
                logger.info(f"Summary Excel report saved to {excel_path}")
            
            # Create visualizations
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                  # Set style to a default nice looking style
                plt.style.use('default')
                
                # 1. Returns Comparison Plot
                plt.figure(figsize=(15, 8))
                  # Get strategy names and clean them for lookup
                strategy_names = []
                for col in summary_df.columns:
                    if '_Return' in col:
                        # Get everything before _Return
                        strategy_name = col.split('_Return')[0]
                        strategy_names.append(strategy_name)
                strategy_names = list(set(strategy_names))  # Remove duplicates
                
                # Set up bar positions
                num_strategies = len(strategy_names)
                bar_width = 0.8 / num_strategies
                file_positions = np.arange(len(summary_df['File']))
                  # Plot bars for each strategy with enhanced styling
                for i, strategy in enumerate(strategy_names):
                    col_name = f'{strategy}_Return'
                    if col_name in summary_df.columns:
                        returns = summary_df[col_name]
                        position = file_positions + i * bar_width - (num_strategies-1) * bar_width/2
                        plt.bar(position, returns*100, width=bar_width, label=strategy, alpha=0.8)
                
                plt.xlabel('Instruments', fontsize=12)
                plt.ylabel('Total Return (%)', fontsize=12)
                plt.title('Strategy Performance Comparison', fontsize=14, pad=20)
                plt.xticks(file_positions, summary_df['File'], rotation=45, ha='right')
                plt.legend(title='Strategies', title_fontsize=12, fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                  # Save returns comparison plot
                returns_plot_path = os.path.join(output_dir, "returns_comparison.png")
                plt.savefig(returns_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                  # Create Win Rate comparison plot
                plt.figure(figsize=(15, 8))
                for i, strategy in enumerate(strategy_names):
                    col_name = f'{strategy}_WinRate'
                    if col_name in summary_df.columns:
                        win_rates = summary_df[col_name]
                        position = file_positions + i * bar_width - (num_strategies-1) * bar_width/2
                        plt.bar(position, win_rates*100, width=bar_width, label=strategy, alpha=0.8)
                
                plt.xlabel('Instruments', fontsize=12)
                plt.ylabel('Win Rate (%)', fontsize=12)
                plt.title('Strategy Win Rate Comparison', fontsize=14, pad=20)
                plt.xticks(file_positions, summary_df['File'], rotation=45, ha='right')
                plt.legend(title='Strategies', title_fontsize=12, fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save win rate comparison plot
                winrate_plot_path = os.path.join(output_dir, "winrate_comparison.png")
                plt.savefig(winrate_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Performance visualizations saved to {output_dir}")
                
            except Exception as e:
                logger.error(f"Error creating performance visualizations: {e}")
                
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise
