import os
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from src.backtest import Backtester
from src.data_loader import DataLoader
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.random_forest import RandomForestStrategy
from src.reporting import Reporting

# Configure logger
from src.logger import setup_logger
setup_logger()

def main():
    data_loader = DataLoader()
    data_path = "data/BTCUSD_Candlestick_1_D_BID_03.08.2022-03.08.2024.csv"
    try:
        df = data_loader.load_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    

    backtester = Backtester(initial_cash=10000, transaction_cost=0.001, train_size=0.7)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up run directory structure

    run_dir = os.path.join("results", timestamp)
    # All subfolders/logs should be inside the current run folder
    train_dir = os.path.join(run_dir, "train")
    strategies_dir = os.path.join(run_dir, "strategies")
    logs_dir = run_dir
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(strategies_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    strategies = [
        MovingAverageStrategy(name="MA_Strategy", short_window=10, long_window=30),
        RandomForestStrategy(name="RF_Strategy", n_estimators=100, max_depth=5, output_dir=train_dir)
    ]

    # Set up single log file for the entire run (inside run_dir/logs)
    log_file = os.path.join(logs_dir, f"run_{timestamp}.log")
    setup_logger(log_file=log_file)

    reporting = Reporting()
    all_returns = {}
    all_benchmarks = None

    for strategy in strategies:
        strategy_dir = os.path.join(strategies_dir, strategy.name)
        os.makedirs(strategy_dir, exist_ok=True)

        logger.info(f"Running backtest for {strategy.name}")
        try:
            logger.info(f"Starting backtest.run for {strategy.name}")

            portfolio, test_df = backtester.run(df, strategy)
            logger.info(f"Backtest completed for {strategy.name}")

            # Debug portfolio stats
            logger.info(f"Portfolio total trades: {portfolio.trades.count()} | Total return: {portfolio.total_return():.2%}")
            trades_df = portfolio.trades.records_readable
            logger.info(f"First 5 trades:\n{trades_df.head()}\n")

            # Save reports in strategy folder
            logger.info(f"Generating report for {strategy.name}")
            returns = portfolio.returns()
            benchmark_returns = test_df["Close"].pct_change().fillna(0)
            reporting.generate_report(returns, benchmark_returns, strategy_dir, strategy_name=strategy.name)

            # Save executed trades to Excel
            reporting.save_trades_to_excel(trades_df, strategy_dir, strategy_name=strategy.name)

            # Save returns for comparison
            all_returns[strategy.name] = returns
            if all_benchmarks is None:
                all_benchmarks = benchmark_returns

        except Exception as e:
            logger.error(f"Error processing {strategy.name}: {e}")

    # Add Buy and Hold to comparison
    if all_benchmarks is not None:
        all_returns["Buy and Hold"] = all_benchmarks

    # Generate multi-strategy comparison report (HTML only, no Excel)
    if len(all_returns) > 2:
        try:
            logger.info("Generating multi-strategy HTML report")
            multi_report_path = os.path.join(run_dir, "multi_strategy_report.html")
            reporting.generate_multi_report(all_returns, multi_report_path)
            logger.info(f"Multi-strategy HTML report saved at {multi_report_path}")
        except Exception as e:
            logger.error(f"Error generating multi-strategy report: {e}")

if __name__ == "__main__":
    main()