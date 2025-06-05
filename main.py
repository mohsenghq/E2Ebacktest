import os
import shutil

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from src.backtest import Backtester
from src.data_loader import DataLoader
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.random_forest import RandomForestStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.reporting import Reporting

# Configure logger
from src.logger import setup_logger
setup_logger()

def main():
    data_loader = DataLoader()
    data_path = "data/BTCUSD_Candlestick_1_D_BID_03.08.2022-03.08.2024.csv"
    run_dir = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        df = data_loader.load_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    backtester = Backtester(initial_cash=10000, transaction_cost=0.001, train_size=0.7)
    run_dir = os.path.join("results", timestamp)
    train_dir = os.path.join(run_dir, "train")
    strategies_dir = os.path.join(run_dir, "strategies")
    logs_dir = run_dir
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(strategies_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    try:
        from src.strategies.feature_engineering import feature_engineer
        # Define features for RandomForest (expand as needed)
        rf_features = ["returns", "ma10", "ma30", "rsi", "volatility", "open_close_diff", "high_low_diff", "macd", "bband_upper", "bband_lower", "stocastic_k", "cci", "atr", "ema20", "lag1", "rolling_mean10", "rolling_std10", "ewm_mean10", "day_of_week", "month", "quarter", "is_holiday", "open_close_diff", "high_low_diff"]# + [f"lag{i}" for i in range(1, 101)]
        # Ensure 'returns' is present in df before feature generation
        df["returns"] = df["Close"].pct_change()
        features_df = feature_engineer.generate(df, features=rf_features)
        features_df.index = df.index
        features_df.index = df.index
        strategies = [
            MovingAverageStrategy(name="MA_Strategy", short_window=10, long_window=30),
            RandomForestStrategy(name="RF_Strategy", n_estimators=100, max_depth=5, output_dir=train_dir),
            XGBoostStrategy(name="XGB_Strategy", n_estimators=100, max_depth=5, output_dir=os.path.join(train_dir, "xgboost"))
            # RLStrategy(name="PPO_Simple", output_dir=os.path.join(train_dir, "ppo_simple"), reward_function="simple", model_type="PPO"),
            # RLStrategy(name="DQN_Simple", output_dir=os.path.join(train_dir, "dqn_simple"), reward_function="simple", model_type="DQN"),
            # RLStrategy(name="PPO_Sharpe", output_dir=os.path.join(train_dir, "ppo_sharpe"), reward_function="sharpe", model_type="PPO"),
            # RLStrategy(name="DQN_Sharpe", output_dir=os.path.join(train_dir, "dqn_sharpe"), reward_function="sharpe", model_type="DQN"),
            # RLStrategy(name="PPO_RiskAdjusted", output_dir=os.path.join(train_dir, "ppo_risk_adjusted"), reward_function="risk_adjusted", model_type="PPO"),
            # RLStrategy(name="DQN_RiskAdjusted", output_dir=os.path.join(train_dir, "dqn_risk_adjusted"), reward_function="risk_adjusted", model_type="DQN"),
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
                if isinstance(strategy, RandomForestStrategy):
                    n = len(features_df)
                    train_end = int(n * backtester.train_size)
                    train_features = features_df.iloc[:train_end]
                    test_features = features_df.iloc[train_end:].copy()
                    test_features.index = df.iloc[train_end:].set_index("Date").index if "Date" in df.columns else test_features.index
                    strategy.train(train_features)
                    test_df = df.iloc[train_end:].copy()
                    test_df.set_index("Date", inplace=True)
                    signals_dict = strategy.generate_signals(test_features)
                    portfolio, test_df = backtester.run(df, strategy)
                    portfolio, test_df = backtester.run(df, strategy)
                else:
                    portfolio, test_df = backtester.run(df, strategy)

                logger.info(f"Backtest completed for {strategy.name}")
                logger.info(f"Portfolio total trades: {portfolio.trades.count()} | Total return: {portfolio.total_return():.2%}")
                trades_df = portfolio.trades.records_readable
                logger.info(f"First 5 trades:\n{trades_df.head()}\n")
                logger.info(f"Generating report for {strategy.name}")
                returns = portfolio.returns()
                benchmark_returns = test_df["Close"].pct_change().fillna(0)
                reporting.generate_report(returns, benchmark_returns, strategy_dir, strategy_name=strategy.name)
                reporting.save_trades_to_excel(trades_df, strategy_dir, strategy_name=strategy.name)
                all_returns[strategy.name] = returns
                if all_benchmarks is None:
                    all_benchmarks = benchmark_returns
            except Exception as e:
                logger.error(f"Error processing {strategy.name}: {e}")

        if all_benchmarks is not None:
            all_returns["Buy and Hold"] = all_benchmarks
        if len(all_returns) > 2:
            try:
                logger.info("Generating multi-strategy HTML report")
                multi_report_path = os.path.join(run_dir, "multi_strategy_report.html")
                reporting.generate_multi_report(all_returns, multi_report_path)
                logger.info(f"Multi-strategy HTML report saved at {multi_report_path}")
            except Exception as e:
                logger.error(f"Error generating multi-strategy report: {e}")
    except Exception as e:
        logger.error(f"Fatal error in main run: {e}")
        # raise
    # finally:
    #     if run_dir and os.path.exists(run_dir):
    #         try:
    #             from loguru import logger as loguru_logger
    #             loguru_logger.remove()  # Remove all handlers to close log files
    #             shutil.rmtree(run_dir)
    #             print(f"Deleted run directory due to error: {run_dir}")
    #         except Exception as cleanup_error:
    #             print(f"Failed to delete run directory: {cleanup_error}")
        # raise
    # finally:
    #     if run_dir and os.path.exists(run_dir):
    #         try:
    #             from loguru import logger as loguru_logger
    #             loguru_logger.remove()  # Remove all handlers to close log files
    #             shutil.rmtree(run_dir)
    #             print(f"Deleted run directory due to error: {run_dir}")
    #         except Exception as cleanup_error:
    #             print(f"Failed to delete run directory: {cleanup_error}")

if __name__ == "__main__":
    main()