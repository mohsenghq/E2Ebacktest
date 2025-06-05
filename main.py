import os
import sys
import glob
from datetime import datetime
from loguru import logger
from src.backtest import Backtester
from src.data_loader import DataLoader
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.random_forest import RandomForestStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.reporting import Reporting
from src.logger import setup_logger

setup_logger()

def process_single_file(data_path, base_run_dir, backtester, timestamp):
    data_loader = DataLoader()
    reporting = Reporting()
    
    # Extract file name without extension for the subfolder
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    run_dir = os.path.join(base_run_dir, file_name)
    train_dir = os.path.join(run_dir, "train")
    strategies_dir = os.path.join(run_dir, "strategies")
    logs_dir = run_dir
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(strategies_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    try:
        df = data_loader.load_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return {}

    all_returns = {}
    try:
        from src.strategies.feature_engineering import feature_engineer
        rf_features = ["returns", "ma10", "ma30", "rsi", "volatility", "open_close_diff", "high_low_diff", "macd", 
                      "bband_upper", "bband_lower", "stocastic_k", "cci", "atr", "ema20", "lag1", "rolling_mean10", 
                      "rolling_std10", "ewm_mean10", "day_of_week", "month", "quarter", "is_holiday", 
                      "open_close_diff", "high_low_diff"] + [f"lag{i}" for i in range(1, 101)]
        
        df["returns"] = df["Close"].pct_change()
        features_df = feature_engineer.generate(df, features=rf_features)
        features_df.index = df.index

        strategies = [
            MovingAverageStrategy(name="MA_Strategy", short_window=10, long_window=30),
            RandomForestStrategy(name="RF_Strategy", n_estimators=100, max_depth=5, output_dir=train_dir),
            XGBoostStrategy(name="XGB_Strategy", n_estimators=100, max_depth=5, 
                          output_dir=os.path.join(train_dir, "xgboost"))
        ]

        # Set up log file for this specific data file
        log_file = os.path.join(logs_dir, f"run_{timestamp}_{file_name}.log")
        setup_logger(log_file=log_file)

        reporting = Reporting()
        all_returns = {}
        all_benchmarks = None

        for strategy in strategies:
            strategy_dir = os.path.join(strategies_dir, strategy.name)
            os.makedirs(strategy_dir, exist_ok=True)

            logger.info(f"Running backtest for {strategy.name} on {file_name}")
            try:
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
                else:
                    portfolio, test_df = backtester.run(df, strategy)

                logger.info(f"Backtest completed for {strategy.name} on {file_name}")
                logger.info(f"Portfolio total trades: {portfolio.trades.count()} | Total return: {portfolio.total_return():.2%}")
                
                trades_df = portfolio.trades.records_readable
                logger.info(f"First 5 trades:\n{trades_df.head()}\n")
                
                returns = portfolio.returns()
                benchmark_returns = test_df["Close"].pct_change().fillna(0)
                reporting.generate_report(returns, benchmark_returns, strategy_dir, strategy_name=strategy.name)
                reporting.save_trades_to_excel(trades_df, strategy_dir, strategy_name=strategy.name)
                
                all_returns[strategy.name] = returns
                if all_benchmarks is None:
                    all_benchmarks = benchmark_returns
                    
            except Exception as e:
                logger.error(f"Error processing {strategy.name} for {file_name}: {e}")

        if all_benchmarks is not None:
            all_returns["Buy and Hold"] = all_benchmarks
        if len(all_returns) > 2:
            try:
                logger.info(f"Generating multi-strategy HTML report for {file_name}")
                multi_report_path = os.path.join(run_dir, f"multi_strategy_report_{file_name}.html")
                reporting.generate_multi_report(all_returns, multi_report_path)
                logger.info(f"Multi-strategy HTML report saved at {multi_report_path}")
            except Exception as e:
                logger.error(f"Error generating multi-strategy report for {file_name}: {e}")
        
        return all_returns
                
    except Exception as e:
        logger.error(f"Fatal error in processing {file_name}: {e}")
        return {}

def main(data_files):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_run_dir = os.path.join("results", timestamp)
    os.makedirs(base_run_dir, exist_ok=True)

    # Validate all files exist before processing
    for file_path in data_files:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

    backtester = Backtester(initial_cash=10000, transaction_cost=0.001, train_size=0.7)
    reporting = Reporting()
    all_file_returns = {}
    
    for data_path in data_files:
        logger.info(f"Processing file: {data_path}")
        file_returns = process_single_file(data_path, base_run_dir, backtester, timestamp)
        if file_returns:
            file_name = os.path.splitext(os.path.basename(data_path))[0]
            all_file_returns[file_name] = file_returns
    
    # Generate summary report if we have results
    if all_file_returns:
        reporting.save_summary_report(all_file_returns, base_run_dir)

if __name__ == "__main__":
    # Example usage:
    files = glob.glob('datasmall/*.csv')
    main(files)
