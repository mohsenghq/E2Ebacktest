import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sklearn.preprocessing import StandardScaler

class EnhancedTradingEnv(gym.Env):
    """
    Enhanced trading environment with:
    - Transaction costs
    - Advanced reward function
    - Data normalization
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, transaction_cost=0.001, initial_balance=10000):
        super().__init__()
        
        # Data preprocessing
        self.original_df = df.copy()
        self.scaler = StandardScaler()
        self.df = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)
        self.df = self.df.reset_index(drop=True)
        
        # Environment parameters
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: long, 2: short
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32
        )
        
        # Trading parameters
        self.position = 0  # 0: flat, 1: long, -1: short
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.transaction_cost = transaction_cost
        self.trade_size = 0.1  # 10% of balance per trade
        self.entry_price = None
        self.trades = []
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.entry_price = None
        self.trades = []
        return self._get_obs(), {}

    def _get_obs(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)
    
    def _get_current_price(self):
        # Get the current close price from original (unscaled) data
        return self.original_df.iloc[self.current_step]['Close']

    def _calculate_reward(self):
        if self.position == 0 or self.entry_price is None:
            return 0
            
        current_price = self._get_current_price()
        price_change = (current_price - self.entry_price)/self.entry_price
        
        if self.position == 1:  # long
            return price_change
        elif self.position == -1:  # short
            return -price_change
        return 0

    def step(self, action):
        done = False
        current_price = self._get_current_price()
        reward = 0
        info = {}

        # Calculate current portfolio value
        if self.position == 1:  # long
            self.net_worth = self.balance + (current_price / self.entry_price) * (self.balance * self.trade_size)
        elif self.position == -1:  # short
            self.net_worth = self.balance + (self.entry_price / current_price) * (self.balance * self.trade_size)
        else:
            self.net_worth = self.balance

        # Execute action
        if action == 1 and self.position != 1:  # Enter long
            # Close existing position if any
            if self.position == -1:
                self.balance *= (1 - self.transaction_cost)  # Pay transaction cost
            
            self.position = 1
            self.entry_price = current_price
            self.balance *= (1 - self.transaction_cost)  # Pay transaction cost
            
        elif action == 2 and self.position != -1:  # Enter short
            # Close existing position if any
            if self.position == 1:
                self.balance *= (1 - self.transaction_cost)  # Pay transaction cost
            
            self.position = -1
            self.entry_price = current_price
            self.balance *= (1 - self.transaction_cost)  # Pay transaction cost
            
        elif action == 0:  # Hold
            pass

        # Calculate reward
        reward = self._calculate_reward()
        
        # Add penalty for frequent trading
        if action != 0 and self.current_step > 0:
            reward -= 0.001  # Small penalty for trading
            
        # Update step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        # Info dictionary
        info = {
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'price': current_price
        }
        
        return self._get_obs(), reward, done, False, info

    def render(self):
        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}, Position: {self.position}, Net Worth: {self.net_worth:.2f}")

def main():
    # Setup run folders for training (match backtest structure)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trained_models')) #os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    run_dir = os.path.join(results_dir, timestamp)
    train_dir = os.path.join(run_dir, 'train')
    
    # log_file = os.path.join(run_dir, f"train_{timestamp}.log")

    # # Setup logger for this run
    # from src.logger import setup_logger
    # setup_logger(log_file=log_file)

    # Load and prepare data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'BTCUSD_Candlestick_1_D_BID_03.08.2022-03.08.2024.csv')
    data_path = os.path.abspath(data_path)
    df = pd.read_csv(data_path)
    # Select relevant features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features]

    # Create environment
    env = EnhancedTradingEnv(df, transaction_cost=0.001)
    check_env(env, warn=True)

    # Train RL agent
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    # Train for longer
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # Save model and info in run's train folder
    os.makedirs(train_dir, exist_ok=True)
    model.save(os.path.join(train_dir, 'enhanced_model.zip'))
    with open(os.path.join(train_dir, 'info.txt'), 'w') as f:
        f.write(f"Trained at: {timestamp}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Total timesteps: 1000000\n")
    print(f"Model and logs saved to {train_dir}")

if __name__ == "__main__":
    main()