import pandas as pd
import numpy as np
from loguru import logger
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from .moving_average import Strategy
from collections import deque

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio given returns and risk-free rate."""
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate Sortino ratio given returns and risk-free rate."""
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)  # Annualized

def calculate_calmar_ratio(returns, window=252):
    """Calculate Calmar ratio (returns / maximum drawdown)."""
    if len(returns) < window:
        return 0
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = abs(drawdowns.min())
    if max_drawdown == 0:
        return 0
    return np.mean(returns) / max_drawdown * np.sqrt(252)  # Annualized

class TradingEnv(gym.Env):
    """Custom Environment for trading with multiple reward functions"""
    def __init__(self, data: pd.DataFrame, features: pd.DataFrame, initial_balance=10000,
                 reward_function: str = "simple", window_size: int = 20):
        super(TradingEnv, self).__init__()

        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.n_features = features.shape[1]
        self.window_size = window_size
        self.returns_history = deque(maxlen=window_size)
        
        # Set reward function
        self.reward_function = reward_function
        self.reward_functions = {
            "simple": self._simple_reward,
            "sharpe": self._sharpe_reward,
            "sortino": self._sortino_reward,
            "calmar": self._calmar_reward,
            "asymmetric": self._asymmetric_reward,
            "risk_adjusted": self._risk_adjusted_reward
        }
        
        # Actions: 0 (buy/long), 1 (sell/short)
        self.action_space = spaces.Discrete(2)
        
        # Prices array of length n_features + position + balance + current price + returns history
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features + 3 + window_size,), 
            dtype=np.float32
        )    
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.prev_position = 0  # Track previous position
        self.done = False
        self.returns_history.clear()
        self.returns_history.extend([0.0] * self.window_size)
        return self._next_observation(), {}

    def _next_observation(self):
        features = self.features.iloc[self.current_step].values
        returns_history = np.array(self.returns_history)
        
        obs = np.concatenate([
            features,
            [self.position, self.balance, self.data.iloc[self.current_step]['Close']],
            returns_history
        ])
        return obs.astype(np.float32)
        
    def _simple_reward(self, price_change: float) -> float:
        """Simple reward based on price change and position"""
        return self.position * price_change

    def _sharpe_reward(self, price_change: float) -> float:
        """Reward based on Sharpe ratio"""
        self.returns_history.append(price_change * self.position)
        return calculate_sharpe_ratio(np.array(self.returns_history))

    def _sortino_reward(self, price_change: float) -> float:
        """Reward based on Sortino ratio"""
        self.returns_history.append(price_change * self.position)
        return calculate_sortino_ratio(np.array(self.returns_history))

    def _calmar_reward(self, price_change: float) -> float:
        """Reward based on Calmar ratio"""
        self.returns_history.append(price_change * self.position)
        return calculate_calmar_ratio(np.array(self.returns_history))

    def _asymmetric_reward(self, price_change: float) -> float:
        """Asymmetric reward function that penalizes losses more than rewards gains"""
        position_return = self.position * price_change
        return np.sign(position_return) * (abs(position_return) ** 0.5)

    def _risk_adjusted_reward(self, price_change: float) -> float:
        """Combination of simple return and risk metrics"""
        self.returns_history.append(price_change * self.position)
        simple_reward = self._simple_reward(price_change)
        sharpe = calculate_sharpe_ratio(np.array(self.returns_history))
        sortino = calculate_sortino_ratio(np.array(self.returns_history))
        
        return simple_reward * (1 + 0.1 * (sharpe + sortino))

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            self.done = True
            truncated = True
            return self._next_observation(), 0, self.done, truncated, {}

        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        price_change = (next_price - current_price) / current_price

        # Update position based on action (0 = buy/long, 1 = sell/short)
        self.prev_position = self.position
        self.position = 1 if action == 0 else -1

        # Calculate reward based on selected reward function
        reward = self.reward_functions[self.reward_function](price_change)

        info = {
            'position': self.position,
            'current_price': current_price,
            'price_change': price_change,
            'reward': reward
        }
        
        self.current_step += 1
        truncated = False

        return self._next_observation(), reward, self.done, truncated, info

    def render(self, mode='human'):
        pass

class RLStrategy(Strategy):    
    def __init__(self, name: str, output_dir: str = "models", model_type: str = "PPO",
                 reward_function: str = "risk_adjusted", window_size: int = 20):
        super().__init__(name, model_type=model_type, output_dir=output_dir)
        self.model = None
        self.output_dir = output_dir
        self.model_type = model_type
        self.reward_function = reward_function
        self.window_size = window_size

    def train(self, features: pd.DataFrame):
        logger.info(f"Training {self.name}")
        features = features.fillna(0)
          # Create environment for training
        base_env = TradingEnv(features, features, reward_function=self.reward_function,
                        window_size=self.window_size)
        # Wrap with Monitor first
        monitored_env = Monitor(base_env, os.path.join(self.output_dir, 'training_logs'))
        env = DummyVecEnv([lambda: monitored_env])
        
        # Select RL algorithm
        if self.model_type.upper() == "PPO":
            RLModel = PPO
            model_kwargs = dict(
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[128, 128, 64],
                        vf=[128, 128, 64]
                    )
                ),
                learning_rate=0.0001,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                verbose=0
            )
        elif self.model_type.upper() == "DQN":
            RLModel = DQN
            model_kwargs = dict(
                learning_rate=0.0001,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                target_update_interval=1000,
                policy_kwargs=dict(net_arch=[128, 128, 64]),
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported RL model_type: {self.model_type}")
            
        self.model = RLModel('MlpPolicy', env, **model_kwargs)
        from stable_baselines3.common.callbacks import EvalCallback        # Create and wrap evaluation environment
        eval_base_env = TradingEnv(features, features,
                                reward_function=self.reward_function,
                                window_size=self.window_size)
        eval_monitored_env = Monitor(eval_base_env, os.path.join(self.output_dir, 'eval_logs'))
        eval_env = DummyVecEnv([lambda: eval_monitored_env])
        eval_callback = EvalCallback(eval_env, 
                                   best_model_save_path=self.output_dir,
                                   log_path=self.output_dir, 
                                   eval_freq=1000,
                                   deterministic=True, 
                                   render=False)
        self.model.learn(
            total_timesteps=10000,
            callback=eval_callback,
            progress_bar=True
        )
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = os.path.join(self.output_dir, f"{self.name}_model_final")
        self.model.save(model_path)
        logger.info(f"Final model saved to {model_path}")
        logger.info(f"Training completed for {self.name}")
        logger.info(f"Best evaluation reward: {eval_callback.best_mean_reward:.2f}")

    def generate_signals(self, features: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Use passed features directly
        features = features.fillna(0)
        
        # Create environment for prediction
        env = TradingEnv(features, features, reward_function=self.reward_function,
                        window_size=self.window_size)
        
        signals = []
        obs, _ = env.reset()  # Gymnasium returns (obs, info)
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            # 0=buy/long (1), 1=sell/short (-1)
            signals.append(1 if action.item() == 0 else -1)
            obs, _, done, truncated, info = env.step(action.item())  # Gymnasium returns 5 values
            
            # Log trading activity
            logger.debug(f"Step: {env.current_step}, Action: {action.item()}, " +
                      f"Position: {info['position']}, Price: {info['current_price']:.2f}")
        
        # No need to append extra signal, keep signals length same as features
        signals = pd.Series(signals, index=features.index[:len(signals)])
        # Debug: print agent outputs
        logger.info(f"[RL DEBUG] signals value counts: {signals.value_counts().to_dict()}")
        
        # Generate entry/exit signals with enhanced logic
        prev = signals.shift(1, fill_value=0)
        
        long_entry = ((signals == 1) & (prev != 1)).astype(bool)
        long_exit = ((signals != 1) & (prev == 1)).astype(bool)
        short_entry = ((signals == -1) & (prev != -1)).astype(bool)
        short_exit = ((signals != -1) & (prev == -1)).astype(bool)
                
        logger.info(f"[RL DEBUG] long_entry sum: {long_entry.sum()}, long_exit sum: {long_exit.sum()}, short_entry sum: {short_entry.sum()}, short_exit sum: {short_exit.sum()}")
        
        return {
            'long_entry': long_entry.astype(bool),
            'long_exit': long_exit.astype(bool),
            'short_entry': short_entry.astype(bool),
            'short_exit': short_exit.astype(bool),
            'raw_signal': signals.astype(int)
        }
