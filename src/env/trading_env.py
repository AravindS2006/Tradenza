import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, Any
from src.rewards import Reward_Sharpe, Reward_Sortino, Reward_Calmar_Duration

class GoldTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for trading Gold (XAUUSD).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml", render_mode: str = None, reward_type: str = 'sharpe'):
        """
        Initialize the environment.

        Args:
            df (pd.DataFrame): The preprocessed and scaled DataFrame with features.
            config_path (str): Path to configuration file.
            render_mode (str): Rendering mode.
            reward_type (str): 'sharpe', 'sortino', or 'calmar_duration'.
        """
        super(GoldTradingEnv, self).__init__()
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.df = df
        self.render_mode = render_mode
        self.config_env = self.config['env']
        
        # Initialize Reward Strategy
        if reward_type == 'sortino':
            self.reward_strategy = Reward_Sortino()
        elif reward_type == 'calmar_duration':
            self.reward_strategy = Reward_Calmar_Duration()
        else:
            self.reward_strategy = Reward_Sharpe()
            
        # Actions: 0 = Hold/Close, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Window of features + Account State
        self.window_size = self.config_env['window_size']
        self.n_features = len(df.columns)
        self.obs_shape = (self.window_size * self.n_features + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        # State variables
        self.current_step = 0
        self.balance = self.config_env['initial_balance']
        self.max_balance = self.balance
        self.position = 0 # 0: None, 1: Long, -1: Short
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.history = []

    # ... (reset and _get_observation methods remain mostly same, ensure history clear in reset) ...
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.config_env['initial_balance']
        self.max_balance = self.balance
        self.position = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.history = []
        
        # Reset reward strategy state if needed
        if hasattr(self.reward_strategy, 'current_drawdown_duration'):
            self.reward_strategy.current_drawdown_duration = 0
            self.reward_strategy.peak_balance = -1
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _get_observation(self) -> np.ndarray:
        # Get window of data
        window_data = self.df.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        
        # Account info
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        account_state = np.array([self.position, current_drawdown])
        
        return np.concatenate([window_data, account_state]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.df.iloc[self.current_step]['close']
        
        reward = 0
        terminated = False
        truncated = False
        
        # Execute Action
        if action == 0: # Close/Stay Flat
            if self.position != 0:
                # Close position
                pnl = (current_price - self.entry_price) / self.entry_price * self.position * self.balance
                pnl -= self.balance * self.config_env['commission']
                self.balance += pnl
                self.position = 0
                self.entry_price = 0.0
        elif action == 1: # Long
            if self.position == -1: # Close Short
                pnl = (self.entry_price - current_price) / self.entry_price * self.balance
                pnl -= self.balance * self.config_env['commission']
                self.balance += pnl
                self.position = 0
            
            if self.position == 0: # Open Long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
                
        elif action == 2: # Short
            if self.position == 1: # Close Long
                pnl = (current_price - self.entry_price) / self.entry_price * self.balance
                pnl -= self.balance * self.config_env['commission']
                self.balance += pnl
                self.position = 0
                
            if self.position == 0: # Open Short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
        
        # Update Stats
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        # Calculate PnL / Returns for Reward
        # We need to calculate the step log return of the portfolio value
        unrealized_pnl = 0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.balance
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price * self.balance
            
        current_portfolio_value = self.balance + unrealized_pnl
        
        if not hasattr(self, 'prev_portfolio_value'):
            self.prev_portfolio_value = self.config_env['initial_balance']
            
        log_return = 0
        if self.prev_portfolio_value > 0:
            log_return = np.log(current_portfolio_value / self.prev_portfolio_value)
            
        self.prev_portfolio_value = current_portfolio_value
        
        # Maintain History
        drawdown = (self.max_balance - self.balance) / self.max_balance
        step_info = {
            'portfolio_value': current_portfolio_value,
            'return': log_return,
            'drawdown': drawdown,
            'position': self.position
        }
        self.history.append(step_info)
        
        # Calculate Reward using Strategy
        reward = self.reward_strategy.calculate(
            log_return=log_return,
            history=self.history,
            position=self.position,
            current_step=self.current_step
        )
        
        # Check termination
        if self.current_step >= len(self.df) - 1:
            terminated = True
        
        if self.balance <= self.config_env['initial_balance'] * 0.5: # Bust
            terminated = True
            reward -= 100 # Big penalty for busting
            
        self.current_step += 1
        self.total_reward += reward
        
        info = {
            'portfolio_value': current_portfolio_value,
            'return': log_return,
            'drawdown': drawdown,
            'position': self.position
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}, Reward: {self.total_reward:.4f}")
