import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, Any

class GoldTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for trading Gold (XAUUSD).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, config_path: str = "config/config.yaml", render_mode: str = None):
        """
        Initialize the environment.

        Args:
            df (pd.DataFrame): The preprocessed and scaled DataFrame with features.
            config_path (str): Path to configuration file.
            render_mode (str): Rendering mode.
        """
        super(GoldTradingEnv, self).__init__()
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.df = df
        self.render_mode = render_mode
        self.config_env = self.config['env']
        
        # Actions: 0 = Hold/Close, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Window of features + Account State
        # Account State: [Balance/MaxBalance, Position (0, 1, -1), Unrealized PnL/Balance]
        # We need to determine the number of features from the dataframe
        # Assuming keys like 'timestamp' and OHLCV might be dropped or kept.
        # We will use the feature columns defined in FeatureEngineer (passed implicitly via df columns)
        # Ideally, we should pass the feature columns explicitly or infer them.
        # For now, we assume all columns in df are features except 'target' if any.
        
        self.window_size = self.config_env['window_size']
        self.n_features = len(df.columns)
        self.shape = (self.window_size, self.n_features)
        
        # Append 2 extra dimensions for account info to the flattened space or separate?
        # Standard approach: Flatten window or use Dict space. 
        # For MlpPolicy, Box space 1D is best. 
        # Let's flatten the window features and add account info.
        # However, for potential CNN/LSTM use (though user said MLP), Box(window, features) is natural.
        # But MlpPolicy in SB3 flattens automatically.
        # We need to append scalar account info. This is tricky with a simple Box.
        # Sol: Normalize account info and append to *every* step in the window or just flattening.
        # We will use a flat vector: (window_size * n_features) + 2 (position, drawdown_metric)
        
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

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.config_env['initial_balance']
        self.max_balance = self.balance
        self.position = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.history = []
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _get_observation(self) -> np.ndarray:
        # Get window of data
        window_data = self.df.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        
        # Account info
        # Normalization helps: Position is -1, 0, 1. Drawdown/Balance ratio.
        # Drawdown 0 to 1 (or small percentage).
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        account_state = np.array([self.position, current_drawdown])
        
        return np.concatenate([window_data, account_state]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step - 1]['close'] # Or entry price
        
        reward = 0
        terminated = False
        truncated = False
        
        # Calculate Returns / Steps
        # We can calculate unrealized PnL change for reward at every step
        
        step_reward = 0.0
        
        # Execute Action
        if action == 0: # Close/Stay Flat
            if self.position != 0:
                # Close position
                pnl = (current_price - self.entry_price) / self.entry_price * self.position * self.balance
                pnl -= self.balance * self.config_env['commission'] # Exit cost
                self.balance += pnl
                self.position = 0
                self.entry_price = 0.0
                step_reward += pnl # Realized PnL is part of reward
            else:
                # Already flat, no cost, small penalty for inactivity? Or 0.
                pass
                
        elif action == 1: # Long
            if self.position == 1:
                # Hold Long
                pass 
            elif self.position == -1:
                # Close Short and Go Long
                # Close Short
                pnl = (self.entry_price - current_price) / self.entry_price * self.balance
                pnl -= self.balance * self.config_env['commission']
                self.balance += pnl
                # Open Long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
            else:
                # Open Long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
                
        elif action == 2: # Short
            if self.position == -1:
                # Hold Short
                pass
            elif self.position == 1:
                # Close Long and Go Short
                pnl = (current_price - self.entry_price) / self.entry_price * self.balance
                pnl -= self.balance * self.config_env['commission']
                self.balance += pnl
                # Open Short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
            else:
                # Open Short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.balance * self.config_env['commission']
        
        # Update Stats
        if self.balance > self.max_balance:
            self.max_balance = self.balance
            
        drawdown = (self.max_balance - self.balance) / self.max_balance
        
        # Reward Calculation
        # Reward = (Log_Return * Sharpe_Bonus) - (Drawdown_Penalty * lambda) - Transaction_Costs
        # Calculating step log return of the PORTFOLIO
        # Current Portfolio Value = Balance + Unrealized PnL
        unrealized_pnl = 0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.balance
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price * self.balance
            
        portfolio_value = self.balance + unrealized_pnl
        # We need previous portfolio value to calc return. 
        # Easier: Approximate Reward = Change in Portfolio Value - Costs - Drawdown Penalty
        
        # Let's stick to the user formula more directly if possible.
        # "Log_Return" usually refers to the asset or the portfolio. Here portfolio.
        # If we just opened, prev value was Balance (minus cost).
        
        # Using simple PnL change for stability in this step
        # But we need to incorporate the penalty.
        
        reward = (portfolio_value - self.max_balance) * 0.0 # Just tracking? No.
        
        # My interpretation: 
        # Reward = Scaled_Return - (Drawdown * Lambda)
        
        # We store portfolio value history to calc Sharpe properly, but for Step Reward:
        # We can use instantaneous return.
        
        # Note: self.balance is Cash. Portfolio Value is Cash + Positions.
        # We should track Portfolio Value primarily.
        
        # Let's refine the state tracking to be simpler:
        # Always track 'Portfolio Value'.
        # Previous Step Portfolio Value vs Current.
        
        # Re-calc properly
        pass # Implementation detail in next block
        
        # Correct implementation approach for step reward:
        # 1. Calc Portfolio Value at t-1
        # 2. Calc Portfolio Value at t
        # 3. Log Return = ln(V_t / V_{t-1})
        # 4. Reward = Log_Return * Bonus - Drawdown * Penalty
        
        # We need self.prev_portfolio_value
        if not hasattr(self, 'prev_portfolio_value'):
            self.prev_portfolio_value = self.config_env['initial_balance']
            
        current_portfolio_value = self.balance + unrealized_pnl
        
        if self.prev_portfolio_value > 0:
            log_return = np.log(current_portfolio_value / self.prev_portfolio_value)
        else:
            log_return = 0
            
        # Update prev
        self.prev_portfolio_value = current_portfolio_value
        
        # Constants
        sharpe_bonus = 100.0 # Configurable
        drawdown_penalty_factor = 10.0 # Lambda
        
        reward = (log_return * sharpe_bonus) - (drawdown * drawdown_penalty_factor)
        
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
