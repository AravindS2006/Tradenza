import numpy as np

class RewardStrategy:
    """Base class for reward strategies."""
    def calculate(self, log_return: float, history: list, position: int, current_step: int) -> float:
        raise NotImplementedError

class Reward_Sharpe(RewardStrategy):
    """
    Reward = Log_Return - (StdDev_of_Returns * alpha)
    Prioritizes consistent returns over volatility.
    """
    def __init__(self, alpha: float = 0.1, window: int = 50):
        self.alpha = alpha
        self.window = window
        
    def calculate(self, log_return: float, history: list, position: int, current_step: int) -> float:
        # history is list of dicts with 'return' key
        if len(history) < 2:
            return log_return
            
        returns = [h['return'] for h in history[-self.window:]]
        std_dev = np.std(returns) if len(returns) > 0 else 0
        
        # Scale return to be significant
        return (log_return * 1000) - (std_dev * self.alpha * 100)

class Reward_Sortino(RewardStrategy):
    """
    Reward = Log_Return - (Downside_Deviation * beta)
    Penalizes only negative volatility.
    """
    def __init__(self, beta: float = 0.1, window: int = 50):
        self.beta = beta
        self.window = window
        
    def calculate(self, log_return: float, history: list, position: int, current_step: int) -> float:
        if len(history) < 2:
            return log_return
            
        returns = np.array([h['return'] for h in history[-self.window:]])
        negative_returns = returns[returns < 0]
        
        downside_dev = np.std(negative_returns) if len(negative_returns) > 0 else 0
        
        return (log_return * 1000) - (downside_dev * self.beta * 100)

class Reward_Calmar_Duration(RewardStrategy):
    """
    Reward = Log_Return - (Drawdown_Duration * gamma)
    Heavily penalizes the LENGTH of time spent in drawdown.
    """
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma
        self.current_drawdown_duration = 0
        self.peak_balance = -1
        
    def calculate(self, log_return: float, history: list, position: int, current_step: int) -> float:
        # We need balance info, which should be in history or passed
        # For simplicity, we assume history contains 'portfolio_value'
        if not history:
            self.peak_balance = 0 # Init
            return 0
            
        current_val = history[-1]['portfolio_value']
        
        if self.peak_balance < 0:
            self.peak_balance = current_val
            
        if current_val > self.peak_balance:
            self.peak_balance = current_val
            self.current_drawdown_duration = 0
        else:
            self.current_drawdown_duration += 1
            
        # Penalty increases linearly with duration
        duration_penalty = self.current_drawdown_duration * self.gamma
        
        return (log_return * 1000) - duration_penalty
