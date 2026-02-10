import pandas as pd
import numpy as np
import yaml
import os
from stable_baselines3 import PPO
from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv
import quantstats as qs

def backtest():
    # Load Config
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Prepare Data
    dl = DataLoader(config_path)
    df = dl.load_data()
    
    fe = FeatureEngineer(config_path)
    df = fe.add_technical_indicators(df)
    
    train_df, test_df = dl.split_data(df)
    fe.fit_scaler(train_df)
    test_df = fe.transform(test_df)
    
    # Load Model
    model_path = os.path.join(config['training']['model_save_path'], "final_model.zip")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    model = PPO.load(model_path)
    
    # Create Env
    env = GoldTradingEnv(test_df, config_path)
    
    # Run Backtest
    obs, _ = env.reset()
    done = False
    
    portfolio_values = []
    actions = []
    timestamps = test_df.index[env.window_size:] # Approx alignment
    
    # We need to capture the exact timestamps corresponding to steps
    # The env steps through data from window_size to end.
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        actions.append(action)
        
    # Create Analysis DataFrame
    # Length of portfolio_values should match the number of steps taken
    # test_df has length N. Steps = N - window_size.
    # timestamps should be test_df.index[window_size:]
    
    if len(portfolio_values) > len(timestamps):
        timestamps = test_df.index[env.window_size : env.window_size + len(portfolio_values)]
    else:
        timestamps = timestamps[:len(portfolio_values)]
        
    results_df = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'action': actions
    }, index=timestamps)
    
    # Calculate Returns
    results_df['returns'] = results_df['portfolio_value'].pct_change().fillna(0)
    
    # Metrics
    total_return = (results_df['portfolio_value'].iloc[-1] / config['env']['initial_balance']) - 1
    sharpe = qs.stats.sharpe(results_df['returns'])
    max_drawdown = qs.stats.max_drawdown(results_df['returns'])
    win_rate = len(results_df[results_df['returns'] > 0]) / len(results_df[results_df['returns'] != 0]) if len(results_df[results_df['returns'] != 0]) > 0 else 0
    
    print("Backtest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    
    # Generate Report
    if results_df['returns'].abs().sum() == 0:
        print("WARNING: Agent made no trades or returns are all zero. Skipping report generation.")
    else:
        qs.reports.html(results_df['returns'], output='backtest_report.html', title='Gold DRL Agent Backtest')
        print("Report saved to backtest_report.html")

if __name__ == "__main__":
    backtest()
