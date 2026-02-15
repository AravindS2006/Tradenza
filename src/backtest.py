import pandas as pd
import numpy as np
import yaml
import os
import argparse
from stable_baselines3 import PPO, A2C, SAC
try:
    from sb3_contrib import QRDQN
except ImportError:
    QRDQN = None

from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv
import quantstats as qs

def backtest(model_name="PPO_sharpe"):
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
    # Handle filenames with or without .zip extension
    if not model_name.endswith('.zip'):
        model_filename = f"{model_name}.zip"
    else:
        model_filename = model_name
        
    model_path = os.path.join(config['training']['model_save_path'], model_filename)
    if not os.path.exists(model_path):
        # Fallback to old default if simple run
        if model_name == "PPO_sharpe" and os.path.exists(os.path.join(config['training']['model_save_path'], "final_model.zip")):
             print("PPO_sharpe not found, falling back to final_model.zip...")
             model_path = os.path.join(config['training']['model_save_path'], "final_model.zip")
        else:
            print(f"Model not found at {model_path}. Please train first.")
            return
        
    print(f"Loading model from {model_path}...")
    
    # Determine algo class from name logic or try/except
    # Standard: Algo_Reward.zip
    if "PPO" in model_filename:
        model = PPO.load(model_path)
    elif "A2C" in model_filename:
        model = A2C.load(model_path)
    elif "SAC" in model_filename:
        model = SAC.load(model_path)
    elif "QRDQN" in model_filename and QRDQN:
        model = QRDQN.load(model_path)
    else:
        # Default fallback
        print("Unknown algorithm in filename, trying PPO...")
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
        qs.reports.html(results_df['returns'], output=f'backtest_report_{model_name}.html', title=f'Gold DRL Agent Backtest ({model_name})')
        print(f"Report saved to backtest_report_{model_name}.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PPO_sharpe", help="Model name (e.g., PPO_sharpe, A2C_sortino)")
    args = parser.parse_args()
    backtest(args.model)
