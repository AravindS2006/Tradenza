import pandas as pd
import numpy as np
import yaml
import os
from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv

def debug_env():
    # Load Config
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    print("Loading Config...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Prepare Data
    print("Loading Data...")
    dl = DataLoader(config_path)
    df = dl.load_data()
    print(f"Total Rows: {len(df)}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    
    fe = FeatureEngineer(config_path)
    df = fe.add_technical_indicators(df)
    
    train_df, test_df = dl.split_data(df)
    print(f"Train Rows: {len(train_df)}")
    print(f"Test Rows: {len(test_df)}")
    
    if len(train_df) == 0:
        print("CRITAL ERROR: Training data is empty!")
        return

    fe.fit_scaler(train_df)
    train_df = fe.transform(train_df)
    
    # Create Env
    print("\nInitializing Environment...")
    env = GoldTradingEnv(train_df, config_path)
    
    obs, _ = env.reset()
    print(f"Observation Shape: {obs.shape}")
    print(f"Initial Balance: {env.balance}")
    
    print("\nRunning random steps...")
    rewards = []
    actions = []
    for i in range(100):
        # Force some trades
        action = np.random.choice([0, 1, 2])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        
        if i < 5:
            print(f"Step {i}: Action={action}, Reward={reward:.4f}, Balance={env.balance:.2f}, Pos={env.position}")
            
        if terminated or truncated:
            break
            
    print(f"\nAverage Reward: {np.mean(rewards)}")
    print(f"Action Distribution: {np.unique(actions, return_counts=True)}")
    print(f"Non-zero rewards: {np.count_nonzero(rewards)}")

if __name__ == "__main__":
    debug_env()
