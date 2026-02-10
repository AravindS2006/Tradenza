import pandas as pd
from stable_baselines3.common.env_checker import check_env
from src.env import GoldTradingEnv
from src.data import DataLoader
from src.features import FeatureEngineer
import yaml

def check_custom_env():
    # Load Config
    # Check if config exists, if not use default path
    config_path = "config/config.yaml"
    
    # Load Data (Synthetic)
    dl = DataLoader(config_path)
    df = dl.load_data() # Uses default path from config
    
    # Features
    fe = FeatureEngineer(config_path)
    df = fe.add_technical_indicators(df)
    
    # Split (just use train for check)
    train_df, _ = dl.split_data(df)
    fe.fit_scaler(train_df)
    train_df = fe.transform(train_df)
    
    # Create Env
    env = GoldTradingEnv(train_df, config_path)
    
    # Check Env
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")
    
    # Test Reset and Step
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step reward: {reward}")
    print(f"Step info: {info}")

if __name__ == "__main__":
    check_custom_env()
