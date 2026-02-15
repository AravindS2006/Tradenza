import pandas as pd
import yaml
import os
import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv

# Try importing QR-DQN from sb3-contrib
try:
    from sb3_contrib import QRDQN
    HAS_SB3_CONTRIB = True
except ImportError:
    HAS_SB3_CONTRIB = False
    print("Warning: sb3-contrib not installed. QR-DQN will be skipped or replaced by PPO.")

def train_meta():
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Load Data
    dl = DataLoader(config_path)
    df = dl.load_data()
    fe = FeatureEngineer(config_path)
    df = fe.add_technical_indicators(df)
    train_df, test_df = dl.split_data(df)
    fe.fit_scaler(train_df)
    train_df = fe.transform(train_df)
    
    # Grid of Variants
    algorithms = ['PPO', 'A2C']
    if HAS_SB3_CONTRIB:
        algorithms.append('QRDQN')
        
    rewards = ['sharpe', 'sortino', 'calmar_duration']
    
    # Training Loop
    # We will train a small subset or full set based on user needs.
    # For demonstration, let's train one of each algo with one reward, or full grid.
    # Due to time, let's run: PPO-Sharpe, A2C-Sortino, QRDQN-Calmar
    
    variants = [
        ('PPO', 'sharpe'),
        ('A2C', 'sortino'),
        ('QRDQN', 'calmar_duration') if HAS_SB3_CONTRIB else ('PPO', 'calmar_duration')
    ]
    
    for algo_name, reward_type in variants:
        print(f"\nTraining Variant: {algo_name} with {reward_type} Reward")
        
        # Create Env with specific reward
        # Note: Vectorized envs need the arg passed correctly. 
        # Lambda is easiest for DummyVecEnv
        env = DummyVecEnv([lambda: GoldTradingEnv(train_df, config_path, reward_type=reward_type)])
        
        model_name = f"{algo_name}_{reward_type}"
        timestamps = 50000 # Reduced for meta-demo, user can increase
        
        if algo_name == 'PPO':
            model = PPO("MlpPolicy", env, verbose=1, 
                        ent_coef=config['agent']['ent_coef'],
                        learning_rate=config['agent']['learning_rate'])
        elif algo_name == 'A2C':
            model = A2C("MlpPolicy", env, verbose=1,
                        ent_coef=config['agent']['ent_coef'],
                        learning_rate=config['agent']['learning_rate'])
        elif algo_name == 'QRDQN':
            model = QRDQN("MlpPolicy", env, verbose=1,
                          learning_rate=config['agent']['learning_rate'])
                          
        model.learn(total_timesteps=timestamps)
        
        save_path = os.path.join(config['training']['model_save_path'], f"{model_name}.zip")
        model.save(save_path)
        print(f"Saved variant to {save_path}")

if __name__ == "__main__":
    train_meta()
