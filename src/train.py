import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pandas as pd
import yaml
import os
from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv

def train_agent():
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
    train_df = fe.transform(train_df)
    test_df = fe.transform(test_df) # For evaluation
    
    print(f"Training Data Shape: {train_df.shape}")
    print(f"Testing Data Shape: {test_df.shape}")
    
    # Create Environments
    # Using DummyVecEnv for now, can switch to Subproc for parallel training
    env = DummyVecEnv([lambda: GoldTradingEnv(train_df, config_path)])
    eval_env = DummyVecEnv([lambda: GoldTradingEnv(test_df, config_path)])
    
    # Callbacks
    save_path = config['training']['model_save_path']
    log_path = config['training']['log_dir']
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="ppo_gold_model"
    )
    
    # Agent Setup
    agent_config = config['agent']
    policy_kwargs = dict(net_arch=[256, 256, 128])
    
    model = PPO(
        agent_config['policy'],
        env,
        learning_rate=agent_config['learning_rate'],
        n_steps=agent_config['n_steps'],
        batch_size=agent_config['batch_size'],
        gamma=agent_config['gamma'],
        ent_coef=agent_config['ent_coef'],
        clip_range=agent_config['clip_range'],
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda" # Auto-detects, but explicit preference
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=agent_config['total_timesteps'],
        callback=[eval_callback, checkpoint_callback]
    )
    print("Training finished!")
    
    # Save final model
    model.save(os.path.join(save_path, "final_model"))

if __name__ == "__main__":
    train_agent()
