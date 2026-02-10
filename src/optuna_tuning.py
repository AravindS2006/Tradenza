import optuna
import yaml
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from src.data import DataLoader
from src.features import FeatureEngineer
from src.env import GoldTradingEnv

def objective(trial):
    # Load Config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Prepare Data Only Once (Global or outside loop would be better for speed, but for simplicity here)
    # Ideally, pass data to avoid reloading
    if 'train_df' not in globals():
        dl = DataLoader(config_path)
        df = dl.load_data()
        fe = FeatureEngineer(config_path)
        df = fe.add_technical_indicators(df)
        train_df, test_df = dl.split_data(df)
        fe.fit_scaler(train_df)
        train_df = fe.transform(train_df)
        test_df = fe.transform(test_df)
        globals()['train_df'] = train_df
        globals()['test_df'] = test_df
    
    train_data = globals()['train_df']
    test_data = globals()['test_df']
    
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # Setup Env
    train_env = DummyVecEnv([lambda: GoldTradingEnv(train_data, config_path)])
    eval_env = DummyVecEnv([lambda: GoldTradingEnv(test_data, config_path)])
    
    # Agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        ent_coef=ent_coef,
        n_steps=n_steps,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )
    
    # Pruning handle
    # Train for a shorter period for tuning
    total_timesteps = 20000 
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    
    return mean_reward

def main():
    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=20, timeout=3600) # 1 hour timeout or 20 trials
    except KeyboardInterrupt:
        pass
        
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params to a file
    with open("config/best_params.yaml", "w") as f:
        yaml.dump(trial.params, f)

if __name__ == "__main__":
    main()
