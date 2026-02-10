# Tradenza: Deep Reinforcement Learning for Algorithmic Trading 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-3776AB?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Overview

**Tradenza** is a production-grade algorithmic trading system powered by **Deep Reinforcement Learning (DRL)**. It is designed to autonomously trade **XAUUSD (Gold)** by learning optimal policies that maximize **Risk-Adjusted Returns (Sharpe Ratio)** while strictly limiting **Maximum Drawdown**.

Built with a modular architecture, Tradenza leverages **State-of-the-Art (SOTA)** RL algorithms (PPO/SAC) from `stable-baselines3`, robust feature engineering, and a custom `Gymnasium` environment that simulates realistic market mechanics including spread and transaction costs.

## 🏗️ Architecture

The system is designed with a separation of concerns, ensuring scalability and ease of maintenance.

```mermaid
graph TD
    Data[Data Pipeline] -->|Raw OHLCV| FE[Feature Engineering]
    FE -->|Tech Indicators + Normalization| Env[GoldTradingEnv]
    
    subgraph "RL Agent (PPO/SAC)"
        Policy[Policy Network]
        Value[Value Network]
    end
    
    Env -->|State (Windowed)| Policy
    Policy -->|Action (Long/Short/Hold)| Env
    Env -->|Reward (Sharpe/Drawdown)| Policy
    
    Optuna[Optuna HPO] -->|Hyperparams| Policy
    
    subgraph "Evaluation"
        Backtest[Backtesting Engine]
        Metrics[QuantStats Report]
    end
    
    Policy -->|Trained Model| Backtest
    Backtest --> Metrics
```

## ✨ Key Features

-   **Deep Reinforcement Learning**: Implements **Proximal Policy Optimization (PPO)** with custom MLP architectures to navigate complex market dynamics.
-   **Advanced Feature Engineering**: Utilizes a robust set of technical indicators:
    -   **Trend**: MACD, Moving Averages.
    -   **Momentum**: RSI.
    -   **Volatility**: Bollinger Bands, ATR.
    -   **Stationarity**: Log-returns and rolling window normalization to prevent look-ahead bias.
-   **Custom Gymnasium Environment**:
    -   Realistic trade simulation with commission and spread modeling.
    -   **Reward Shaping**: Custom reward function balancing profit maximization ($R$) with risk penalties ($MDD$).
    -   $$ Reward = \text{LogReturn} \times \alpha - \text{Drawdown} \times \lambda - \text{Costs} $$
-   **Hyperparameter Optimization**: Integrated **Optuna** pipeline to automatically tune learning rates, batch sizes, and network architectures for optimal convergence.
-   **Institutional-Grade Backtesting**: Comprehensive performance analytics using **QuantStats**, generating detailed HTML reports with Win Rate, Sharpe Ratio, Sortino Ratio, and Drawdown analysis.

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/Tradenza.git
    cd Tradenza
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Configuration
Modify `config/config.yaml` to set your data paths, hyperparameters, and environment settings.

### 2. Training
Train the DRL agent on your dataset:
```bash
python -m src.train
```
*The model checkpoints will be saved to `models/`.*

### 3. Backtesting
Evaluate the trained agent on unseen test data:
```bash
python -m src.backtest
```
*Performance reports are generated as `backtest_report.html`.*

### 4. Hyperparameter Tuning
Optimize the agent using Optuna:
```bash
python -m src.optuna_tuning
```

## 📊 Performance Metrics

The system targets the following performance benchmarks:
-   **Sharpe Ratio**: > 1.5
-   **Profit Factor**: > 1.8
-   **Max Drawdown**: < 15%

## 📂 Project Structure

```bash
Tradenza/
├── config/             # Configuration management
├── data/               # Datasets and loaders
├── models/             # Saved RL models
├── src/
│   ├── data/           # Data cleaning and loading
│   ├── features/       # Feature engineering & scaling
│   ├── env/            # Custom Gymnasium environment
│   ├── train.py        # Training loop
│   ├── backtest.py     # Evaluation & Backtesting
│   └── optuna_tuning.py # Optuna optimization
└── notebook/           # Research notebooks
```

## 👨‍💻 Author

**Aravind**  
*Quantitative Developer & AI Researcher*

---
*Disclaimer: This project is for educational and research purposes only. Financial trading involves significant risk.*
