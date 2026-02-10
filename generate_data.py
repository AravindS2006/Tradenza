import pandas as pd
import numpy as np
import datetime

def generate_synthetic_data(filename="data/XAUUSD.csv", days=1000, freq='1H'):
    """
    Generates synthetic XAUUSD-like price data.
    """
    np.random.seed(42)
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days*24, freq=freq)
    
    n = len(dates)
    
    # Random walk with drift
    returns = np.random.normal(loc=0.0001, scale=0.002, size=n)
    price_path = 1800 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    close = price_path
    high = close * (1 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n)))
    open_p = close * (1 + np.random.normal(0, 0.001, n)) # Approximation
    volume = np.random.poisson(lam=1000, size=n) * 1.0
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Ensure High is highest and Low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

if __name__ == "__main__":
    generate_synthetic_data()
