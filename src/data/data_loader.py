import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
import yaml

class DataLoader:
    """
    Handles loading and initial preprocessing of financial market data.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DataLoader with configuration settings.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_path = self.config['data']['raw_file_path']
        self.processed_path = self.config['data']['processed_file_path']

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path (Optional[str]): Path to the CSV file. If None, uses config default.

        Returns:
            pd.DataFrame: Loaded DataFrame with datetime index.
        """
        path = file_path if file_path else self.raw_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")

        df = pd.read_csv(path)
        
        # Ensure standard column names
        df.columns = [col.lower() for col in df.columns]
        
        # Rename commonly used columns if necessary (simple heuristic)
        rename_map = {
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'volume': 'volume'
        }
        df = df.rename(columns=rename_map)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        else:
             # If no timestamp column, assume index is already datetime or handle accordingly
             pass

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets based on time.
        Strictly prevents look-ahead bias by splitting chronologically.

        Args:
            df (pd.DataFrame): The full dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        split_ratio = self.config['data']['split_ratio']
        split_idx = int(len(df) * split_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
