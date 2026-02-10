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

        # Detect delimiter
        try:
            with open(path, 'r') as f:
                header = f.readline()
                if ';' in header:
                    sep = ';'
                else:
                    sep = ','
        except:
            sep = ','
            
        df = pd.read_csv(path, sep=sep)
        
        # Rename commonly used columns if necessary
        # Normalize to lower case first
        df.columns = [col.strip() for col in df.columns]
        
        # Mapping for various formats
        rename_map = {
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'gmt time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'volume': 'volume'
        }
        
        # Create a new map based on lowercase version of columns
        final_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in rename_map:
                final_map[col] = rename_map[lower_col]
            else:
                final_map[col] = lower_col # Default to lowercase
                
        df = df.rename(columns=final_map)

        if 'timestamp' in df.columns:
            # Handle specific date format if needed, e.g. "2006.01.02 01:00"
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                # Try explicit format matching the user's data "YYYY.MM.DD HH:MM"
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y.%m.%d %H:%M")
                except:
                    # Fallback to flexible
                     df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

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
        Split data into training and testing sets.
        Supports explicit date ranges from config if available, otherwise ratio split.

        Args:
            df (pd.DataFrame): The full dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        data_config = self.config['data']
        
        # Check if explicit dates are provided
        if 'train_start_date' in data_config and 'val_start_date' in data_config:
            train_start = pd.Timestamp(data_config['train_start_date'])
            train_end = pd.Timestamp(data_config['train_end_date'])
            val_start = pd.Timestamp(data_config['val_start_date'])
            val_end = pd.Timestamp(data_config['val_end_date'])
            
            # Ensure timestamps are localized if the index is (or normalize)
            # data_loader.load_data ensures index is datetime
            
            train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
            test_df = df[(df.index >= val_start) & (df.index <= val_end)].copy()
            
            if len(train_df) == 0:
                print(f"Warning: Training set empty for range {train_start} - {train_end}")
            if len(test_df) == 0:
                print(f"Warning: Validation set empty for range {val_start} - {val_end}")
                
            return train_df, test_df
            
        else:
            # Fallback to ratio split
            split_ratio = data_config.get('split_ratio', 0.8)
            split_idx = int(len(df) * split_ratio)
            
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            return train_df, test_df
