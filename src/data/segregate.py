import pandas as pd
import yaml
import os
from src.data.data_loader import DataLoader

def segregate_data(config_path="config/config.yaml"):
    """
    Segregates the main data file into training and validation sets based on config dates.
    Saves them as separate CSV files.
    """
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    data_config = config['data']
    raw_path = data_config['raw_file_path']
    
    print(f"Loading data from {raw_path}")
    # Use existing DataLoader logic to handle formats
    dl = DataLoader(config_path)
    df = dl.load_data()
    
    # Extract Dates
    train_start = pd.Timestamp(data_config['train_start_date'])
    train_end = pd.Timestamp(data_config['train_end_date'])
    val_start = pd.Timestamp(data_config['val_start_date'])
    val_end = pd.Timestamp(data_config['val_end_date'])
    
    print(f"Filtering Training Data: {train_start} to {train_end}")
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    
    print(f"Filtering Validation Data: {val_start} to {val_end}")
    val_df = df[(df.index >= val_start) & (df.index <= val_end)]
    
    print(f"Training Samples: {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")
    
    # Save to files
    data_dir = os.path.dirname(raw_path)
    train_path = os.path.join(data_dir, "train_data.csv")
    val_path = os.path.join(data_dir, "validation_data.csv")
    
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    
    print(f"Saved training data to: {train_path}")
    print(f"Saved validation data to: {val_path}")

if __name__ == "__main__":
    segregate_data()
