import pandas as pd
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'input', 'data_2min.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')

# Configuration for symbols to process
SYMBOLS_CONFIG = [
    {"symbol": "GC=F", "filename": "clean_gold_data.csv", "name": "Gold Futures"},
    {"symbol": "ETH-USD", "filename": "clean_eth_data.csv", "name": "Ethereum"},
    {"symbol": "BTC-USD", "filename": "clean_btc_data.csv", "name": "Bitcoin"}
]

def process_symbol(df, symbol_config):
    symbol = symbol_config["symbol"]
    filename = symbol_config["filename"]
    name = symbol_config["name"]
    
    print(f"\nProcessing {name} ({symbol}) data...")
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    if "symbol" in df.columns:
        df_filtered = df[df["symbol"] == symbol].copy()
    else:
        print(f"Warning: 'symbol' column not found. Skipping {symbol}.")
        return

    if df_filtered.empty:
        print(f"No data found for {symbol}.")
        return

    # Columns to drop
    columns_to_drop = ['requested_timestamp', 'error', 'name', 'query_timestamp']
    
    # Drop columns
    existing_cols = df_filtered.columns
    cols_to_drop_existing = [c for c in columns_to_drop if c in existing_cols]
    df_filtered = df_filtered.drop(columns=cols_to_drop_existing)
    
    # Convert datetime
    if "datetime" in df_filtered.columns:
        df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"])
        
        # Drop duplicates based on datetime
        df_filtered = df_filtered.drop_duplicates(subset=["datetime"])
        
        # Sort by datetime
        df_filtered = df_filtered.sort_values(by="datetime")
    
    # Filter volume > 0
    if "volume" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["volume"] > 0]
    
    # Show schema and sample data
    print(f"Transformed Data Schema for {symbol}:")
    print(df_filtered.info())
    print(f"Sample Data for {symbol}:")
    print(df_filtered.head(5))
    
    # Save Data
    print(f"Saving data to {output_path}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

def process_data():
    print(f"Reading data from {INPUT_FILE}...")
    
    # Read CSV
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    for config in SYMBOLS_CONFIG:
        process_symbol(df, config)
        
    print("\nETL Process Completed Successfully.")

def main():
    process_data()

if __name__ == "__main__":
    main()
