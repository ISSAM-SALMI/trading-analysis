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

# ==================== INDICATOR FUNCTIONS ====================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_trend(ema50, ema200):
    """Detect market trend based on EMA crossover"""
    if ema50 > ema200:
        return "UPTREND"
    elif ema50 < ema200:
        return "DOWNTREND"
    else:
        return "NEUTRAL"

# ==================== SIGNAL GENERATION ====================

def generate_signals(data):
    """
    Generate BUY and SELL signals based on Trend Pullback Strategy
    
    Rules:
    LONG (BUY):
    - EMA50 > EMA200 (Uptrend)
    - Price pulls back close to EMA50 (within 2%)
    - RSI between 40 and 70
    
    SHORT (SELL):
    - EMA50 < EMA200 (Downtrend)
    - Price pulls back close to EMA50 (within 2%)
    - RSI between 30 and 60
    """
    
    # Calculate indicators
    data['ema_50'] = calculate_ema(data, 50)
    data['ema_200'] = calculate_ema(data, 200)
    data['rsi'] = calculate_rsi(data, 14)
    
    # Initialize signal columns
    data['signal'] = None
    data['signal_price'] = None
    
    # Loop through data to find signals
    for i in range(200, len(data)):  # Start after EMA200 is calculated
        ema50 = data['ema_50'].iloc[i]
        ema200 = data['ema_200'].iloc[i]
        close = data['close'].iloc[i]
        rsi = data['rsi'].iloc[i]
        
        # Skip if indicators not ready
        if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
            continue
        
        # Calculate distance to EMAs
        dist_to_ema50 = ((close - ema50) / ema50) * 100
        
        # Detect trend
        trend = detect_trend(ema50, ema200)
        
        # ===== LONG SIGNAL (BUY) =====
        if trend == "UPTREND":
            # Price pulled back to EMA50 (within 2%)
            if abs(dist_to_ema50) <= 2.0:
                # RSI confirmation
                if 40 <= rsi <= 70:
                    data.at[data.index[i], 'signal'] = 'BUY'
                    data.at[data.index[i], 'signal_price'] = close
        
        # ===== SHORT SIGNAL (SELL) =====
        elif trend == "DOWNTREND":
            # Price pulled back to EMA50 (within 2%)
            if abs(dist_to_ema50) <= 2.0:
                # RSI confirmation
                if 30 <= rsi <= 60:
                    data.at[data.index[i], 'signal'] = 'SELL'
                    data.at[data.index[i], 'signal_price'] = close
    
    return data

# ==================== ETL PROCESS ====================

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
    
    # Standardize column names to lowercase
    df_filtered.columns = df_filtered.columns.str.lower()
    
    # Convert datetime
    if "datetime" in df_filtered.columns:
        df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"])
        
        # Drop duplicates based on datetime
        df_filtered = df_filtered.drop_duplicates(subset=["datetime"])
        
        # Sort by datetime
        df_filtered = df_filtered.sort_values(by="datetime")
        
        # Reset index after sorting
        df_filtered = df_filtered.reset_index(drop=True)
    
    # Filter volume > 0
    if "volume" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["volume"] > 0]
        df_filtered = df_filtered.reset_index(drop=True)
    
    # ===== GENERATE SIGNALS =====
    print(f"Generating trading signals for {symbol}...")
    df_filtered = generate_signals(df_filtered)
    
    # Show schema and sample data
    print(f"\nTransformed Data Schema for {symbol}:")
    print(df_filtered.info())
    print(f"\nSample Data for {symbol}:")
    print(df_filtered.head(5))
    
    # Show signals summary
    signals_count = df_filtered['signal'].value_counts()
    print(f"\nSignals Summary for {symbol}:")
    print(signals_count)
    
    # Save Data
    print(f"\nSaving data to {output_path}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    
    print(f"✓ {name} data saved successfully with {len(df_filtered)} rows")

def process_data():
    print(f"Reading data from {INPUT_FILE}...")
    
    # Read CSV
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    for config in SYMBOLS_CONFIG:
        process_symbol(df, config)
    print("\n" + "="*60)
    print("ETL Process Completed Successfully.")
    print("="*60)

def main():
    process_data()

if __name__ == "__main__":
    main()