"""
Trading Signal Generator - Trend Pullback Strategy with EMA + RSI
Author: Expert Trading Algorithm
Strategy: Detect pullbacks to EMA50/200 in trending markets with RSI confirmation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==================== INDICATOR CALCULATIONS ====================

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

# ==================== TREND DETECTION ====================

def detect_trend(ema50, ema200):
    """Detect market trend based on EMA crossover"""
    if ema50 > ema200:
        return "UPTREND"
    elif ema50 < ema200:
        return "DOWNTREND"
    else:
        return "NEUTRAL"

# ==================== PULLBACK DETECTION ====================

def detect_pullback_to_ema(data, ema50, ema200, lookback=5):
    """
    Detect if price has pulled back to EMA50 or EMA200
    Returns: distance to EMA50 and EMA200 as percentage
    """
    close = data['close']
    distance_to_ema50 = ((close - ema50) / ema50) * 100
    distance_to_ema200 = ((close - ema200) / ema200) * 100
    
    return distance_to_ema50, distance_to_ema200

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
        dist_to_ema200 = ((close - ema200) / ema200) * 100
        
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

# ==================== VISUALIZATION ====================

def visualize_signals(data, output_file='trading_signals.html'):
    """
    Create interactive candlestick chart with BUY/SELL signals
    """
    
    # Filter signals
    buy_signals = data[data['signal'] == 'BUY']
    sell_signals = data[data['signal'] == 'SELL']
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Action with Signals', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # EMA 50
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ema_50'],
        name='EMA 50',
        line=dict(color='#2196F3', width=2)
    ), row=1, col=1)
    
    # EMA 200
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ema_200'],
        name='EMA 200',
        line=dict(color='#FF6D00', width=2)
    ), row=1, col=1)
    
    # BUY Signals (Green Arrows)
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['low'] * 0.995,  # Slightly below the low
            mode='markers+text',
            name='BUY Signal',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='#00ff00',
                line=dict(color='#00cc00', width=2)
            ),
            text=['BUY'] * len(buy_signals),
            textposition='bottom center',
            textfont=dict(size=10, color='#00ff00', family='Arial Black'),
            hovertemplate='<b>BUY SIGNAL</b><br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # SELL Signals (Red Arrows)
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['high'] * 1.005,  # Slightly above the high
            mode='markers+text',
            name='SELL Signal',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='#ff0000',
                line=dict(color='#cc0000', width=2)
            ),
            text=['SELL'] * len(sell_signals),
            textposition='top center',
            textfont=dict(size=10, color='#ff0000', family='Arial Black'),
            hovertemplate='<b>SELL SIGNAL</b><br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['rsi'],
        name='RSI',
        line=dict(color='#9C27B0', width=2)
    ), row=2, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Trading Signals - Trend Pullback Strategy (EMA + RSI)',
        xaxis_title='Date/Time',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=900,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    # Save to HTML
    fig.write_html(output_file)
    print(f"✅ Chart saved to: {output_file}")
    
    return fig

# ==================== MAIN EXECUTION ====================

def analyze_trading_signals(csv_path, output_html='trading_signals.html'):
    """
    Main function to analyze CSV data and generate trading signals
    
    Args:
        csv_path: Path to CSV file with OHLC data
        output_html: Output file for interactive chart
    """
    
    print("=" * 60)
    print("🚀 TRADING SIGNAL ANALYZER - Trend Pullback Strategy")
    print("=" * 60)
    
    # 1. Load data
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Standardize column names (handle different formats)
    df.columns = df.columns.str.lower()
    
    # Convert datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Generate signals
    print("\n🔍 Calculating indicators and generating signals...")
    df_with_signals = generate_signals(df)
    
    # 3. Count signals
    buy_count = len(df_with_signals[df_with_signals['signal'] == 'BUY'])
    sell_count = len(df_with_signals[df_with_signals['signal'] == 'SELL'])
    
    print(f"\n📊 RESULTS:")
    print(f"   🟢 BUY signals: {buy_count}")
    print(f"   🔴 SELL signals: {sell_count}")
    print(f"   📈 Total signals: {buy_count + sell_count}")
    
    # 4. Display signals
    if buy_count > 0 or sell_count > 0:
        print("\n" + "=" * 60)
        print("📋 DETAILED SIGNALS:")
        print("=" * 60)
        
        signals = df_with_signals[df_with_signals['signal'].notna()][['signal', 'signal_price', 'rsi', 'ema_50', 'ema_200']]
        for idx, row in signals.iterrows():
            signal_type = "🟢 BUY " if row['signal'] == 'BUY' else "🔴 SELL"
            print(f"\n{signal_type} at {idx}")
            print(f"   Price: ${row['signal_price']:.2f}")
            print(f"   RSI: {row['rsi']:.2f}")
            print(f"   EMA50: ${row['ema_50']:.2f}")
            print(f"   EMA200: ${row['ema_200']:.2f}")
    
    # 5. Visualize
    print("\n📈 Generating visualization...")
    visualize_signals(df_with_signals, output_html)
    
    # 6. Save signals to CSV
    output_csv = csv_path.replace('.csv', '_signals.csv')
    signals_only = df_with_signals[df_with_signals['signal'].notna()]
    if len(signals_only) > 0:
        signals_only.to_csv(output_csv)
        print(f"💾 Signals saved to: {output_csv}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return df_with_signals

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Path to your data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'clean_btc_data.csv')
    OUTPUT_HTML = os.path.join(BASE_DIR, 'website', 'trading_signals.html')
    
    # Run analysis
    df_results = analyze_trading_signals(CSV_PATH, OUTPUT_HTML)
    
    print(f"\n🌐 Open the file in your browser to see the interactive chart:")
    print(f"   {OUTPUT_HTML}")
