import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "TradingView" look
st.markdown("""
    <style>
        .stApp {
            background-color: #131722;
            color: #d1d4dc;
        }
        .stHeader {
            background-color: #131722;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Technical Indicators Functions
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

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_adx(data, period=14):
    """Calculate Average Directional Index (ADX)"""
    high_diff = data['high'].diff()
    low_diff = -data['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    atr = calculate_atr(data, period)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def detect_trend(ema_50, ema_200, adx, adx_threshold=25):
    """Detect trend based on EMA crossover and ADX strength"""
    if pd.isna(ema_50) or pd.isna(ema_200) or pd.isna(adx):
        return "No Data", "#808080"
    
    if adx < adx_threshold:
        return "No Clear Trend", "#ffa726"
    elif ema_50 > ema_200:
        return "Strong Uptrend", "#26a69a"
    else:
        return "Strong Downtrend", "#ef5350"

# Load Data
@st.cache_data
def load_data():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, 'data', 'output', 'clean_btc_data.csv')
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar for Timeframe selection
    st.sidebar.header("Settings")
    st.sidebar.info(f"📊 Data timeframe: 2 minutes")
    st.sidebar.info(f"📈 Total candles: {len(df)}")
    
    timeframe = st.sidebar.selectbox(
        "Aggregate to Timeframe", 
        ["2min (Raw)", "5min", "15min", "30min", "1H", "4H", "1D"], 
        index=0
    )
    
    # Show signals toggle
    show_signals = st.sidebar.checkbox("Show BUY/SELL Signals", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI Panel", value=True)
    
    # Parse timeframe
    if timeframe == "2min (Raw)":
        df_resampled = df.set_index('datetime').copy()
        st.sidebar.success(f"✅ Showing all {len(df_resampled)} candles")
    else:
        tf = timeframe.split()[0]
        df_resampled = df.resample(tf, on='datetime').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        st.sidebar.success(f"✅ Aggregated to {len(df_resampled)} candles")

    # Calculate Technical Indicators
    if len(df_resampled) > 0:
        df_temp = df_resampled.reset_index()
        df_temp['ema_50'] = calculate_ema(df_temp, 50)
        df_temp['ema_200'] = calculate_ema(df_temp, 200)
        df_temp['rsi'] = calculate_rsi(df_temp, 14)
        df_temp['adx'], df_temp['plus_di'], df_temp['minus_di'] = calculate_adx(df_temp, 14)
        
        # Generate signals
        df_temp['signal'] = None
        df_temp['signal_price'] = None
        
        for i in range(200, len(df_temp)):
            ema50 = df_temp['ema_50'].iloc[i]
            ema200 = df_temp['ema_200'].iloc[i]
            close = df_temp['close'].iloc[i]
            rsi = df_temp['rsi'].iloc[i]
            
            if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
                continue
            
            dist_to_ema50 = ((close - ema50) / ema50) * 100
            
            # LONG SIGNAL (BUY)
            if ema50 > ema200 and abs(dist_to_ema50) <= 2.0 and 40 <= rsi <= 70:
                df_temp.at[i, 'signal'] = 'BUY'
                df_temp.at[i, 'signal_price'] = close
            
            # SHORT SIGNAL (SELL)
            elif ema50 < ema200 and abs(dist_to_ema50) <= 2.0 and 30 <= rsi <= 60:
                df_temp.at[i, 'signal'] = 'SELL'
                df_temp.at[i, 'signal_price'] = close
        
        # Put back in df_resampled
        df_resampled['ema_50'] = df_temp['ema_50'].values
        df_resampled['ema_200'] = df_temp['ema_200'].values
        df_resampled['rsi'] = df_temp['rsi'].values
        df_resampled['adx'] = df_temp['adx'].values
        df_resampled['signal'] = df_temp['signal'].values
        df_resampled['signal_price'] = df_temp['signal_price'].values
    
    # Detect current trend
    if len(df_resampled) > 0:
        latest_ema50 = df_resampled['ema_50'].iloc[-1]
        latest_ema200 = df_resampled['ema_200'].iloc[-1]
        latest_adx = df_resampled['adx'].iloc[-1]
        latest_rsi = df_resampled['rsi'].iloc[-1]
        current_trend, trend_color = detect_trend(latest_ema50, latest_ema200, latest_adx)
        
        # Count total signals (will be filtered for display)
        total_buy = df_resampled['signal'].value_counts().get('BUY', 0)
        total_sell = df_resampled['signal'].value_counts().get('SELL', 0)
        
        # Display trend info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Trend Analysis:**")
        st.sidebar.markdown(f"**EMA 50:** ${latest_ema50:.2f}" if not pd.isna(latest_ema50) else "**EMA 50:** Calculating...")
        st.sidebar.markdown(f"**EMA 200:** ${latest_ema200:.2f}" if not pd.isna(latest_ema200) else "**EMA 200:** Calculating...")
        st.sidebar.markdown(f"**RSI:** {latest_rsi:.2f}" if not pd.isna(latest_rsi) else "**RSI:** Calculating...")
        st.sidebar.markdown(f"<h3 style='color:{trend_color};'>{current_trend}</h3>", unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎯 Trading Signals:**")
        st.sidebar.markdown(f"🟢 **Total BUY:** {total_buy}")
        st.sidebar.markdown(f"🔴 **Total SELL:** {total_sell}")
        st.sidebar.markdown(f"📊 **Total:** {total_buy + total_sell}")
        st.sidebar.info("💡 Only high-quality, well-spaced signals are displayed on chart")

    # Header
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        st.title("BTC/USD")
        st.caption(f"Bitcoin / US Dollar ({timeframe})")
    with c2:
        if len(df_resampled) > 0:
            latest = df_resampled.iloc[-1]
            prev = df_resampled.iloc[-2] if len(df_resampled) > 1 else latest
            change = latest['close'] - prev['close']
            pct_change = (change / prev['close']) * 100
            color = "green" if change >= 0 else "red"
            
            st.markdown(f"""
                <div style="display: flex; align-items: baseline; margin-top: 20px;">
                    <h2 style="margin: 0; color: {color};">${latest['close']:.2f}</h2>
                    <span style="margin-left: 10px; color: {color}; font-size: 1.2em;">
                        {change:+.2f} ({pct_change:+.2f}%)
                    </span>
                </div>
            """, unsafe_allow_html=True)
    with c3:
        if len(df_resampled) > 0:
            st.markdown(f"""
                <div style="margin-top: 20px; padding: 10px; background-color: {trend_color}33; border: 2px solid {trend_color}; border-radius: 10px; text-align: center;">
                    <div style="color: {trend_color}; font-weight: bold; font-size: 1.1em;">{current_trend}</div>
                </div>
            """, unsafe_allow_html=True)

    # Create Figure with subplots
    if show_rsi:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('', '')
        )
        rsi_row = 2
    else:
        fig = go.Figure()
        rsi_row = None

    # Calculate initial visible range
    total_candles = len(df_resampled)
    visible_candles = min(150, total_candles)
    
    if total_candles > 0:
        start_idx = max(0, total_candles - visible_candles)
        initial_range = [df_resampled.index[start_idx], df_resampled.index[-1]]

    # Candlestick
    candlestick = go.Candlestick(
        x=df_resampled.index,
        open=df_resampled['open'],
        high=df_resampled['high'],
        low=df_resampled['low'],
        close=df_resampled['close'],
        name='BTC/USD',
        increasing=dict(line=dict(color='#26a69a', width=1.5), fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350', width=1.5), fillcolor='#ef5350'),
        whiskerwidth=1,
        line=dict(width=1.5),
        hovertext=[
            f"<b>{date.strftime('%Y-%m-%d %H:%M')}</b><br><br>" +
            f"<b style='color:#26a69a'>Open:</b>  ${row['open']:,.2f}<br>" +
            f"<b style='color:#ef5350'>Close:</b> ${row['close']:,.2f}<br>" +
            f"<b style='color:#4a9eff'>High:</b>  ${row['high']:,.2f}<br>" +
            f"<b style='color:#ff9800'>Low:</b>   ${row['low']:,.2f}<br><br>" +
            f"<b>Change:</b> ${row['close'] - row['open']:+,.2f}<br>" +
            f"<b>Volume:</b> {row['volume']:,.0f}"
            for date, row in df_resampled.iterrows()
        ],
        hoverinfo='text',
        showlegend=False
    )
    
    if show_rsi:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add EMAs
    ema50 = go.Scatter(
        x=df_resampled.index,
        y=df_resampled['ema_50'],
        name='EMA 50',
        line=dict(color='#2196F3', width=2),
        hovertemplate='<b>EMA 50:</b> $%{y:.2f}<extra></extra>'
    )
    
    ema200 = go.Scatter(
        x=df_resampled.index,
        y=df_resampled['ema_200'],
        name='EMA 200',
        line=dict(color='#FF6D00', width=2),
        hovertemplate='<b>EMA 200:</b> $%{y:.2f}<extra></extra>'
    )
    
    if show_rsi:
        fig.add_trace(ema50, row=1, col=1)
        fig.add_trace(ema200, row=1, col=1)
    else:
        fig.add_trace(ema50)
        fig.add_trace(ema200)

    # Add BUY/SELL Signals (filtered for quality)
    if show_signals and 'signal' in df_resampled.columns:
        def filter_quality_signals(signals_df, signal_type, min_distance_candles=10):
            """
            Filter signals to show only high-quality, well-spaced ones
            - Remove signals too close together (minimum distance)
            - Prioritize signals with better RSI values
            - Keep signals at key EMA touchpoints
            """
            if signals_df.empty:
                return signals_df
            
            filtered_signals = []
            last_signal_idx = None
            
            for idx, row in signals_df.iterrows():
                # Calculate position in index
                current_pos = signals_df.index.get_loc(idx)
                
                # Check minimum distance from last signal
                if last_signal_idx is None:
                    filtered_signals.append(idx)
                    last_signal_idx = current_pos
                else:
                    distance = current_pos - last_signal_idx
                    
                    # Add signal if far enough OR if it's significantly better
                    if distance >= min_distance_candles:
                        filtered_signals.append(idx)
                        last_signal_idx = current_pos
                    else:
                        # Check if this signal is better (closer RSI to ideal range)
                        prev_signal = signals_df.iloc[last_signal_idx]
                        if signal_type == 'BUY':
                            # Better BUY: RSI closer to 50-60 range
                            ideal_rsi = 55
                            current_score = abs(row['rsi'] - ideal_rsi)
                            prev_score = abs(prev_signal['rsi'] - ideal_rsi)
                            
                            if current_score < prev_score * 0.8:  # 20% better
                                filtered_signals[-1] = idx  # Replace last signal
                                last_signal_idx = current_pos
                        else:  # SELL
                            # Better SELL: RSI closer to 40-45 range
                            ideal_rsi = 42
                            current_score = abs(row['rsi'] - ideal_rsi)
                            prev_score = abs(prev_signal['rsi'] - ideal_rsi)
                            
                            if current_score < prev_score * 0.8:  # 20% better
                                filtered_signals[-1] = idx  # Replace last signal
                                last_signal_idx = current_pos
            
            return signals_df.loc[filtered_signals]
        
        # Filter signals based on timeframe
        if timeframe == "2min (Raw)":
            min_distance = 20  # 40 minutes minimum between signals
        elif timeframe in ["5min", "15min"]:
            min_distance = 10  # Reasonable spacing
        elif timeframe in ["30min", "1H"]:
            min_distance = 5
        else:  # 4H, 1D
            min_distance = 3
        
        buy_signals = df_resampled[df_resampled['signal'] == 'BUY']
        sell_signals = df_resampled[df_resampled['signal'] == 'SELL']
        
        # Apply filtering
        buy_signals = filter_quality_signals(buy_signals, 'BUY', min_distance)
        sell_signals = filter_quality_signals(sell_signals, 'SELL', min_distance)
        
        if not buy_signals.empty:
            buy_scatter = go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.995,
                mode='markers+text',
                name='BUY',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00ff00',
                    line=dict(color='#00cc00', width=2)
                ),
                text=['BUY'] * len(buy_signals),
                textposition='bottom center',
                textfont=dict(size=10, color='#00ff00', family='Arial Black'),
                hovertemplate='<b>🟢 BUY SIGNAL</b><br>Price: $%{y:.2f}<br>RSI: ' + 
                             buy_signals['rsi'].apply(lambda x: f'{x:.2f}').astype(str) + '<extra></extra>'
            )
            if show_rsi:
                fig.add_trace(buy_scatter, row=1, col=1)
            else:
                fig.add_trace(buy_scatter)
        
        if not sell_signals.empty:
            sell_scatter = go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.005,
                mode='markers+text',
                name='SELL',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#ff0000',
                    line=dict(color='#cc0000', width=2)
                ),
                text=['SELL'] * len(sell_signals),
                textposition='top center',
                textfont=dict(size=10, color='#ff0000', family='Arial Black'),
                hovertemplate='<b>🔴 SELL SIGNAL</b><br>Price: $%{y:.2f}<br>RSI: ' + 
                             sell_signals['rsi'].apply(lambda x: f'{x:.2f}').astype(str) + '<extra></extra>'
            )
            if show_rsi:
                fig.add_trace(sell_scatter, row=1, col=1)
            else:
                fig.add_trace(sell_scatter)

    # Add RSI panel
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled['rsi'],
            name='RSI',
            line=dict(color='#9C27B0', width=2),
            hovertemplate='<b>RSI:</b> %{y:.2f}<extra></extra>'
        ), row=2, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1, opacity=0.3)

    # Layout updates
    fig.update_layout(
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#d1d4dc', size=11),
        margin=dict(l=10, r=80, t=50, b=80),
        height=900 if show_rsi else 800,
        xaxis=dict(
            showgrid=True, 
            gridcolor='#363c4e',
            gridwidth=0.5,
            zeroline=False,
            type='date',
            range=initial_range if total_candles > 0 else None,
            rangeslider=dict(
                visible=True,
                bgcolor='#1e222d',
                thickness=0.08,
                range=[df_resampled.index[0], df_resampled.index[-1]] if total_candles > 0 else None
            )
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#363c4e',
            gridwidth=0.5,
            zeroline=False,
            side='right',
            tickformat=',.2f',
            tickfont=dict(size=10),
            fixedrange=False
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e222d',
            font_size=13,
            font_family='Consolas, monospace',
            bordercolor='#363c4e',
            align='left'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0.01,
            bgcolor='#1e222d',
            bordercolor='#363c4e',
            borderwidth=1
        ),
        dragmode='pan'
    )
    
    if show_rsi:
        fig.update_xaxes(showgrid=True, gridcolor='#363c4e', row=2, col=1)
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='#363c4e',
            side='right',
            range=[0, 100],
            row=2, 
            col=1
        )
    
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'doubleClick': 'reset',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'btc_chart',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Trading Strategy:**")
    st.sidebar.markdown("**BUY Signal:**")
    st.sidebar.markdown("• EMA50 > EMA200 (Uptrend)")
    st.sidebar.markdown("• Price near EMA50 (±2%)")
    st.sidebar.markdown("• RSI: 40-70")
    st.sidebar.markdown("")
    st.sidebar.markdown("**SELL Signal:**")
    st.sidebar.markdown("• EMA50 < EMA200 (Downtrend)")
    st.sidebar.markdown("• Price near EMA50 (±2%)")
    st.sidebar.markdown("• RSI: 30-60")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🖱️ Navigation:**")
    st.sidebar.markdown("• **Cliquer-glisser** : Déplacer")
    st.sidebar.markdown("• **Molette** : Zoom")
    st.sidebar.markdown("• **Double-clic** : Réinitialiser")
    
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("No data found. Please check the file path.")