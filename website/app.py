import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Technical Indicators Functions
def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['close'].ewm(span=period, adjust=False).mean()

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
    # Calculate +DM and -DM
    high_diff = data['high'].diff()
    low_diff = -data['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate ATR
    atr = calculate_atr(data, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # Calculate DX and ADX
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
    # Tente de charger le fichier avec signaux, sinon fallback sur le brut
    try:
        path_signals = r"C:\Users\abdel\OneDrive\Bureau\Prof_karim_Project\data\output\clean_btc_data_signals.csv"
        if os.path.exists(path_signals):
            df = pd.read_csv(path_signals)
        else:
            df = pd.read_csv(r"C:\Users\abdel\OneDrive\Bureau\Prof_karim_Project\data\output\clean_btc_data.csv")
        df.columns = df.columns.str.lower()
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

import os
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
    
    # Parse timeframe
    if timeframe == "2min (Raw)":
        # Use original 2-minute data as-is
        df_resampled = df.set_index('datetime').copy()
        st.sidebar.success(f"✅ Showing all {len(df_resampled)} candles")
    else:
        # Resample data to create aggregated candles
        tf = timeframe.split()[0]  # Remove "(Raw)" if present
        df_resampled = df.resample(tf, on='datetime').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        st.sidebar.success(f"✅ Aggregated to {len(df_resampled)} candles")

    # Calculate Technical Indicators
    # Make sure we have the required data
    if len(df_resampled) > 0:
        # Reset index temporarily for calculations, then restore
        df_temp = df_resampled.reset_index()
        df_temp['ema_50'] = calculate_ema(df_temp, 50)
        df_temp['ema_200'] = calculate_ema(df_temp, 200)
        df_temp['adx'], df_temp['plus_di'], df_temp['minus_di'] = calculate_adx(df_temp, 14)
        
        # Put back in df_resampled
        df_resampled['ema_50'] = df_temp['ema_50'].values
        df_resampled['ema_200'] = df_temp['ema_200'].values
        df_resampled['adx'] = df_temp['adx'].values
        df_resampled['plus_di'] = df_temp['plus_di'].values
        df_resampled['minus_di'] = df_temp['minus_di'].values
    
    # Detect current trend
    if len(df_resampled) > 0:
        latest_ema50 = df_resampled['ema_50'].iloc[-1]
        latest_ema200 = df_resampled['ema_200'].iloc[-1]
        latest_adx = df_resampled['adx'].iloc[-1]
        current_trend, trend_color = detect_trend(latest_ema50, latest_ema200, latest_adx)
        
        # Display trend info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Trend Analysis:**")
        st.sidebar.markdown(f"**EMA 50:** ${latest_ema50:.2f}" if not pd.isna(latest_ema50) else "**EMA 50:** Calculating...")
        st.sidebar.markdown(f"**EMA 200:** ${latest_ema200:.2f}" if not pd.isna(latest_ema200) else "**EMA 200:** Calculating...")
        st.sidebar.markdown(f"<h3 style='color:{trend_color};'>{current_trend}</h3>", unsafe_allow_html=True)

    # Header
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        st.title("BTC/USD")
        st.caption(f"Bitcoin / US Dollar ({timeframe})")
    with c2:
        # Display latest price stats
        if len(df_resampled) > 0:
            latest = df_resampled.iloc[-1]
            prev = df_resampled.iloc[-2] if len(df_resampled) > 1 else latest
            change = latest['close'] - prev['close']
            pct_change = (change / prev['close']) * 100
            color = "green" if change >= 0 else "red"
            
            st.markdown(f"""
                <div style="display: flex; align-items: baseline; margin-top: 20px;">
                    <h2 style="margin: 0; color: {color};">{latest['close']:.2f}</h2>
                    <span style="margin-left: 10px; color: {color}; font-size: 1.2em;">
                        {change:+.2f} ({pct_change:+.2f}%)
                    </span>
                </div>
            """, unsafe_allow_html=True)
    with c3:
        # Display trend badge
        if len(df_resampled) > 0:
            st.markdown(f"""
                <div style="margin-top: 20px; padding: 10px; background-color: {trend_color}33; border: 2px solid {trend_color}; border-radius: 10px; text-align: center;">
                    <div style="color: {trend_color}; font-weight: bold; font-size: 1.1em;">{current_trend}</div>
                </div>
            """, unsafe_allow_html=True)

    # Create Figure with candlesticks and EMA only
    fig = go.Figure()

    # Calculate initial visible range (show last 100-150 candles for clarity)
    total_candles = len(df_resampled)
    visible_candles = min(150, total_candles)  # Show max 150 candles initially
    
    # Get the date range for initial view
    if total_candles > 0:
        start_idx = max(0, total_candles - visible_candles)
        initial_range = [df_resampled.index[start_idx], df_resampled.index[-1]]

    # Candlestick with detailed hover info and professional styling
    fig.add_trace(go.Candlestick(
        x=df_resampled.index,
        open=df_resampled['open'],
        high=df_resampled['high'],
        low=df_resampled['low'],
        close=df_resampled['close'],
        name='BTC/USD',
        increasing=dict(
            line=dict(color='#26a69a', width=1.5),
            fillcolor='#26a69a'
        ),
        decreasing=dict(
            line=dict(color='#ef5350', width=1.5),
            fillcolor='#ef5350'
        ),
        whiskerwidth=1,  # Width of the wicks
        line=dict(width=1.5),  # Border width of candle body
        hovertext=[
            f"<b>{date.strftime('%Y-%m-%d %H:%M')}</b><br><br>" +
            f"<b style='color:#26a69a'>Open:</b>  ${row['open']:,.2f}<br>" +
            f"<b style='color:#ef5350'>Close:</b> ${row['close']:,.2f}<br>" +
            f"<b style='color:#4a9eff'>High:</b>  ${row['high']:,.2f}<br>" +
            f"<b style='color:#ff9800'>Low:</b>   ${row['low']:,.2f}<br><br>" +
            f"<b>Change:</b> ${row['close'] - row['open']:+,.2f}<br>" +
            f"<b>Range:</b> ${row['high'] - row['low']:,.2f}<br>" +
            f"<b>Volume:</b> {row['volume']:,.0f}"
            for date, row in df_resampled.iterrows()
        ],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add EMA 50
    fig.add_trace(go.Scatter(
        x=df_resampled.index,
        y=df_resampled['ema_50'],
        name='EMA 50',
        line=dict(color='#2196F3', width=2),
        hovertemplate='<b>EMA 50:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add EMA 200
    fig.add_trace(go.Scatter(
        x=df_resampled.index,
        y=df_resampled['ema_200'],
        name='EMA 200',
        line=dict(color='#FF6D00', width=2),
        hovertemplate='<b>EMA 200:</b> $%{y:.2f}<extra></extra>'
    ))
    

    # Add trend change annotations
    trend_changes = []
    for i in range(200, len(df_resampled)):
        current_trend = "up" if df_resampled['ema_50'].iloc[i] > df_resampled['ema_200'].iloc[i] else "down"
        prev_trend = "up" if df_resampled['ema_50'].iloc[i-1] > df_resampled['ema_200'].iloc[i-1] else "down"
        if current_trend != prev_trend and df_resampled['adx'].iloc[i] >= 25:
            trend_changes.append({
                'x': df_resampled.index[i],
                'y': df_resampled['high'].iloc[i],
                'text': '📈 Uptrend' if current_trend == "up" else '📉 Downtrend',
                'color': '#26a69a' if current_trend == "up" else '#ef5350'
            })
    for change in trend_changes:
        fig.add_annotation(
            x=change['x'],
            y=change['y'],
            text=change['text'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=change['color'],
            ax=0,
            ay=-40,
            bgcolor=change['color'],
            font=dict(color='white', size=10),
            bordercolor=change['color'],
            borderwidth=2,
            borderpad=4
        )

    # === Affichage des signaux BUY/SELL (max 3 à 5 par intervalle) ===
    if 'signal' in df_resampled.columns:
        # Choix de l'intervalle pour le groupement (dépend du timeframe sélectionné)
        if timeframe == "2min (Raw)":
            group_freq = '1H'  # Grouper par heure sur du 2min
        else:
            group_freq = tf  # Utilise le même intervalle que l'agrégation

        def select_top_signals(df_signals, n=3):
            # Pour chaque intervalle, on prend les premiers signaux (ou les plus espacés)
            if df_signals.empty:
                return df_signals
            df_signals = df_signals.copy()
            df_signals['interval'] = df_signals.index.to_series().dt.floor(group_freq)
            return df_signals.groupby('interval').head(n)

        buy_signals = df_resampled[df_resampled['signal'] == 'BUY']
        sell_signals = df_resampled[df_resampled['signal'] == 'SELL']
        buy_signals = select_top_signals(buy_signals, n=3)
        sell_signals = select_top_signals(sell_signals, n=3)
        # BUY: flèche verte sous le chandelier
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.995,
                mode='markers+text',
                name='BUY',
                marker=dict(symbol='triangle-up', size=8, color='#00ff00', line=dict(color='#008800', width=1)),
                text=['BUY'] * len(buy_signals),
                textposition='bottom center',
                textfont=dict(size=8, color='#00ff00', family='Arial Black'),
                hovertemplate='<b>BUY SIGNAL</b><br>Price: $%{y:.2f}<extra></extra>'
            ))
        # SELL: flèche rouge au-dessus du chandelier
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.005,
                mode='markers+text',
                name='SELL',
                marker=dict(symbol='triangle-down', size=8, color='#ff0000', line=dict(color='#880000', width=1)),
                text=['SELL'] * len(sell_signals),
                textposition='top center',
                textfont=dict(size=8, color='#ff0000', family='Arial Black'),
                hovertemplate='<b>SELL SIGNAL</b><br>Price: $%{y:.2f}<extra></extra>'
            ))

    # Layout updates for "TradingView" feel with full data display
    fig.update_layout(
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#d1d4dc', size=11),
        margin=dict(l=10, r=80, t=50, b=80),
        height=800,
        xaxis=dict(
            showgrid=True, 
            gridcolor='#363c4e',
            gridwidth=0.5,
            zeroline=False,
            type='date',
            range=initial_range if total_candles > 0 else None,  # Set initial visible range
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
            side='right',  # Price on right like TradingView
            tickformat=',.2f',
            tickfont=dict(size=10),
            fixedrange=False  # Allow vertical zoom
        ),
        hovermode='closest',
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
        dragmode='pan'  # Enable panning by default
    )
    
    # Configure modebar for better interaction
    config = {
        'scrollZoom': True,  # Enable scroll to zoom
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawopenpath',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'doubleClick': 'reset',  # Double-click to reset view
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'btc_chart',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    # Info about interaction
    st.sidebar.markdown("---")
    st.sidebar.markdown("**� Structure du Chandelier:**")
    st.sidebar.markdown("🟢 **Vert** : Haussier (Close > Open)")
    st.sidebar.markdown("🔴 **Rouge** : Baissier (Close < Open)")
    st.sidebar.markdown("• **Corps** : Open ↔ Close")
    st.sidebar.markdown("• **Mèche haute** : High")
    st.sidebar.markdown("• **Mèche basse** : Low")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Indicateurs:**")
    st.sidebar.markdown("🔵 **EMA 50** : Tendance court terme")
    st.sidebar.markdown("🟠 **EMA 200** : Tendance long terme")
    st.sidebar.markdown("")
    st.sidebar.markdown("**📈 Détection de Tendance:**")
    st.sidebar.markdown("• EMA 50 > EMA 200 → Uptrend")
    st.sidebar.markdown("• EMA 50 < EMA 200 → Downtrend")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🖱️ Navigation:**")
    st.sidebar.markdown("• **Cliquer-glisser** : Déplacer gauche/droite")
    st.sidebar.markdown("• **Molette** : Zoomer/Dézoomer")
    st.sidebar.markdown("• **Barre en bas** : Naviguer rapidement")
    st.sidebar.markdown("• **Double-clic** : Réinitialiser la vue")
    st.sidebar.markdown("• **Shift + Glisser** : Sélectionner zone")
    
    st.plotly_chart(fig, width='stretch', config=config)

else:
    st.warning("No data found. Please check the file path.")
