import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import requests
import threading

# ==================== CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="🥓 Bacon Trader Pro - Quantum Ultimate",
    page_icon="🥓",
    initial_sidebar_state="expanded"
)

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "8144444384:AAFlblBgToY7ew50ufZTmNd5qJGRid-TtVA")
CHAT_ID = st.secrets.get("CHAT_ID", "813100618")

def send_telegram_alert(message):
    """Envoie une alerte Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

# ==================== SESSION STATE INIT ====================
if 'day_positions' not in st.session_state:
    st.session_state.day_positions = []

if 'long_positions' not in st.session_state:
    st.session_state.long_positions = []

if 'telegram_enabled' not in st.session_state:
    st.session_state.telegram_enabled = True

if 'last_telegram_sent' not in st.session_state:
    st.session_state.last_telegram_sent = {}

if 'active_signals' not in st.session_state:
    st.session_state.active_signals = []

if 'closed_signals' not in st.session_state:
    st.session_state.closed_signals = []

if 'auto_scan_enabled' not in st.session_state:
    st.session_state.auto_scan_enabled = False

if 'last_auto_scan' not in st.session_state:
    st.session_state.last_auto_scan = None

if 'auto_scan_results' not in st.session_state:
    st.session_state.auto_scan_results = []

# ==================== SIGNAL TRACKER FUNCTIONS ====================
def add_signal_to_tracker(symbol, entry, stop, tp1, tp2, tp3, quantum_score, ai_score, tier, flow, rsi):
    """Ajoute un signal au tracker - Version améliorée"""
    
    try:
        # Vérifie si le signal existe déjà (ACTIVE seulement)
        exists = any(s['symbol'] == symbol and s['status'] == 'ACTIVE' for s in st.session_state.active_signals)
        
        if exists:
            return False
        
        signal = {
            'symbol': symbol,
            'entry_price': float(entry),
            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'stop': float(stop),
            'tp1': float(tp1),
            'tp2': float(tp2),
            'tp3': float(tp3),
            'quantum_score': float(quantum_score),
            'ai_score': float(ai_score),
            'tier': tier,
            'flow': flow,
            'rsi': float(rsi),
            'status': 'ACTIVE',
            'exit_price': None,
            'exit_reason': None,
            'exit_date': None,
            'pnl': 0,
            'pnl_pct': 0
        }
        
        st.session_state.active_signals.append(signal)
        
        # Envoie notification Telegram pour Diamond/Platinum
        if st.session_state.telegram_enabled and tier in ['💎 DIAMOND', '🥇 PLATINUM']:
            msg = f"""
🥓 <b>SIGNAL TRACKED!</b>

{tier} <b>{symbol}</b>

📊 Quantum: {quantum_score:.0f}/300
🤖 AI: {ai_score:.0f}/100

💰 Entry: ${entry:.2f}
🛑 Stop: ${stop:.2f}
🎯 TP1: ${tp1:.2f} | TP2: ${tp2:.2f} | TP3: ${tp3:.2f}

📈 Flow: {flow}
📊 RSI: {rsi:.1f}

⚡ Now tracking automatically!
            """
            send_telegram_alert(msg)
        
        return True
        
    except Exception as e:
        print(f"Error adding signal: {e}")
        return False

def update_signal_status():
    """Update le statut de tous les signaux actifs"""
    to_close = []
    
    for idx, signal in enumerate(st.session_state.active_signals):
        if signal['status'] != 'ACTIVE':
            continue
        
        current_price = get_current_price(signal['symbol'])
        
        if current_price > 0:
            if current_price >= signal['tp3']:
                signal['exit_price'] = signal['tp3']
                signal['exit_reason'] = 'TP3 HIT ✅'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['tp3'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['tp3'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
                
                if st.session_state.telegram_enabled:
                    msg = f"🎯 TP3 HIT!\n\n{signal['tier']} {signal['symbol']}\nEntry: ${signal['entry_price']:.2f}\nExit: ${signal['tp3']:.2f}\n💰 P&L: ${signal['pnl']:.2f} (+{signal['pnl_pct']:.1f}%)"
                    send_telegram_alert(msg)
            
            elif current_price >= signal['tp2']:
                signal['exit_price'] = signal['tp2']
                signal['exit_reason'] = 'TP2 HIT ✅'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['tp2'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['tp2'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
                
                if st.session_state.telegram_enabled:
                    msg = f"🎯 TP2 HIT!\n\n{signal['tier']} {signal['symbol']}\nEntry: ${signal['entry_price']:.2f}\nExit: ${signal['tp2']:.2f}\n💰 P&L: ${signal['pnl']:.2f} (+{signal['pnl_pct']:.1f}%)"
                    send_telegram_alert(msg)
            
            elif current_price >= signal['tp1']:
                signal['exit_price'] = signal['tp1']
                signal['exit_reason'] = 'TP1 HIT ✅'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['tp1'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['tp1'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
                
                if st.session_state.telegram_enabled:
                    msg = f"✅ TP1 HIT!\n\n{signal['tier']} {signal['symbol']}\nEntry: ${signal['entry_price']:.2f}\nExit: ${signal['tp1']:.2f}\n💰 P&L: ${signal['pnl']:.2f} (+{signal['pnl_pct']:.1f}%)"
                    send_telegram_alert(msg)
            
            elif current_price <= signal['stop']:
                signal['exit_price'] = signal['stop']
                signal['exit_reason'] = 'STOP HIT ❌'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['stop'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['stop'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
                
                if st.session_state.telegram_enabled:
                    msg = f"🛑 STOP HIT!\n\n{signal['tier']} {signal['symbol']}\nEntry: ${signal['entry_price']:.2f}\nExit: ${signal['stop']:.2f}\n💸 P&L: ${signal['pnl']:.2f} ({signal['pnl_pct']:.1f}%)"
                    send_telegram_alert(msg)
    
    for idx in reversed(to_close):
        closed_signal = st.session_state.active_signals.pop(idx)
        st.session_state.closed_signals.append(closed_signal)

# ==================== CACHE ====================
@st.cache_data(ttl=300)
def get_live_data(symbol, period="1mo", interval="1d"):
    """Récupère données live avec cache 5min"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if len(hist) > 0:
            return hist
        return None
    except:
        return None

# ==================== MARCHÉS COMPLETS ====================
MARKETS = {
    "🔥 TOP 10 US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
        "META", "TSLA", "BRK-B", "LLY", "V"
    ],
    
    "⭐ TOP 50 US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V",
        "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST", "PG",
        "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO", "ADBE",
        "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS", "CMCSA",
        "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON", "UNP"
    ],
    
    "💯 TOP 100 US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V",
        "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST", "PG",
        "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO", "ADBE",
        "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS", "CMCSA",
        "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON", "UNP",
        "RTX", "AMAT", "LOW", "CAT", "UBER", "INTU", "GE", "MS", "COP", "SPGI",
        "BKNG", "PLD", "AXP", "ISRG", "SYK", "NOW", "T", "BLK", "AMGN", "DE",
        "TJX", "PFE", "BSX", "BMY", "MDT", "GILD", "REGN", "VRTX", "ADP", "CB",
        "SCHW", "MMC", "CI", "LRCX", "ADI", "PANW", "MU", "MDLZ", "SO", "ZTS",
        "KLAC", "ETN", "SLB", "CME", "EOG", "DUK", "ICE", "SNPS", "CDNS", "PH"
    ],
    
    "🇺🇸 S&P 500": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "LLY",
        "V", "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST",
        "PG", "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO",
        "ADBE", "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS",
        "CMCSA", "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON"
    ],
    
    "🚀 NASDAQ 100": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "NFLX",
        "COST", "AMD", "ADBE", "CSCO", "PEP", "QCOM", "INTC", "TXN", "CMCSA", "INTU",
        "AMGN", "AMAT", "HON", "ISRG", "BKNG", "PANW", "ADP", "MU", "LRCX", "KLAC"
    ],
    
    "📊 DOW 30": [
        "AAPL", "MSFT", "AMZN", "UNH", "JNJ", "V", "JPM", "WMT", "PG", "HD",
        "CVX", "MRK", "DIS", "CSCO", "NKE", "MCD", "VZ", "INTC", "KO", "CRM",
        "IBM", "AXP", "CAT", "GS", "HON", "BA", "TRV", "MMM", "WBA", "DOW"
    ],
    
    "⚡ FUTURES": [
        "ES=F", "NQ=F", "YM=F", "RTY=F",
        "GC=F", "SI=F", "CL=F", "NG=F"
    ],
    
    "₿ CRYPTO TOP 20": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD",
        "UNI-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "VET-USD",
        "ICP-USD", "FIL-USD"
    ],
    
    "🇨🇦 TSX TOP 30": [
        "RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNR.TO", "BMO.TO", "CNQ.TO", "CP.TO",
        "SU.TO", "BCE.TO", "CM.TO", "TRP.TO", "ABX.TO", "MFC.TO", "SLF.TO", "FNV.TO",
        "NTR.TO", "WCN.TO", "BAM.TO", "QSR.TO", "SHOP.TO", "GIB-A.TO", "CCL-B.TO",
        "WPM.TO", "MG.TO", "AEM.TO", "POW.TO", "ATD.TO", "IFC.TO", "DOL.TO"
    ],
    
    "🔋 AI & TECH": [
        "NVDA", "AMD", "AVGO", "QCOM", "INTC", "TSM", "AMAT", "LRCX",
        "PLTR", "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW", "AI"
    ],
    
    "🏥 BIOTECH": [
        "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "BMY",
        "AMGN", "GILD", "REGN", "VRTX", "ISRG", "BSX", "MDT", "BIIB"
    ],
    
    "⚡ CLEAN ENERGY": [
        "TSLA", "ENPH", "SEDG", "FSLR", "RUN", "PLUG", "BE", "BLNK", "CHPT", "NEE"
    ],
    
    "🌎 AUTO-SCAN MARKET": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V",
        "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST", "PG",
        "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO", "ADBE",
        "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS", "CMCSA",
        "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON", "UNP"
    ]
}

# ==================== INDICATEURS TECHNIQUES ====================
def calculate_atr(df, period=14):
    try:
        if len(df) < period:
            return 0
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
    except:
        return 0

def calculate_rsi(df, period=14):
    try:
        if len(df) < period:
            return 50
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_macd(df):
    try:
        if len(df) < 26:
            return 0, 0, 0
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1])
    except:
        return 0, 0, 0

# ==================== ORDER FLOW ====================
def calculate_order_flow_advanced(df):
    try:
        if len(df) < 20:
            return {'delta': 0, 'cumulative_delta': 0, 'imbalance': 'NEUTRAL', 
                    'whale_detected': False, 'flow_score': 50, 'vol_ratio': 1}
        
        df = df.copy()
        df['Price_Change'] = df['Close'] - df['Open']
        df['Direction'] = np.where(df['Price_Change'] > 0, 1, 
                                   np.where(df['Price_Change'] < 0, -1, 0))
        df['Delta'] = df['Volume'] * df['Direction']
        df['Cumulative_Delta'] = df['Delta'].cumsum()
        
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_current = df['Volume'].iloc[-1]
        vol_ratio = (vol_current / vol_avg) if vol_avg > 0 else 1
        
        whale_detected = vol_ratio > 3.0
        
        delta_normalized = (df['Cumulative_Delta'].iloc[-1] / df['Volume'].sum()) * 100
        flow_score = min(max(50 + delta_normalized, 0), 100)
        
        if flow_score > 65:
            imbalance = 'BULLISH'
        elif flow_score < 35:
            imbalance = 'BEARISH'
        else:
            imbalance = 'NEUTRAL'
        
        return {
            'delta': float(df['Delta'].iloc[-1]),
            'cumulative_delta': float(df['Cumulative_Delta'].iloc[-1]),
            'imbalance': imbalance,
            'whale_detected': whale_detected,
            'flow_score': float(flow_score),
            'vol_ratio': float(vol_ratio)
        }
    except:
        return {'delta': 0, 'cumulative_delta': 0, 'imbalance': 'NEUTRAL',
                'whale_detected': False, 'flow_score': 50, 'vol_ratio': 1}

# ==================== VOLUME PROFILE ====================
def calculate_volume_profile(df, num_bins=30):
    try:
        if len(df) < 20:
            return {'poc': 0, 'vah': 0, 'val': 0, 'profile': None}
        
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_range = np.linspace(price_min, price_max, num_bins)
        
        volume_at_price = []
        
        for i in range(len(price_range) - 1):
            price_low = price_range[i]
            price_high = price_range[i + 1]
            
            mask = (df['Low'] <= price_high) & (df['High'] >= price_low)
            vol = df[mask]['Volume'].sum()
            
            volume_at_price.append({
                'price': (price_low + price_high) / 2,
                'volume': vol
            })
        
        vp_df = pd.DataFrame(volume_at_price)
        
        if len(vp_df) == 0:
            return {'poc': 0, 'vah': 0, 'val': 0, 'profile': None}
        
        poc_price = float(vp_df.loc[vp_df['volume'].idxmax(), 'price'])
        
        total_vol = vp_df['volume'].sum()
        vp_sorted = vp_df.sort_values('volume', ascending=False).copy()
        vp_sorted['cumsum'] = vp_sorted['volume'].cumsum()
        
        va_prices = vp_sorted[vp_sorted['cumsum'] <= total_vol * 0.7]['price']
        vah = float(va_prices.max()) if len(va_prices) > 0 else poc_price
        val = float(va_prices.min()) if len(va_prices) > 0 else poc_price
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'profile': vp_df
        }
    except:
        return {'poc': 0, 'vah': 0, 'val': 0, 'profile': None}

# ==================== SFP ====================
def detect_sfp(df, lookback=20):
    try:
        if len(df) < lookback + 5:
            return []
        
        sfp_signals = []
        
        for i in range(lookback, len(df)-1):
            window = df.iloc[i-lookback:i]
            current = df.iloc[i]
            next_bar = df.iloc[i+1]
            
            prev_high = window['High'].max()
            prev_low = window['Low'].min()
            
            if (current['Low'] < prev_low and 
                current['Close'] > prev_low and
                next_bar['Close'] > current['Close']):
                sfp_signals.append({
                    'date': df.index[i],
                    'type': 'BULLISH_SFP',
                    'price': float(current['Close']),
                    'level': float(prev_low),
                    'score': 85
                })
            
            if (current['High'] > prev_high and 
                current['Close'] < prev_high and
                next_bar['Close'] < current['Close']):
                sfp_signals.append({
                    'date': df.index[i],
                    'type': 'BEARISH_SFP',
                    'price': float(current['Close']),
                    'level': float(prev_high),
                    'score': 85
                })
        
        return sfp_signals[-5:] if sfp_signals else []
    except:
        return []

# ==================== ELLIOTT WAVE ====================
def detect_elliott_wave(df):
    try:
        if len(df) < 50:
            return {'wave_count': 0, 'direction': 'NEUTRAL', 'confidence': 0}
        
        pivots = []
        window = 5
        
        for i in range(window, len(df)-window):
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                pivots.append({'type': 'HIGH', 'price': df['High'].iloc[i], 'index': i})
            elif df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                pivots.append({'type': 'LOW', 'price': df['Low'].iloc[i], 'index': i})
        
        if len(pivots) < 5:
            return {'wave_count': 0, 'direction': 'NEUTRAL', 'confidence': 0}
        
        recent = pivots[-9:] if len(pivots) >= 9 else pivots
        types = [p['type'] for p in recent]
        
        bullish_pattern = ['LOW', 'HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH', 'LOW']
        bearish_pattern = ['HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH']
        
        if types == bullish_pattern:
            return {'wave_count': 5, 'direction': 'BULLISH', 'confidence': 80,
                   'next_move': 'Correction (ABC) expected'}
        elif types == bearish_pattern:
            return {'wave_count': 5, 'direction': 'BEARISH', 'confidence': 80,
                   'next_move': 'Correction (ABC) expected'}
        
        wave_count = len([i for i in range(len(types)-1) if types[i] != types[i+1]]) // 2
        
        return {'wave_count': wave_count, 'direction': 'IN_PROGRESS', 'confidence': 40}
    except:
        return {'wave_count': 0, 'direction': 'NEUTRAL', 'confidence': 0}

# ==================== SMC ====================
def detect_market_structure(df):
    try:
        if len(df) < 30:
            return {'structure': 'NEUTRAL', 'bias': 'NEUTRAL', 'order_blocks': []}
        
        highs = []
        lows = []
        
        for i in range(5, len(df)-5):
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                highs.append({'price': df['High'].iloc[i], 'index': i})
            if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+6].min():
                lows.append({'price': df['Low'].iloc[i], 'index': i})
        
        if len(highs) < 2 or len(lows) < 2:
            return {'structure': 'NEUTRAL', 'bias': 'NEUTRAL', 'order_blocks': []}
        
        last_high = highs[-1]['price']
        last_low = lows[-1]['price']
        current_price = df['Close'].iloc[-1]
        
        if current_price > last_high:
            structure = 'BULLISH'
            bias = 'BUY'
        elif current_price < last_low:
            structure = 'BEARISH'
            bias = 'SELL'
        else:
            structure = 'RANGING'
            bias = 'NEUTRAL'
        
        order_blocks = []
        for i in range(len(df)-10, len(df)):
            candle = df.iloc[i]
            if candle['Close'] > candle['Open']:
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'top': float(candle['High']),
                    'bottom': float(candle['Low']),
                    'date': df.index[i]
                })
        
        return {
            'structure': structure,
            'bias': bias,
            'order_blocks': order_blocks[-3:],
            'last_high': float(last_high),
            'last_low': float(last_low)
        }
    except:
        return {'structure': 'NEUTRAL', 'bias': 'NEUTRAL', 'order_blocks': []}

# ==================== SETUP 80% WIN RATE ====================
def check_80_percent_setup(df, analysis):
    """Setup 80% Win Rate Ultra-Conservateur"""
    try:
        if len(df) < 50:
            return {'valid': False, 'score': 0, 'reasons': []}
        
        reasons = []
        score = 0
        
        if analysis['quantum_score'] >= 260:
            score += 20
            reasons.append("✅ Quantum Score >= 260")
        else:
            reasons.append(f"❌ Quantum {analysis['quantum_score']:.0f} < 260")
        
        if analysis['order_flow']['imbalance'] == 'BULLISH' and analysis['order_flow']['flow_score'] > 75:
            score += 20
            reasons.append("✅ Strong Bullish Flow")
        else:
            reasons.append(f"❌ Flow: {analysis['order_flow']['imbalance']}")
        
        if 40 <= analysis['rsi'] <= 60:
            score += 15
            reasons.append(f"✅ RSI {analysis['rsi']:.1f} optimal")
        else:
            reasons.append(f"❌ RSI {analysis['rsi']:.1f} out of range")
        
        ema9 = df['Close'].ewm(span=9).mean().iloc[-1]
        ema21 = df['Close'].ewm(span=21).mean().iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        
        if analysis['current_price'] > ema9 > ema21 > ema50:
            score += 15
            reasons.append("✅ Price > EMA9 > EMA21 > EMA50")
        else:
            reasons.append("❌ EMA alignment failed")
        
        if analysis['order_flow']['vol_ratio'] > 2.0:
            score += 10
            reasons.append(f"✅ Volume {analysis['order_flow']['vol_ratio']:.1f}x")
        else:
            reasons.append(f"❌ Volume {analysis['order_flow']['vol_ratio']:.1f}x < 2x")
        
        near_poc = abs(analysis['current_price'] - analysis['volume_profile']['poc']) / analysis['current_price'] < 0.02
        has_bullish_sfp = len(analysis['sfp_signals']) > 0 and analysis['sfp_signals'][-1]['type'] == 'BULLISH_SFP'
        
        if near_poc or has_bullish_sfp:
            score += 10
            if near_poc:
                reasons.append("✅ Near POC support")
            if has_bullish_sfp:
                reasons.append("✅ Bullish SFP detected")
        else:
            reasons.append("❌ No POC/SFP confirmation")
        
        if analysis['smc']['bias'] == 'BUY':
            score += 10
            reasons.append("✅ Bullish Market Structure")
        else:
            reasons.append("❌ Neutral/Bearish structure")
        
        valid = score >= 80
        
        return {
            'valid': valid,
            'score': score,
            'reasons': reasons,
            'confidence': 'ULTRA HIGH' if score >= 90 else 'HIGH'
        }
    except:
        return {'valid': False, 'score': 0, 'reasons': ['Error in calculation']}

# ==================== QUANTUM AI SCORE ====================
def calculate_quantum_ai_score(df, symbol=""):
    if df is None or len(df) < 50:
        return {
            'quantum_score': 0, 'ai_score': 0, 'recommendation': 'WAIT', 'signal': '⚫',
            'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False,
            'atr': 0, 'rsi': 50, 'current_price': 0, 'setup_80': {'valid': False, 'score': 0}
        }
    
    try:
        atr = calculate_atr(df, 14)
        rsi = calculate_rsi(df, 14)
        macd, macd_signal, macd_hist = calculate_macd(df)
        current_price = float(df['Close'].iloc[-1])
        
        order_flow = calculate_order_flow_advanced(df)
        volume_profile = calculate_volume_profile(df)
        sfp_signals = detect_sfp(df)
        elliott = detect_elliott_wave(df)
        smc = detect_market_structure(df)
        
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        
        trend_score = 100 if current_price > sma_20 > sma_50 else (70 if current_price > sma_20 else 30)
        rsi_score = 100 if 45 <= rsi <= 55 else (80 if 40 <= rsi <= 60 else 50)
        macd_score = 100 if macd > macd_signal and macd_hist > 0 else 40
        
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_current = df['Volume'].iloc[-1]
        vol_ratio = (vol_current / vol_avg) if vol_avg > 0 else 1
        volume_score = min(vol_ratio * 50, 100)
        
        flow_score = order_flow['flow_score']
        
        poc_distance = abs(current_price - volume_profile['poc']) / current_price if volume_profile['poc'] > 0 else 0
        vp_score = 100 if poc_distance < 0.02 else 70
        
        sfp_score = 100 if len(sfp_signals) > 0 and sfp_signals[-1]['type'] == 'BULLISH_SFP' else 50
        elliott_score = elliott['confidence']
        smc_score = 100 if smc['bias'] == 'BUY' else (50 if smc['bias'] == 'NEUTRAL' else 20)
        
        quantum_ai = (
            trend_score * 0.15 +
            rsi_score * 0.10 +
            macd_score * 0.10 +
            volume_score * 0.15 +
            flow_score * 0.15 +
            vp_score * 0.10 +
            sfp_score * 0.10 +
            elliott_score * 0.05 +
            smc_score * 0.10
        )
        
        momentum = ((current_price / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
        quantum_score = (quantum_ai / 100) * 250 + min(max(momentum * 5, -50), 50)
        quantum_score = min(max(quantum_score, 0), 300)
        
        if quantum_score >= 250:
            tier = '💎 DIAMOND'
        elif quantum_score >= 220:
            tier = '🥇 PLATINUM'
        elif quantum_score >= 200:
            tier = '🥈 GOLD'
        elif quantum_score >= 180:
            tier = '🥉 SILVER'
        else:
            tier = '🔸 BRONZE'
        
        if quantum_ai >= 85 and order_flow['imbalance'] == 'BULLISH':
            recommendation, signal, confidence = 'STRONG BUY', '🟢', 'VERY HIGH'
        elif quantum_ai >= 70:
            recommendation, signal, confidence = 'BUY', '🟢', 'HIGH'
        elif quantum_ai >= 55:
            recommendation, signal, confidence = 'HOLD', '🟡', 'MEDIUM'
        elif quantum_ai >= 40:
            recommendation, signal, confidence = 'WAIT', '🟠', 'LOW'
        else:
            recommendation, signal, confidence = 'AVOID', '🔴', 'VERY LOW'
        
        result = {
            'quantum_score': round(quantum_score, 1),
            'ai_score': round(quantum_ai, 1),
            'recommendation': recommendation,
            'signal': signal,
            'confidence': confidence,
            'tier': tier,
            'whale_alert': order_flow['whale_detected'],
            'atr': atr,
            'rsi': rsi,
            'current_price': current_price,
            'order_flow': order_flow,
            'volume_profile': volume_profile,
            'sfp_signals': sfp_signals,
            'elliott': elliott,
            'smc': smc
        }
        
        result['setup_80'] = check_80_percent_setup(df, result)
        
        return result
    
    except:
        return {
            'quantum_score': 0, 'ai_score': 0, 'recommendation': 'ERROR', 'signal': '⚫',
            'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False,
            'atr': 0, 'rsi': 50, 'current_price': 0, 'setup_80': {'valid': False, 'score': 0}
        }

def calculate_targets(entry, atr, ai_score):
    stop = round(entry - (atr * 2.0), 2)
    risk = entry - stop
    
    if ai_score >= 80:
        tp1, tp2, tp3 = round(entry + risk * 1.2, 2), round(entry + risk * 2.0, 2), round(entry + risk * 3.5, 2)
    elif ai_score >= 65:
        tp1, tp2, tp3 = round(entry + risk * 1.0, 2), round(entry + risk * 1.8, 2), round(entry + risk * 3.0, 2)
    else:
        tp1, tp2, tp3 = round(entry + risk * 0.8, 2), round(entry + risk * 1.5, 2), round(entry + risk * 2.5, 2)
    
    return stop, tp1, tp2, tp3

# ==================== SCANNER ====================
def scan_market_quantum(symbols, min_score=220, min_ai=75, show_progress=True):
    results = []
    
    if show_progress:
        progress_bar = st.progress(0)
        status = st.empty()
    
    for i, symbol in enumerate(symbols):
        if show_progress:
            status.text(f"🔍 Quantum Scanning {symbol}... ({i+1}/{len(symbols)})")
        
        df = get_live_data(symbol, period="3mo", interval="1d")
        
        if df is not None and len(df) > 50:
            analysis = calculate_quantum_ai_score(df, symbol)
            
            if analysis['quantum_score'] >= min_score and analysis['ai_score'] >= min_ai:
                stop, tp1, tp2, tp3 = calculate_targets(
                    analysis['current_price'],
                    analysis['atr'],
                    analysis['ai_score']
                )
                
                if analysis['tier'] in ['💎 DIAMOND', '🥇 PLATINUM'] and st.session_state.telegram_enabled:
                    last_sent = st.session_state.last_telegram_sent.get(symbol, 0)
                    time_since = time.time() - last_sent
                    
                    if time_since > 3600:
                        telegram_msg = f"""
🥓 <b>BACON TRADER PRO ALERT</b>

{analysis['tier']} <b>{symbol}</b>

📊 Quantum Score: {analysis['quantum_score']:.0f}/300
🤖 AI Score: {analysis['ai_score']:.0f}/100

💰 Entry: ${analysis['current_price']:.2f}
🛑 Stop: ${stop}
🎯 TP1: ${tp1} | TP2: ${tp2} | TP3: ${tp3}

📈 Flow: {analysis['order_flow']['imbalance']}
📊 RSI: {analysis['rsi']:.1f}
{'🐋 WHALE DETECTED!' if analysis['whale_alert'] else ''}

⚡ {analysis['recommendation']}
                        """
                        
                        if send_telegram_alert(telegram_msg):
                            st.session_state.last_telegram_sent[symbol] = time.time()
                
                results.append({
                    'Time': datetime.now().strftime("%H:%M"),
                    'Symbol': symbol,
                    'Quantum': analysis['quantum_score'],
                    'AI': analysis['ai_score'],
                    'Tier': analysis['tier'],
                    'Signal': analysis['signal'],
                    'Entry': round(analysis['current_price'], 2),
                    'Stop': stop,
                    'TP1': tp1,
                    'TP2': tp2,
                    'TP3': tp3,
                    'RSI': round(analysis['rsi'], 1),
                    'Flow': analysis['order_flow']['imbalance'],
                    'Whale': '🐋' if analysis['whale_alert'] else '',
                    '80% Setup': '⭐' if analysis['setup_80']['valid'] else '',
                    'Recommendation': analysis['recommendation']
                })
        
        if show_progress:
            progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.05)
    
    if show_progress:
        progress_bar.empty()
        status.empty()
    
    return pd.DataFrame(results)

# ==================== AUTO-SCAN FUNCTION ====================
def run_auto_scan():
    """Lance un auto-scan toutes les 30 minutes"""
    symbols = MARKETS["🌎 AUTO-SCAN MARKET"]
    results = scan_market_quantum(symbols, min_score=240, min_ai=80, show_progress=False)
    
    if len(results) > 0:
        st.session_state.auto_scan_results = results
        st.session_state.last_auto_scan = datetime.now()
        
        # Auto-track les meilleurs signaux
        for _, row in results.iterrows():
            if row['Tier'] in ['💎 DIAMOND', '🥇 PLATINUM']:
                add_signal_to_tracker(
                    row['Symbol'], row['Entry'], row['Stop'],
                    row['TP1'], row['TP2'], row['TP3'],
                    row['Quantum'], row['AI'], row['Tier'],
                    row['Flow'], row['RSI']
                )

# ==================== BACKTEST ENGINE ====================
def run_backtest(symbol, period="6mo", initial_capital=10000):
    """Backtest Engine Complet"""
    df = get_live_data(symbol, period=period, interval="1d")
    
    if df is None or len(df) < 60:
        return None
    
    trades = []
    equity_curve = [initial_capital]
    balance = initial_capital
    position = None
    
    for i in range(60, len(df)):
        df_slice = df.iloc[:i+1]
        analysis = calculate_quantum_ai_score(df_slice, symbol)
        
        current_price = df_slice['Close'].iloc[-1]
        date = df_slice.index[-1]
        
        if position is None:
            if analysis['recommendation'] in ['STRONG BUY', 'BUY'] and analysis['quantum_score'] >= 220:
                stop, tp1, tp2, tp3 = calculate_targets(current_price, analysis['atr'], analysis['ai_score'])
                
                risk_amount = balance * 0.02
                risk_per_share = current_price - stop
                
                if risk_per_share > 0:
                    qty = int(risk_amount / risk_per_share)
                    
                    if qty > 0:
                        position = {
                            'entry': current_price,
                            'stop': stop,
                            'tp1': tp1,
                            'tp2': tp2,
                            'tp3': tp3,
                            'entry_date': date,
                            'qty': qty,
                            'ai_score': analysis['ai_score'],
                            'tier': analysis['tier']
                        }
        
        elif position is not None:
            exit_reason = None
            exit_price = current_price
            
            if current_price >= position['tp3']:
                exit_reason, exit_price = 'TP3', position['tp3']
            elif current_price >= position['tp2']:
                exit_reason, exit_price = 'TP2', position['tp2']
            elif current_price >= position['tp1']:
                exit_reason, exit_price = 'TP1', position['tp1']
            elif current_price <= position['stop']:
                exit_reason, exit_price = 'STOP', position['stop']
            elif (i - df.index.get_loc(position['entry_date'])) > 20:
                exit_reason = 'TIME'
            
            if exit_reason:
                pnl = (exit_price - position['entry']) * position['qty']
                pnl_pct = ((exit_price / position['entry']) - 1) * 100
                balance += pnl
                
                trades.append({
                    'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                    'Exit Date': date.strftime('%Y-%m-%d'),
                    'Entry': round(position['entry'], 2),
                    'Exit': round(exit_price, 2),
                    'Qty': position['qty'],
                    'Exit Reason': exit_reason,
                    'Tier': position['tier'],
                    'AI Score': position['ai_score'],
                    'P&L': round(pnl, 2),
                    'P&L%': round(pnl_pct, 2),
                    'Result': 'WIN' if pnl > 0 else 'LOSS'
                })
                
                position = None
        
        equity_curve.append(balance)
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        wins = len(trades_df[trades_df['Result'] == 'WIN'])
        losses = len(trades_df[trades_df['Result'] == 'LOSS'])
        win_rate = (wins / len(trades_df)) * 100
        
        avg_win = trades_df[trades_df['Result'] == 'WIN']['P&L'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['Result'] == 'LOSS']['P&L'].mean() if losses > 0 else 0
        
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0
        
        total_pnl = trades_df['P&L'].sum()
        total_pnl_pct = ((balance - initial_capital) / initial_capital) * 100
        
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        stats = {
            'Total Trades': len(trades_df),
            'Wins': wins,
            'Losses': losses,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Total P&L': total_pnl,
            'Total P&L%': total_pnl_pct,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe,
            'Final Balance': balance
        }
        
        return {
            'trades': trades_df,
            'stats': stats,
            'equity_curve': equity_curve,
            'dates': df.index[:len(equity_curve)]
        }
    
    return None

# ==================== PORTFOLIO FUNCTIONS ====================
def get_current_price(symbol):
    df = get_live_data(symbol, '5d', '1d')
    if df is not None and len(df) > 0:
        return float(df['Close'].iloc[-1])
    return 0

def add_day_position(symbol, qty, entry):
    st.session_state.day_positions.append({
        'symbol': symbol,
        'qty': qty,
        'entry': entry,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M')
    })

def add_long_position(symbol, qty, entry):
    st.session_state.long_positions.append({
        'symbol': symbol,
        'qty': qty,
        'entry': entry,
        'date': datetime.now().strftime('%Y-%m-%d')
    })

def remove_position(position_type, index):
    if position_type == 'day':
        st.session_state.day_positions.pop(index)
    else:
        st.session_state.long_positions.pop(index)

def get_portfolio_with_live_prices(position_type='day'):
    positions = st.session_state.day_positions if position_type == 'day' else st.session_state.long_positions
    
    if len(positions) == 0:
        return pd.DataFrame()
    
    results = []
    
    for pos in positions:
        current_price = get_current_price(pos['symbol'])
        
        if current_price > 0:
            pnl = (current_price - pos['entry']) * pos['qty']
            pnl_pct = ((current_price / pos['entry']) - 1) * 100
            
            df = get_live_data(pos['symbol'], '1mo', '1d')
            atr = calculate_atr(df, 14) if df is not None else 0
            
            stop = round(pos['entry'] - (atr * 2.0), 2) if atr > 0 else round(pos['entry'] * 0.98, 2)
            
            if current_price <= stop:
                status = '🔴'
            elif pnl_pct > 5:
                status = '🟢'
            elif pnl_pct > 0:
                status = '🟡'
            else:
                status = '🟠'
            
            results.append({
                'Symbol': pos['symbol'],
                'Qty': pos['qty'],
                'Entry': f"${pos['entry']:.2f}",
                'Current': f"${current_price:.2f}",
                'Stop': f"${stop:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L%': f"{pnl_pct:+.2f}%",
                'Status': status,
                'Date': pos['date']
            })
    
    return pd.DataFrame(results)

def calculate_portfolio_stats(position_type='day'):
    positions = st.session_state.day_positions if position_type == 'day' else st.session_state.long_positions
    
    if len(positions) == 0:
        return {'total_value': 0, 'total_pnl': 0, 'total_pnl_pct': 0}
    
    total_invested = 0
    total_current = 0
    
    for pos in positions:
        current_price = get_current_price(pos['symbol'])
        if current_price > 0:
            total_invested += pos['entry'] * pos['qty']
            total_current += current_price * pos['qty']
    
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    return {
        'total_value': total_current,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct
    }

# ==================== THEME ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1410 100%);
        color: #ff8c00;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #ff8c00;
        font-weight: bold;
        text-shadow: 0 0 10px #ff8c00;
    }
    .diamond-signal {
        background: linear-gradient(135deg, rgba(0,255,255,0.3) 0%, rgba(138,43,226,0.4) 100%);
        border: 4px solid cyan;
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 0 20px cyan;
    }
    .setup-80 {
        background: linear-gradient(135deg, rgba(255,215,0,0.3) 0%, rgba(255,140,0,0.4) 100%);
        border: 4px solid gold;
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 0 20px gold;
    }
    .whale-alert {
        background: linear-gradient(135deg, rgba(255,0,255,0.3) 0%, rgba(255,100,255,0.4) 100%);
        border: 3px solid magenta;
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 6, 2])
with col1:
    st.markdown("# 🥓")
with col2:
    st.markdown("# BACON TRADER PRO")
    st.caption("⚡ QUANTUM ULTIMATE v4.2 | ⭐ 80% Setup | 📱 Telegram | 🧪 Backtest | 🔄 AUTO-SCAN")
with col3:
    st.metric("Status", "LIVE 🔴")

st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ QUANTUM CONTROL")
    
    st.session_state.telegram_enabled = st.toggle("📱 Telegram Alerts", value=st.session_state.telegram_enabled, key="toggle_telegram")
    
    if st.session_state.telegram_enabled:
        st.success("✅ Telegram ON")
    else:
        st.info("📴 Telegram OFF")
    
    st.markdown("---")
    
    # AUTO-SCAN
    st.subheader("🔄 AUTO-SCAN (30min)")
    
    st.session_state.auto_scan_enabled = st.toggle("Enable Auto-Scan", value=st.session_state.auto_scan_enabled, key="toggle_autoscan")
    
    if st.session_state.auto_scan_enabled:
        st.success("✅ Auto-Scan ACTIVE")
        
        if st.session_state.last_auto_scan:
            next_scan = st.session_state.last_auto_scan + timedelta(minutes=30)
            time_until = (next_scan - datetime.now()).total_seconds()
            
            if time_until > 0:
                st.info(f"Next scan in {int(time_until/60)} min")
            else:
                st.warning("Scan en attente...")
                if st.button("🚀 FORCE SCAN NOW"):
                    with st.spinner("Auto-scanning..."):
                        run_auto_scan()
                    st.rerun()
        else:
            if st.button("🚀 START AUTO-SCAN"):
                with st.spinner("Running first scan..."):
                    run_auto_scan()
                st.rerun()
    else:
        st.info("📴 Auto-Scan OFF")
    
    st.markdown("---")
    
    day_stats = calculate_portfolio_stats('day')
    st.subheader("💰 DAY TRADING")
    st.metric("Value", f"${day_stats['total_value']:,.2f}")
    st.metric("P&L", f"${day_stats['total_pnl']:+,.2f}", f"{day_stats['total_pnl_pct']:+.2f}%")
    
    with st.expander("➕ ADD DAY POSITION"):
        with st.form("add_day_pos", clear_on_submit=True):
            day_symbol = st.text_input("Symbol", "", key="input_day_symbol").upper()
            day_qty = st.number_input("Quantity", min_value=1, value=10, key="input_day_qty")
            day_entry = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01, key="input_day_entry")
            day_submit = st.form_submit_button("Add Position", use_container_width=True)
            
            if day_submit and day_symbol:
                add_day_position(day_symbol, day_qty, day_entry)
                st.success(f"✅ Added {day_qty} {day_symbol} @ ${day_entry}")
                st.rerun()
    
    st.markdown("---")
    
    lt_stats = calculate_portfolio_stats('long')
    st.subheader("📊 LONG-TERM (CÉLI)")
    st.metric("Value", f"${lt_stats['total_value']:,.2f}")
    st.metric("P&L", f"${lt_stats['total_pnl']:+,.2f}", f"{lt_stats['total_pnl_pct']:+.2f}%")
    
    with st.expander("➕ ADD LONG POSITION"):
        with st.form("add_long_pos", clear_on_submit=True):
            lt_symbol = st.text_input("Symbol", "", key="input_lt_symbol").upper()
            lt_qty = st.number_input("Quantity", min_value=1, value=50, key="input_lt_qty")
            lt_entry = st.number_input("Entry Price", min_value=0.01, value=150.00, step=0.01, key="input_lt_entry")
            lt_submit = st.form_submit_button("Add Position", use_container_width=True)
            
            if lt_submit and lt_symbol:
                add_long_position(lt_symbol, lt_qty, lt_entry)
                st.success(f"✅ Added {lt_qty} {lt_symbol} @ ${lt_entry}")
                st.rerun()
    
    st.markdown("---")
    
    st.subheader("📊 SIGNAL TRACKER")
    active_count = len(st.session_state.active_signals)
    closed_count = len(st.session_state.closed_signals)
    
    st.metric("Active Signals", active_count)
    st.metric("Closed Trades", closed_count)
    
    if closed_count > 0:
        closed_df = pd.DataFrame(st.session_state.closed_signals)
        wins = len(closed_df[closed_df['pnl'] > 0])
        win_rate = (wins / closed_count) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    st.markdown("---")
    
    st.subheader("🎯 QUANTUM SETTINGS")
    min_quantum = st.slider("Min Quantum Score", 0, 300, 220, 10, key="slider_min_quantum")
    min_ai = st.slider("Min AI Score", 0, 100, 75, 5, key="slider_min_ai")
    
    st.markdown("---")
    
    st.subheader("🔬 FEATURES ACTIVE")
    st.success("✅ Order Flow (ATAS)")
    st.success("✅ Volume Profile")
    st.success("✅ SFP Detection")
    st.success("✅ Elliott Wave")
    st.success("✅ SMC")
    st.success("✅ Whale Detection")
    st.success("✅ 80% Setup Filter")
    st.success("✅ Telegram Alerts")
    st.success("✅ Backtest Engine")
    st.success("✅ Signal Tracker")
    st.success("✅ Auto-Scan (30min)")
    
    st.markdown("---")
    st.caption("🥓 Bacon Trader Pro v4.2 ULTIMATE")
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

# ==================== CHECK AUTO-SCAN ====================
if st.session_state.auto_scan_enabled and st.session_state.last_auto_scan:
    time_since_scan = (datetime.now() - st.session_state.last_auto_scan).total_seconds()
    if time_since_scan >= 1800:  # 30 minutes
        run_auto_scan()

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Dashboard",
    "💼 Portfolio Day",
    "📈 Portfolio LT",
    "🔍 Quantum Scanner",
    "🧠 Smart Money",
    "🧪 Backtest",
    "📊 Signal Tracker"
])

# TAB 1: DASHBOARD
with tab1:
    st.header("🎯 QUANTUM DASHBOARD")
    
    # Update signal status
    update_signal_status()
    
    day_stats = calculate_portfolio_stats('day')
    lt_stats = calculate_portfolio_stats('long')
    total_equity = day_stats['total_value'] + lt_stats['total_value']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Equity", f"${total_equity:,.0f}")
    col2.metric("Day Trading", f"${day_stats['total_value']:,.0f}", f"{day_stats['total_pnl_pct']:+.2f}%")
    col3.metric("Long-Term", f"${lt_stats['total_value']:,.0f}", f"{lt_stats['total_pnl_pct']:+.2f}%")
    col4.metric("Active Signals", len(st.session_state.active_signals))
    col5.metric("Total Positions", len(st.session_state.day_positions) + len(st.session_state.long_positions))
    
    st.markdown("---")
    
    # Auto-scan results
    if st.session_state.auto_scan_enabled and len(st.session_state.auto_scan_results) > 0:
        st.subheader("🔄 LATEST AUTO-SCAN RESULTS")
        st.caption(f"Last scan: {st.session_state.last_auto_scan.strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(st.session_state.auto_scan_results, use_container_width=True, hide_index=True)
        st.markdown("---")
    
    st.subheader("⚡ QUANTUM QUICK SCAN")
    
    quick_symbols = ["NVDA", "TSLA", "AMD", "AAPL", "MSFT"]
    quick_results = []
    
    try:
        for symbol in quick_symbols:
            df = get_live_data(symbol, "1mo", "1d")
            if df is not None and len(df) > 30:
                analysis = calculate_quantum_ai_score(df, symbol)
                if analysis['current_price'] > 0:
                    quick_results.append({
                        'Symbol': symbol,
                        'Price': f"${analysis['current_price']:.2f}",
                        'Quantum': f"{analysis['quantum_score']:.0f}/300",
                        'Tier': analysis['tier'],
                        'Signal': analysis['signal'],
                        'Whale': '🐋' if analysis['whale_alert'] else '',
                        '80%': '⭐' if analysis['setup_80']['valid'] else '',
                        'Action': analysis['recommendation']
                    })
        
        if quick_results:
            st.dataframe(pd.DataFrame(quick_results), use_container_width=True, hide_index=True)
        else:
            st.info("⏰ Markets closed or data unavailable")
    except:
        st.warning("⚠️ Quick scan unavailable")
    
    st.markdown("---")
    
    # DEBUG SECTION
    st.subheader("🔧 DEBUG - Signal Tracker")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Active Signals:** {len(st.session_state.active_signals)}")
        if len(st.session_state.active_signals) > 0:
            for s in st.session_state.active_signals:
                st.write(f"- {s['symbol']} ({s['tier']})")
    
    with col2:
        st.write(f"**Closed Signals:** {len(st.session_state.closed_signals)}")
    
    with col3:
        # Bouton de test
        if st.button("🧪 TEST ADD SIGNAL", key="test_add_signal"):
            test_added = add_signal_to_tracker(
                "TEST", 100.0, 95.0, 105.0, 110.0, 115.0,
                250, 85, "🥇 PLATINUM", "BULLISH", 55.0
            )
            if test_added:
                st.success("✅ Test signal added!")
                st.rerun()
            else:
                st.error("❌ Test failed - Already exists!")

# TAB 2: PORTFOLIO DAY
with tab2:
    st.header("💼 PORTFOLIO DAY TRADING - LIVE")
    
    portfolio_day = get_portfolio_with_live_prices('day')
    
    if len(portfolio_day) > 0:
        st.dataframe(portfolio_day, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🗑️ REMOVE POSITIONS")
        
        cols = st.columns(min(len(st.session_state.day_positions), 4))
        for idx, pos in enumerate(st.session_state.day_positions):
            col_idx = idx % 4
            with cols[col_idx]:
                if st.button(f"❌ {pos['symbol']}", key=f"remove_day_{idx}"):
                    remove_position('day', idx)
                    st.rerun()
    else:
        st.info("📝 No day trading positions. Add one in the sidebar! ➕")

# TAB 3: PORTFOLIO LT
with tab3:
    st.header("📈 PORTFOLIO LONG-TERM (CÉLI) - LIVE")
    
    portfolio_lt = get_portfolio_with_live_prices('long')
    
    if len(portfolio_lt) > 0:
        st.dataframe(portfolio_lt, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🗑️ REMOVE POSITIONS")
        
        cols = st.columns(min(len(st.session_state.long_positions), 4))
        for idx, pos in enumerate(st.session_state.long_positions):
            col_idx = idx % 4
            with cols[col_idx]:
                if st.button(f"❌ {pos['symbol']}", key=f"remove_lt_{idx}"):
                    remove_position('long', idx)
                    st.rerun()
    else:
        st.info("📝 No long-term positions. Add one in the sidebar! ➕")

# TAB 4: QUANTUM SCANNER
with tab4:
    st.header("🔍 QUANTUM SCANNER")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Configuration")
        market = st.selectbox("Select Market", list(MARKETS.keys()), key="scanner_market")
        
        show_80_only = st.checkbox("⭐ Show only 80% Setups", key="scanner_checkbox_80")
        
        scan_button = st.button("🚀 QUANTUM SCAN", type="primary", use_container_width=True, key="btn_quantum_scan")
    
    with col2:
        if scan_button:
            st.subheader(f"💎 Results: {market}")
            
            symbols = MARKETS[market]
            
            with st.spinner(f"Quantum scanning {len(symbols)} symbols..."):
                results = scan_market_quantum(symbols, min_quantum, min_ai, show_progress=True)
            
            if len(results) > 0:
                if show_80_only:
                    results = results[results['80% Setup'] == '⭐']
                
                st.success(f"✅ Found {len(results)} signals!")
                
                # Filtres
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filter_tier = st.multiselect(
                        "Filter by Tier",
                        ["💎 DIAMOND", "🥇 PLATINUM", "🥈 GOLD", "🥉 SILVER", "🔸 BRONZE"],
                        default=["💎 DIAMOND", "🥇 PLATINUM"],
                        key="filter_tier"
                    )
                
                with col2:
                    filter_flow = st.multiselect(
                        "Filter by Flow",
                        ["BULLISH", "NEUTRAL", "BEARISH"],
                        default=["BULLISH"],
                        key="filter_flow"
                    )
                
                with col3:
                    show_whale_only = st.checkbox("🐋 Show Whales Only", key="filter_whale")
                
                # Applique les filtres
                filtered_results = results.copy()
                
                if filter_tier:
                    filtered_results = filtered_results[filtered_results['Tier'].isin(filter_tier)]
                
                if filter_flow:
                    filtered_results = filtered_results[filtered_results['Flow'].isin(filter_flow)]
                
                if show_whale_only:
                    filtered_results = filtered_results[filtered_results['Whale'] == '🐋']
                
                st.markdown(f"**Showing {len(filtered_results)} of {len(results)} signals**")
                
                # Affiche les résultats
                st.dataframe(filtered_results, use_container_width=True, hide_index=True, height=400)
                
                # Highlight DIAMOND et PLATINUM
                diamond_signals = filtered_results[filtered_results['Tier'] == '💎 DIAMOND']
                platinum_signals = filtered_results[filtered_results['Tier'] == '🥇 PLATINUM']
                
                if len(diamond_signals) > 0:
                    st.markdown("### 💎 DIAMOND SIGNALS")
                    for _, row in diamond_signals.iterrows():
                        st.markdown(f"""
                        <div class="diamond-signal">
                        <h3>💎 {row['Symbol']} - DIAMOND TIER</h3>
                        <p><b>Quantum Score:</b> {row['Quantum']:.0f}/300 | <b>AI Score:</b> {row['AI']:.0f}/100</p>
                        <p><b>Entry:</b> ${row['Entry']} | <b>Stop:</b> ${row['Stop']}</p>
                        <p><b>TP1:</b> ${row['TP1']} | <b>TP2:</b> ${row['TP2']} | <b>TP3:</b> ${row['TP3']}</p>
                        <p><b>Flow:</b> {row['Flow']} | <b>RSI:</b> {row['RSI']:.1f}</p>
                        <p><b>{row['Recommendation']}</b> {row['Signal']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(platinum_signals) > 0:
                    st.markdown("### 🥇 PLATINUM SIGNALS")
                    for _, row in platinum_signals.iterrows():
                        st.markdown(f"""
                        <div class="setup-80">
                        <h3>🥇 {row['Symbol']} - PLATINUM TIER</h3>
                        <p><b>Quantum Score:</b> {row['Quantum']:.0f}/300 | <b>AI Score:</b> {row['AI']:.0f}/100</p>
                        <p><b>Entry:</b> ${row['Entry']} | <b>Stop:</b> ${row['Stop']}</p>
                        <p><b>TP1:</b> ${row['TP1']} | <b>TP2:</b> ${row['TP2']} | <b>TP3:</b> ${row['TP3']}</p>
                        <p><b>Flow:</b> {row['Flow']} | <b>RSI:</b> {row['RSI']:.1f}</p>
                        <p><b>{row['Recommendation']}</b> {row['Signal']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("➕ TRACK SIGNALS")
                st.caption("Click track button to add signals to Signal Tracker for automatic TP/SL monitoring")
                
                # BOUTON TRACK ALL - VERSION CORRIGÉE
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    if st.button("🚀 TRACK ALL", type="primary", use_container_width=True, key="track_all_btn"):
                        added_count = 0
                        skipped_count = 0
                        
                        for idx, row in filtered_results.iterrows():
                            # Vérifie si déjà tracké
                            already_tracked = any(
                                s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' 
                                for s in st.session_state.active_signals
                            )
                            
                            if not already_tracked:
                                success = add_signal_to_tracker(
                                    row['Symbol'],
                                    row['Entry'],
                                    row['Stop'],
                                    row['TP1'],
                                    row['TP2'],
                                    row['TP3'],
                                    row['Quantum'],
                                    row['AI'],
                                    row['Tier'],
                                    row['Flow'],
                                    row['RSI']
                                )
                                if success:
                                    added_count += 1
                            else:
                                skipped_count += 1
                        
                        # Update et rerun
                        if added_count > 0:
                            update_signal_status()
                            st.success(f"✅ Added {added_count} signals! (Skipped {skipped_count} already tracked)")
                            time.sleep(1.5)
                            st.rerun()
                        elif skipped_count > 0:
                            st.warning(f"⚠️ All {skipped_count} signals are already tracked!")
                        else:
                            st.info("No signals to track")
                
                with col2:
                    # Bouton pour clear tous les signaux trackés
                    if st.button("🗑️ CLEAR ALL", type="secondary", use_container_width=True, key="clear_tracked_btn"):
                        if len(st.session_state.active_signals) > 0:
                            st.session_state.active_signals = []
                            st.success("✅ Cleared all tracked signals!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info("No signals to clear")
                
                with col3:
                    st.info(f"📊 {len(filtered_results)} signals | {len(st.session_state.active_signals)} tracked")
                
                st.markdown("---")
                
                # TABLE DES SIGNAUX AVEC STATUS
                st.subheader("📋 SIGNAL LIST")
                
                # Prépare les données pour affichage
                signal_display = []
                
                for idx, row in filtered_results.iterrows():
                    # Check si déjà tracké
                    is_tracked = any(
                        s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' 
                        for s in st.session_state.active_signals
                    )
                    
                    signal_display.append({
                        'Status': '✅ TRACKED' if is_tracked else '⚪ NEW',
                        'Tier': row['Tier'],
                        'Symbol': row['Symbol'],
                        'Quantum': f"{row['Quantum']:.0f}",
                        'AI': f"{row['AI']:.0f}",
                        'Entry': f"${row['Entry']}",
                        'Stop': f"${row['Stop']}",
                        'TP1': f"${row['TP1']}",
                        'TP2': f"${row['TP2']}",
                        'TP3': f"${row['TP3']}",
                        'Flow': row['Flow'],
                        'RSI': f"{row['RSI']:.1f}",
                        'Whale': row['Whale'],
                        '80%': row['80% Setup'],
                        'Action': row['Recommendation']
                    })
                
                if signal_display:
                    st.dataframe(
                        pd.DataFrame(signal_display), 
                        use_container_width=True, 
                        hide_index=True, 
                        height=400
                    )
                    
                    # TRACK INDIVIDUAL - VERSION SIMPLE
                    st.markdown("---")
                    st.subheader("➕ TRACK INDIVIDUAL SIGNALS")
                    
                    # Liste déroulante pour sélectionner un signal
                    untracked_signals = []
                    for idx, row in filtered_results.iterrows():
                        is_tracked = any(
                            s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' 
                            for s in st.session_state.active_signals
                        )
                        if not is_tracked:
                            untracked_signals.append(row['Symbol'])
                    
                    if len(untracked_signals) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            selected_symbol = st.selectbox(
                                "Select a signal to track:",
                                untracked_signals,
                                key="select_individual_signal"
                            )
                        
                        with col2:
                            if st.button("➕ TRACK THIS", type="secondary", use_container_width=True, key="track_individual_btn"):
                                # Trouve le signal sélectionné
                                selected_row = filtered_results[filtered_results['Symbol'] == selected_symbol].iloc[0]
                                
                                success = add_signal_to_tracker(
                                    selected_row['Symbol'],
                                    selected_row['Entry'],
                                    selected_row['Stop'],
                                    selected_row['TP1'],
                                    selected_row['TP2'],
                                    selected_row['TP3'],
                                    selected_row['Quantum'],
                                    selected_row['AI'],
                                    selected_row['Tier'],
                                    selected_row['Flow'],
                                    selected_row['RSI']
                                )
                                
                                if success:
                                    update_signal_status()
                                    st.success(f"✅ {selected_symbol} tracked!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to track {selected_symbol}")
                    else:
                        st.info("✅ All signals are already tracked!")
                else:
                    st.warning("No signals to display")
            else:
                st.info("📊 No signals found matching your criteria. Try lowering the thresholds!")

# TAB 5: SMART MONEY
with tab5:
    st.header("🧠 SMART MONEY ANALYSIS")
    
    symbol = st.text_input("Enter Symbol", "NVDA", key="sm_symbol").upper()
    
    if st.button("🔍 ANALYZE", type="primary", key="sm_analyze"):
        df = get_live_data(symbol, "3mo", "1d")
        
        if df is not None and len(df) > 50:
            analysis = calculate_quantum_ai_score(df, symbol)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Quantum Score", f"{analysis['quantum_score']:.0f}/300")
            col2.metric("AI Score", f"{analysis['ai_score']:.0f}/100")
            col3.metric("Tier", analysis['tier'])
            col4.metric("Recommendation", analysis['recommendation'])
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Order Flow Analysis")
                of = analysis['order_flow']
                st.write(f"**Imbalance:** {of['imbalance']}")
                st.write(f"**Flow Score:** {of['flow_score']:.1f}/100")
                st.write(f"**Volume Ratio:** {of['vol_ratio']:.2f}x")
                st.write(f"**Delta:** {of['delta']:,.0f}")
                st.write(f"**Cumulative Delta:** {of['cumulative_delta']:,.0f}")
                
                if analysis['whale_alert']:
                    st.markdown("""
                    <div class="whale-alert">
                    🐋 WHALE DETECTED!<br>
                    Abnormal volume activity detected!
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📈 Volume Profile")
                vp = analysis['volume_profile']
                st.write(f"**POC (Point of Control):** ${vp['poc']:.2f}")
                st.write(f"**VAH (Value Area High):** ${vp['vah']:.2f}")
                st.write(f"**VAL (Value Area Low):** ${vp['val']:.2f}")
                st.write(f"**Current Price:** ${analysis['current_price']:.2f}")
                
                distance_poc = abs(analysis['current_price'] - vp['poc']) / analysis['current_price'] * 100
                st.write(f"**Distance from POC:** {distance_poc:.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌊 Elliott Wave")
                ew = analysis['elliott']
                st.write(f"**Wave Count:** {ew['wave_count']}")
                st.write(f"**Direction:** {ew['direction']}")
                st.write(f"**Confidence:** {ew['confidence']}%")
                if 'next_move' in ew:
                    st.write(f"**Next Move:** {ew['next_move']}")
            
            with col2:
                st.subheader("📐 Market Structure (SMC)")
                smc = analysis['smc']
                st.write(f"**Structure:** {smc['structure']}")
                st.write(f"**Bias:** {smc['bias']}")
                if 'last_high' in smc:
                    st.write(f"**Last High:** ${smc['last_high']:.2f}")
                if 'last_low' in smc:
                    st.write(f"**Last Low:** ${smc['last_low']:.2f}")
            
            st.markdown("---")
            
            st.subheader("⚡ SFP (Swing Failure Pattern)")
            sfp = analysis['sfp_signals']
            if len(sfp) > 0:
                for s in sfp:
                    st.write(f"**{s['type']}** at ${s['price']:.2f} (Level: ${s['level']:.2f}) - Score: {s['score']}")
            else:
                st.info("No SFP detected in recent bars")
            
            st.markdown("---")
            
            st.subheader("⭐ 80% Win Rate Setup")
            setup = analysis['setup_80']
            st.write(f"**Valid:** {'✅ YES' if setup['valid'] else '❌ NO'}")
            st.write(f"**Score:** {setup['score']}/100")
            st.write("**Reasons:**")
            for reason in setup['reasons']:
                st.write(f"- {reason}")
        else:
            st.error("❌ Unable to fetch data for this symbol")

# TAB 6: BACKTEST
with tab6:
    st.header("🧪 BACKTEST ENGINE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bt_symbol = st.text_input("Symbol", "NVDA", key="bt_symbol").upper()
    
    with col2:
        bt_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1, key="bt_period")
    
    with col3:
        bt_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000, key="bt_capital")
    
    if st.button("🚀 RUN BACKTEST", type="primary", key="bt_run"):
        with st.spinner(f"Running backtest on {bt_symbol}..."):
            results = run_backtest(bt_symbol, bt_period, bt_capital)
        
        if results:
            stats = results['stats']
            
            st.success("✅ Backtest completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", stats['Total Trades'])
            col2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
            col3.metric("Profit Factor", f"{stats['Profit Factor']:.2f}")
            col4.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Balance", f"${stats['Final Balance']:,.2f}")
            col2.metric("Total P&L", f"${stats['Total P&L']:,.2f}", f"{stats['Total P&L%']:+.2f}%")
            col3.metric("Avg Win", f"${stats['Avg Win']:,.2f}")
            col4.metric("Max Drawdown", f"{stats['Max Drawdown']:.2f}%")
            
            st.markdown("---")
            
            st.subheader("📈 EQUITY CURVE")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['dates'],
                y=results['equity_curve'],
                mode='lines',
                name='Equity',
                line=dict(color='#00ff00', width=2)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title="Date",
                yaxis_title="Balance ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("📋 TRADE LOG")
            st.dataframe(results['trades'], use_container_width=True, hide_index=True, height=400)
        else:
            st.error("❌ Backtest failed. Not enough data or invalid symbol.")

# TAB 7: SIGNAL TRACKER
with tab7:
    st.header("📊 SIGNAL TRACKER - Live Monitoring")
    
    # Update tous les signaux
    update_signal_status()
    
    # Stats globales
    total_active = len(st.session_state.active_signals)
    total_closed = len(st.session_state.closed_signals)
    
    if total_closed > 0:
        closed_df = pd.DataFrame(st.session_state.closed_signals)
        wins = len(closed_df[closed_df['pnl'] > 0])
        losses = len(closed_df[closed_df['pnl'] <= 0])
        win_rate = (wins / total_closed) * 100
        total_pnl = closed_df['pnl'].sum()
        total_pnl_pct = closed_df['pnl_pct'].mean()
    else:
        wins, losses, win_rate, total_pnl, total_pnl_pct = 0, 0, 0, 0, 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active Signals", total_active)
    col2.metric("Closed Signals", total_closed)
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Total P&L", f"${total_pnl:.2f}", f"{total_pnl_pct:.1f}%")
    col5.metric("W/L Ratio", f"{wins}/{losses}")
    
    st.markdown("---")
    
    # Active Signals
    st.subheader("🔴 ACTIVE SIGNALS - Live Tracking")
    
    if len(st.session_state.active_signals) > 0:
        active_data = []
        
        for signal in st.session_state.active_signals:
            current_price = get_current_price(signal['symbol'])
            
            if current_price > 0:
                unrealized_pnl = current_price - signal['entry_price']
                unrealized_pnl_pct = ((current_price / signal['entry_price']) - 1) * 100
                
                # Distance to targets
                to_tp1 = ((signal['tp1'] / current_price) - 1) * 100
                to_tp2 = ((signal['tp2'] / current_price) - 1) * 100
                to_tp3 = ((signal['tp3'] / current_price) - 1) * 100
                to_stop
