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

if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

if 'last_scan_count' not in st.session_state:
    st.session_state.last_scan_count = 0

if 'last_scan_signals' not in st.session_state:
    st.session_state.last_scan_signals = 0

if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# ==================== SIGNAL TRACKER FUNCTIONS ====================
def add_signal_to_tracker(symbol, entry, stop, tp1, tp2, tp3, quantum_score, ai_score, tier, flow, rsi):
    """Ajoute un signal au tracker"""
    try:
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
        st.error(f"❌ Error adding signal: {e}")
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

# ==================== MARCHÉS ====================
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
    
    "🚀 NASDAQ 100": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "NFLX",
        "COST", "AMD", "ADBE", "CSCO", "PEP", "QCOM", "INTC", "TXN", "CMCSA", "INTU",
        "AMGN", "AMAT", "HON", "ISRG", "BKNG", "PANW", "ADP", "MU", "LRCX", "KLAC",
        "REGN", "GILD", "VRTX", "SNPS", "CDNS", "MRVL", "PYPL", "CRWD", "ABNB", "FTNT"
    ],
    
    "₿ CRYPTO TOP 20": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD",
        "UNI-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "VET-USD",
        "ICP-USD", "FIL-USD"
    ],
    
    "🇨🇦 TSX TOP 20": [
        "RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNR.TO", "BMO.TO", "CNQ.TO", "CP.TO",
        "SU.TO", "BCE.TO", "CM.TO", "TRP.TO", "ABX.TO", "MFC.TO", "SLF.TO", "FNV.TO",
        "NTR.TO", "WCN.TO", "BAM.TO", "QSR.TO"
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
        
        wave_count = len(pivots) // 2
        
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

# ==================== 80% SETUP ====================
def check_80_percent_setup(df, analysis):
    """Setup 80% Win Rate"""
    try:
        if len(df) < 50:
            return {'valid': False, 'score': 0, 'reasons': []}
        
        reasons = []
        score = 0
        
        if analysis['quantum_score'] >= 260:
            score += 20
            reasons.append("✅ Quantum >= 260")
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
            reasons.append(f"❌ RSI {analysis['rsi']:.1f}")
        
        ema9 = df['Close'].ewm(span=9).mean().iloc[-1]
        ema21 = df['Close'].ewm(span=21).mean().iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        
        if analysis['current_price'] > ema9 > ema21 > ema50:
            score += 15
            reasons.append("✅ EMA alignment")
        else:
            reasons.append("❌ EMA alignment failed")
        
        if analysis['order_flow']['vol_ratio'] > 2.0:
            score += 10
            reasons.append(f"✅ Volume {analysis['order_flow']['vol_ratio']:.1f}x")
        else:
            reasons.append(f"❌ Volume low")
        
        if analysis['smc']['bias'] == 'BUY':
            score += 10
            reasons.append("✅ Bullish structure")
        else:
            reasons.append("❌ Neutral/Bearish")
        
        score += 10
        
        valid = score >= 80
        
        return {
            'valid': valid,
            'score': score,
            'reasons': reasons,
            'confidence': 'ULTRA HIGH' if score >= 90 else 'HIGH'
        }
    except:
        return {'valid': False, 'score': 0, 'reasons': ['Error']}
# TAB 5: SMART MONEY
with tab5:
    st.header("🧠 SMART MONEY CONCEPTS")
    
    symbol_smc = st.text_input("Enter Symbol for SMC Analysis", "AAPL", key="input_smc_symbol").upper()
    
    if st.button("🔍 ANALYZE SMC", type="primary", use_container_width=True, key="btn_analyze_smc"):
        df = get_live_data(symbol_smc, "6mo", "1d")
        
        if df is not None and len(df) > 50:
            analysis = calculate_quantum_ai_score(df, symbol_smc)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Quantum Score", f"{analysis['quantum_score']:.0f}/300")
            col2.metric("AI Score", f"{analysis['ai_score']:.0f}/100")
            col3.metric(analysis['tier'], analysis['recommendation'])
            col4.metric("RSI", f"{analysis['rsi']:.1f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Order Flow")
                st.metric("Imbalance", analysis['order_flow']['imbalance'])
                st.metric("Flow Score", f"{analysis['order_flow']['flow_score']:.1f}/100")
                st.metric("Volume Ratio", f"{analysis['order_flow']['vol_ratio']:.2f}x")
                
                if analysis['order_flow']['whale_detected']:
                    st.markdown('<div class="whale-alert">🐋 WHALE ACTIVITY DETECTED!</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("📈 Volume Profile")
                st.metric("POC", f"${analysis['volume_profile']['poc']:.2f}")
                st.metric("VAH", f"${analysis['volume_profile']['vah']:.2f}")
                st.metric("VAL", f"${analysis['volume_profile']['val']:.2f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Market Structure")
                st.metric("Structure", analysis['smc']['structure'])
                st.metric("Bias", analysis['smc']['bias'])
                
                if len(analysis['smc']['order_blocks']) > 0:
                    st.write("**Order Blocks:**")
                    for ob in analysis['smc']['order_blocks'][:3]:
                        st.write(f"• {ob['type']}: ${ob['bottom']:.2f} - ${ob['top']:.2f}")
            
            with col2:
                st.subheader("🌊 Elliott Wave")
                st.metric("Wave Count", analysis['elliott']['wave_count'])
                st.metric("Direction", analysis['elliott']['direction'])
                st.metric("Confidence", f"{analysis['elliott']['confidence']:.0f}%")
            
            st.markdown("---")
            
            if len(analysis['sfp_signals']) > 0:
                st.subheader("💎 SFP Signals (Recent)")
                for sfp in analysis['sfp_signals'][-3:]:
                    st.write(f"• {sfp['type']} @ ${sfp['price']:.2f} | Level: ${sfp['level']:.2f} | Score: {sfp['score']}/100")
            
            st.markdown("---")
            
            if analysis['setup_80']['valid']:
                st.markdown(f"""
                <div class="setup-80">
                    ⭐ <b>80% WIN RATE SETUP DETECTED!</b><br>
                    Score: {analysis['setup_80']['score']}/100<br>
                    Confidence: {analysis['setup_80']['confidence']}
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("✅ Setup Validation:")
                for reason in analysis['setup_80']['reasons']:
                    st.write(reason)
        else:
            st.error("❌ Unable to fetch data. Try another symbol.")

# TAB 6: BACKTEST
with tab6:
    st.header("🧪 BACKTEST ENGINE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bt_symbol = st.text_input("Symbol to Backtest", "AAPL", key="input_bt_symbol").upper()
    
    with col2:
        bt_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1, key="select_bt_period")
    
    with col3:
        bt_capital = st.number_input("Initial Capital", min_value=1000, max_value=1000000, value=10000, step=1000, key="input_bt_capital")
    
    if st.button("🚀 RUN BACKTEST", type="primary", use_container_width=True, key="btn_run_backtest"):
        with st.spinner(f"Running backtest on {bt_symbol}..."):
            backtest_results = run_backtest(bt_symbol, bt_period, bt_capital)
        
        if backtest_results:
            st.success("✅ Backtest Complete!")
            
            stats = backtest_results['stats']
            
            st.markdown("---")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", stats['Total Trades'])
            col2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
            col3.metric("Profit Factor", f"{stats['Profit Factor']:.2f}")
            col4.metric("Total P&L", f"${stats['Total P&L']:,.2f}", f"{stats['Total P&L%']:.2f}%")
            col5.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Wins", stats['Wins'])
            col2.metric("Losses", stats['Losses'])
            col3.metric("Avg Win", f"${stats['Avg Win']:.2f}")
            col4.metric("Avg Loss", f"${stats['Avg Loss']:.2f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            col1.metric("Initial Capital", f"${bt_capital:,.2f}")
            col2.metric("Final Balance", f"${stats['Final Balance']:,.2f}", f"{stats['Total P&L%']:.2f}%")
            
            st.metric("Max Drawdown", f"{stats['Max Drawdown']:.2f}%")
            
            st.markdown("---")
            
            st.subheader("📈 Equity Curve")
            
            equity_df = pd.DataFrame({
                'Date': backtest_results['dates'],
                'Equity': backtest_results['equity_curve']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['Date'],
                y=equity_df['Equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#ff8c00', width=2)
            ))
            
            fig.update_layout(
                title=f"{bt_symbol} Backtest - Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("📊 Trade History")
            st.dataframe(backtest_results['trades'], use_container_width=True, hide_index=True)
            
            csv_trades = backtest_results['trades'].to_csv(index=False)
            st.download_button(
                "💾 DOWNLOAD TRADES CSV",
                csv_trades,
                f"{bt_symbol}_backtest_{bt_period}.csv",
                "text/csv",
                use_container_width=True,
                key="btn_download_bt"
            )
        else:
            st.error("❌ Backtest failed. Not enough data or invalid symbol.")

# TAB 7: SIGNAL TRACKER
with tab7:
    st.header("📊 SIGNAL TRACKER")
    
    # ✅ UPDATE STATUS À CHAQUE OUVERTURE DU TAB
    update_signal_status()
    
    # Bouton de refresh manuel
    if st.button("🔄 REFRESH SIGNALS", use_container_width=True, key="btn_refresh_signals"):
        update_signal_status()
        st.success("✅ Signals updated!")
        st.rerun()
    
    st.markdown("---")
    
    tab_active, tab_closed = st.tabs(["🟢 Active Signals", "📝 Closed Trades"])
    
    with tab_active:
        if len(st.session_state.active_signals) > 0:
            st.subheader(f"🟢 Active Signals ({len(st.session_state.active_signals)})")
            
            for idx, signal in enumerate(st.session_state.active_signals):
                current_price = get_current_price(signal['symbol'])
                
                if current_price > 0:
                    pnl = (current_price - signal['entry_price'])
                    pnl_pct = ((current_price / signal['entry_price']) - 1) * 100
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
                    
                    with col1:
                        st.markdown(f"### {signal['tier']}")
                        st.markdown(f"**{signal['symbol']}**")
                    
                    with col2:
                        st.metric("Entry", f"${signal['entry_price']:.2f}")
                        st.metric("Current", f"${current_price:.2f}")
                        st.caption(f"📅 {signal['entry_date']}")
                    
                    with col3:
                        st.metric("Stop", f"${signal['stop']:.2f}")
                        st.metric("TP1", f"${signal['tp1']:.2f}", "✅" if current_price >= signal['tp1'] else "")
                        st.metric("TP2", f"${signal['tp2']:.2f}", "✅" if current_price >= signal['tp2'] else "")
                    
                    with col4:
                        st.metric("TP3", f"${signal['tp3']:.2f}", "✅" if current_price >= signal['tp3'] else "")
                        st.metric("Unrealized P&L", f"${pnl:+.2f}", f"{pnl_pct:+.1f}%")
                        
                        if st.button("❌ Close Manually", key=f"close_{idx}"):
                            signal['exit_price'] = current_price
                            signal['exit_reason'] = 'MANUAL CLOSE'
                            signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                            signal['pnl'] = pnl
                            signal['pnl_pct'] = pnl_pct
                            signal['status'] = 'CLOSED'
                            
                            st.session_state.closed_signals.append(signal)
                            st.session_state.active_signals.pop(idx)
                            st.rerun()
                    
                    # Progress bar
                    distance_to_tp1 = (current_price - signal['entry_price']) / (signal['tp1'] - signal['entry_price'])
                    progress = min(max(distance_to_tp1, 0), 1)
                    st.progress(progress)
                    
                    st.markdown("---")
        else:
            st.info("No active signals. Go to Quantum Scanner to find signals!")
    
    with tab_closed:
        if len(st.session_state.closed_signals) > 0:
            st.subheader(f"📝 Closed Trades ({len(st.session_state.closed_signals)})")
            
            closed_df = pd.DataFrame(st.session_state.closed_signals)
            
            total_pnl = closed_df['pnl'].sum()
            wins = len(closed_df[closed_df['pnl'] > 0])
            losses = len(closed_df[closed_df['pnl'] <= 0])
            win_rate = (wins / len(closed_df)) * 100 if len(closed_df) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(closed_df))
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total P&L", f"${total_pnl:+,.2f}")
            col4.metric("Wins / Losses", f"{wins} / {losses}")
            
            st.markdown("---")
            
            display_df = pd.DataFrame({
                'Symbol': closed_df['symbol'],
                'Tier': closed_df['tier'],
                'Entry': closed_df['entry_price'].apply(lambda x: f"${x:.2f}"),
                'Exit': closed_df['exit_price'].apply(lambda x: f"${x:.2f}"),
                'Entry Date': closed_df['entry_date'],
                'Exit Date': closed_df['exit_date'],
                'Exit Reason': closed_df['exit_reason'],
                'P&L': closed_df['pnl'].apply(lambda x: f"${x:+.2f}"),
                'P&L%': closed_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%"),
                'Result': closed_df['pnl'].apply(lambda x: '🟢 WIN' if x > 0 else '🔴 LOSS')
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv_closed = display_df.to_csv(index=False)
            st.download_button(
                "💾 DOWNLOAD CLOSED TRADES",
                csv_closed,
                "closed_trades.csv",
                "text/csv",
                use_container_width=True,
                key="btn_download_closed"
            )
            
            if st.button("🗑️ CLEAR ALL CLOSED TRADES", type="secondary"):
                st.session_state.closed_signals = []
                st.rerun()
        else:
            st.info("No closed trades yet.")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("🥓 Bacon Trader Pro v4.4 ULTIMATE | Made with 🔥 by We Bacon ")
st.caption(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
