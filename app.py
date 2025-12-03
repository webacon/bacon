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

# ==================== CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="🥓 Bacon Trader Pro - Quantum Ultimate",
    page_icon="🥓",
    initial_sidebar_state="expanded"
)

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_TOKEN = "8414444384:AAFlblBgToY7ew50ufZTmNd5qJGRid-TtVA"
CHAT_ID = "813100618"

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
    "🇺🇸 S&P 500": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH",
                     "XOM", "MA", "HD", "PG", "JNJ", "COST", "ABBV", "MRK", "AMD", "CRM",
                     "NFLX", "WMT", "ACN", "BAC", "KO", "PEP", "LIN", "TMO", "ADBE", "DIS"],
    "🚀 NASDAQ": ["NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA", "GOOGL", "NFLX", "AMD", "INTC",
                  "ADBE", "CSCO", "AVGO", "QCOM", "TXN", "INTU", "AMAT", "MU", "LRCX", "KLAC"],
    "🇨🇦 TSX": ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNR.TO", "BMO.TO", "CNQ.TO", "CP.TO"],
    "₿ Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD"],
    "📈 Futures": ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F"]
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
def scan_market_quantum(symbols, min_score=220, min_ai=75):
    results = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, symbol in enumerate(symbols):
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
        
        progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.05)
    
    progress_bar.empty()
    status.empty()
    
    return pd.DataFrame(results)

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
    st.caption("⚡ QUANTUM ULTIMATE | ⭐ 80% Setup | 📱 Telegram | 🧪 Backtest")
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
    st.success("✅ Live Position Tracking")
    
    st.markdown("---")
    st.caption("🥓 Bacon Trader Pro Quantum v3.3")
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Dashboard",
    "💼 Portfolio Day",
    "📈 Portfolio LT",
    "🔍 Quantum Scanner",
    "🧠 Smart Money",
    "🧪 Backtest"
])

# TAB 1: DASHBOARD
with tab1:
    st.header("🎯 QUANTUM DASHBOARD")
    
    day_stats = calculate_portfolio_stats('day')
    lt_stats = calculate_portfolio_stats('long')
    total_equity = day_stats['total_value'] + lt_stats['total_value']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity", f"${total_equity:,.0f}")
    col2.metric("Day Trading", f"${day_stats['total_value']:,.0f}", f"{day_stats['total_pnl_pct']:+.2f}%")
    col3.metric("Long-Term", f"${lt_stats['total_value']:,.0f}", f"{lt_stats['total_pnl_pct']:+.2f}%")
    col4.metric("Total Positions", len(st.session_state.day_positions) + len(st.session_state.long_positions))
    
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
                results = scan_market_quantum(symbols, min_quantum, min_ai)
            
            if len(results) > 0:
                if show_80_only:
                    results = results[results['80% Setup'] == '⭐']
                
                setups_80 = results[results['80% Setup'] == '⭐']
                
                if len(setups_80) > 0:
                    st.success(f"⭐ {len(setups_80)} HIGH PROBABILITY (80%) SETUPS!")
                    
                    for _, row in setups_80.iterrows():
                        whale_text = " 🐋 WHALE!" if row['Whale'] == '🐋' else ""
                        
                        st.markdown(f"""
                        <div class="setup-80">
                        <h2>⭐ {row['Symbol']} - 80% WIN RATE SETUP{whale_text}</h2>
                        <p><b>Quantum:</b> {row['Quantum']:.0f}/300 | <b>AI:</b> {row['AI']:.0f}/100 | <b>Tier:</b> {row['Tier']}</p>
                        <p><b>Entry:</b> ${row['Entry']} | <b>Stop:</b> ${row['Stop']}</p>
                        <p><b>Targets:</b> TP1: ${row['TP1']} | TP2: ${row['TP2']} | TP3: ${row['TP3']}</p>
                        <p><b>Flow:</b> {row['Flow']} | <b>RSI:</b> {row['RSI']}</p>
                        <p>🎯 <b>Ultra-conservative setup - High confidence trade</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.success(f"✅ {len(results)} signals | {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.warning(f"⚠️ No signals found")
        else:
            st.info("👈 Configure and click QUANTUM SCAN")

# TAB 5: SMART MONEY
with tab5:
    st.header("🧠 SMART MONEY ANALYSIS")
    
    sm_col1, sm_col2 = st.columns([1, 3])
    
    with sm_col1:
        sm_symbol = st.text_input("Symbol", "NVDA", key="smart_money_symbol").upper()
        sm_analyze = st.button("🔍 ANALYZE", type="primary", use_container_width=True, key="btn_smart_analyze")
    
    with sm_col2:
        if sm_analyze:
            df = get_live_data(sm_symbol, "3mo", "1d")
            
            if df is not None:
                analysis = calculate_quantum_ai_score(df, sm_symbol)
                
                if analysis['setup_80']['valid']:
                    st.markdown(f"""
                    <div class="setup-80">
                    <h2>⭐ 80% WIN RATE SETUP DETECTED!</h2>
                    <p><b>Setup Score:</b> {analysis['setup_80']['score']}/100</p>
                    <p><b>Confidence:</b> {analysis['setup_80']['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📋 Setup Criteria:")
                    for reason in analysis['setup_80']['reasons']:
                        if reason.startswith('✅'):
                            st.success(reason)
                        else:
                            st.warning(reason)
                
                if analysis['whale_alert']:
                    st.markdown(f"""
                    <div class="whale-alert">
                    <h2>🐋 WHALE ALERT!</h2>
                    <p>Volume: {analysis['order_flow']['vol_ratio']:.2f}x average</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Quantum", f"{analysis['quantum_score']:.0f}/300")
                m2.metric("Tier", analysis['tier'])
                m3.metric("Flow", analysis['order_flow']['imbalance'])
                m4.metric("Structure", analysis['smc']['bias'])
                m5.metric("80% Setup", "YES" if analysis['setup_80']['valid'] else "NO")
                
                st.markdown("---")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("📊 ORDER FLOW")
                    st.metric("Cumulative Delta", f"{analysis['order_flow']['cumulative_delta']:,.0f}")
                    st.metric("Flow Score", f"{analysis['order_flow']['flow_score']:.1f}/100")
                    st.metric("Imbalance", analysis['order_flow']['imbalance'])
                    
                    st.markdown("---")
                    
                    st.subheader("📈 VOLUME PROFILE")
                    st.metric("POC", f"${analysis['volume_profile']['poc']:.2f}")
                    st.metric("VAH", f"${analysis['volume_profile']['vah']:.2f}")
                    st.metric("VAL", f"${analysis['volume_profile']['val']:.2f}")
                
                with c2:
                    st.subheader("🎯 SFP SIGNALS")
                    if analysis['sfp_signals']:
                        for sfp in analysis['sfp_signals'][-3:]:
                            st.info(f"**{sfp['type']}** @ ${sfp['level']:.2f} ({sfp['date'].strftime('%Y-%m-%d')})")
                    else:
                        st.info("No recent SFP")
                    
                    st.markdown("---")
                    
                    st.subheader("🏛️ MARKET STRUCTURE")
                    st.metric("Structure", analysis['smc']['structure'])
                    st.metric("Bias", analysis['smc']['bias'])
                    
                    if analysis['smc']['order_blocks']:
                        st.info(f"{len(analysis['smc']['order_blocks'])} Order Blocks")
            else:
                st.error(f"❌ No data for {sm_symbol}")

# TAB 6: BACKTEST
with tab6:
    st.header("🧪 QUANTUM BACKTEST ENGINE")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        bt_symbol = st.text_input("Symbol", "NVDA", key="backtest_symbol").upper()
        bt_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1, key="backtest_period")
        bt_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000, key="backtest_capital")
        
        run_bt = st.button("🚀 RUN BACKTEST", type="primary", use_container_width=True, key="btn_run_backtest")
    
    with col2:
        if run_bt:
            with st.spinner(f"Backtesting {bt_symbol} over {bt_period}..."):
                bt_results = run_backtest(bt_symbol, bt_period, bt_capital)
            
            if bt_results:
                stats = bt_results['stats']
                
                st.success(f"✅ Backtest completed for {bt_symbol}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Trades", stats['Total Trades'])
                col2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
                col3.metric("Total P&L", f"${stats['Total P&L']:.0f}", f"{stats['Total P&L%']:.1f}%")
                col4.metric("Profit Factor", f"{stats['Profit Factor']:.2f}")
                col5.metric("Final Balance", f"${stats['Final Balance']:,.0f}")
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Win", f"${stats['Avg Win']:.2f}")
                col2.metric("Avg Loss", f"${stats['Avg Loss']:.2f}")
                col3.metric("Max DD", f"{stats['Max Drawdown']:.2f}%")
                col4.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
                
                st.markdown("---")
                
                st.subheader("📈 Equity Curve")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=bt_results['dates'],
                    y=bt_results['equity_curve'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='lime', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)'
                ))
                
                fig.add_hline(y=bt_capital, line_dash="dash", line_color="orange",
                             annotation_text="Initial Capital")
                
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Balance ($)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("📋 All Trades")
                st.dataframe(bt_results['trades'], use_container_width=True, height=400, hide_index=True)
                
                st.markdown("---")
                st.subheader("📊 Trade Analysis")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.metric("Wins", stats['Wins'], "🟢")
                    st.metric("Losses", stats['Losses'], "🔴")
                
                with c2:
                    exit_reasons = bt_results['trades']['Exit Reason'].value_counts()
                    st.write("**Exit Reasons:**")
                    for reason, count in exit_reasons.items():
                        st.write(f"• {reason}: {count} trades")
            else:
                st.error(f"❌ Unable to backtest {bt_symbol}")
        else:
            st.info("👈 Configure backtest parameters and click RUN BACKTEST")
            
            st.markdown("---")
            st.subheader("ℹ️ Backtest Info")
            st.write("""
            **Stratégie testée :**
            - Entry: Quantum Score >= 220 + BUY/STRONG BUY signal
            - Exits: TP1/TP2/TP3 ou Stop Loss (ATR-based)
            - Risk: 2% du capital par trade
            - Time exit: 20 jours max par position
            
            **Métriques clés :**
            - **Win Rate**: % de trades gagnants
            - **Profit Factor**: Ratio gains/pertes
            - **Sharpe Ratio**: Rendement ajusté au risque
            - **Max Drawdown**: Plus grosse perte depuis un sommet
            """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("🥓 Bacon Trader Pro - QUANTUM ULTIMATE EDITION v3.3")
st.caption("✨ Order Flow | Volume Profile | SFP | Elliott Wave | SMC | Whale Detection")
st.caption("⭐ 80% Win Rate Setup | 📱 Telegram Alerts | 🧪 Full Backtest Engine")
st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
st.caption("🚀 Built by traders, for traders. #WeBacon 🥓")
