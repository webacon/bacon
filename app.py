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
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "8144444384:AAFlblBgToY7ew50ufZTmNd5qJGRid-TtVA")
CHAT_ID = st.secrets.get("CHAT_ID", "813100618")

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
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
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# ==================== CACHE ====================
@st.cache_data(ttl=300)
def get_live_data(symbol, period="1mo", interval="1d"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if len(hist) > 0:
            return hist
        return None
    except:
        return None

# ==================== MARKETS ====================
MARKETS = {
    "🔥 TOP 10 US": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V"],
    "⭐ TOP 50 US": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V",
                     "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST", "PG",
                     "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO", "ADBE",
                     "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS", "CMCSA",
                     "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON", "UNP"],
    "🇺🇸 DOW 30": ["AAPL", "MSFT", "NVDA", "AMZN", "WMT", "JPM", "V", "JNJ", "HD", "PG",
                    "UNH", "CSCO", "CVX", "KO", "IBM", "CAT", "GS", "MRK", "AXP", "CRM",
                    "MCD", "DIS", "AMGN", "VZ", "BA", "HON", "NKE", "MMM", "SHW", "TRV"],
    "💼 S&P 100": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "LLY",
                   "V", "AVGO", "WMT", "JPM", "MA", "XOM", "UNH", "ORCL", "HD", "COST",
                   "PG", "JNJ", "NFLX", "BAC", "CRM", "AMD", "ABBV", "CVX", "MRK", "KO",
                   "ADBE", "PEP", "TMO", "ACN", "CSCO", "LIN", "MCD", "ABT", "INTC", "DIS",
                   "CMCSA", "WFC", "DHR", "VZ", "TXN", "PM", "QCOM", "NEE", "IBM", "HON",
                   "UNP", "INTU", "CAT", "GS", "LOW", "SPGI", "AXP", "BLK", "BA", "SBUX",
                   "RTX", "BKNG", "AMAT", "GILD", "ELV", "SYK", "PLD", "ADP", "MDLZ", "VRTX",
                   "ADI", "CI", "TJX", "REGN", "MMC", "LRCX", "CB", "SO", "PGR", "SCHW",
                   "C", "DE", "BMY", "AMT", "NOW", "ISRG", "PANW", "BX", "FI", "ETN",
                   "MU", "ZTS", "BSX", "SLB", "USB", "DUK", "MCK", "PH", "EOG", "KLAC"],
    "₿ CRYPTO TOP 30": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
                        "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD",
                        "UNI-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "VET-USD",
                        "ICP-USD", "FIL-USD", "HBAR-USD", "APT-USD", "OP-USD", "ARB-USD",
                        "NEAR-USD", "STX-USD", "IMX-USD", "INJ-USD", "TIA-USD", "RUNE-USD"],
    "🇨🇦 TSX TOP 30": ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNR.TO", "BMO.TO", "CNQ.TO", "CP.TO",
                       "SU.TO", "BCE.TO", "CM.TO", "TRP.TO", "ABX.TO", "MFC.TO", "SLF.TO", "FNV.TO",
                       "NTR.TO", "WCN.TO", "BAM.TO", "QSR.TO", "SHOP.TO", "TRI.TO", "WN.TO", "ATD.TO",
                       "L.TO", "NA.TO", "MG.TO", "CCL-B.TO", "AEM.TO", "DOL.TO"],
    "🏦 FINANCE US": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
                      "PNC", "TFC", "BK", "COF", "CME", "ICE", "SPGI", "MCO", "AIG", "MET"],
    "⚡ TECH US": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE", "CRM",
                   "CSCO", "ACN", "INTC", "AMD", "NOW", "QCOM", "AMAT", "ADI", "LRCX", "INTU"],
    "🏥 SANTÉ US": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
                    "AMGN", "GILD", "VRTX", "ELV", "CI", "CVS", "MCK", "BSX", "ISRG", "SYK"],
    "⚙️ INDUSTRIELS US": ["CAT", "HON", "BA", "RTX", "UNP", "DE", "GE", "MMM", "LMT", "NOC",
                          "EMR", "ETN", "FDX", "UPS", "WM", "NSC", "ITW", "CSX", "PH", "CMI"],
    "🛒 CONSOMMATION US": ["AMZN", "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TJX", "LOW", "TGT",
                           "DIS", "BKNG", "ORLY", "AZO", "MAR", "YUM", "CMG", "ROST", "DG", "DLTR"],
    "⚡ ÉNERGIE US": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY",
                      "WMB", "KMI", "HAL", "BKR", "FANG", "DVN", "HES", "APA", "MRO", "CTRA"]
}

# ==================== INDICATORS ====================
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

def calculate_order_flow_advanced(df):
    try:
        if len(df) < 20:
            return {'delta': 0, 'cumulative_delta': 0, 'imbalance': 'NEUTRAL', 'whale_detected': False, 'flow_score': 50, 'vol_ratio': 1}
        df = df.copy()
        df['Price_Change'] = df['Close'] - df['Open']
        df['Direction'] = np.where(df['Price_Change'] > 0, 1, np.where(df['Price_Change'] < 0, -1, 0))
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
        return {'delta': float(df['Delta'].iloc[-1]), 'cumulative_delta': float(df['Cumulative_Delta'].iloc[-1]),
                'imbalance': imbalance, 'whale_detected': whale_detected, 'flow_score': float(flow_score), 'vol_ratio': float(vol_ratio)}
    except:
        return {'delta': 0, 'cumulative_delta': 0, 'imbalance': 'NEUTRAL', 'whale_detected': False, 'flow_score': 50, 'vol_ratio': 1}

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
            volume_at_price.append({'price': (price_low + price_high) / 2, 'volume': vol})
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
        return {'poc': poc_price, 'vah': vah, 'val': val, 'profile': vp_df}
    except:
        return {'poc': 0, 'vah': 0, 'val': 0, 'profile': None}

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
            if (current['Low'] < prev_low and current['Close'] > prev_low and next_bar['Close'] > current['Close']):
                sfp_signals.append({'date': df.index[i], 'type': 'BULLISH_SFP', 'price': float(current['Close']), 'level': float(prev_low), 'score': 85})
            if (current['High'] > prev_high and current['Close'] < prev_high and next_bar['Close'] < current['Close']):
                sfp_signals.append({'date': df.index[i], 'type': 'BEARISH_SFP', 'price': float(current['Close']), 'level': float(prev_high), 'score': 85})
        return sfp_signals[-5:] if sfp_signals else []
    except:
        return []

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
        wave_count = len([i for i in range(len(pivots)-1) if pivots[i]['type'] != pivots[i+1]['type']]) // 2
        return {'wave_count': wave_count, 'direction': 'IN_PROGRESS', 'confidence': 40}
    except:
        return {'wave_count': 0, 'direction': 'NEUTRAL', 'confidence': 0}

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
                order_blocks.append({'type': 'BULLISH_OB', 'top': float(candle['High']), 'bottom': float(candle['Low']), 'date': df.index[i]})
        return {'structure': structure, 'bias': bias, 'order_blocks': order_blocks[-3:], 'last_high': float(last_high), 'last_low': float(last_low)}
    except:
        return {'structure': 'NEUTRAL', 'bias': 'NEUTRAL', 'order_blocks': []}

def check_80_percent_setup(df, analysis):
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
        return {'valid': valid, 'score': score, 'reasons': reasons, 'confidence': 'ULTRA HIGH' if score >= 90 else 'HIGH'}
    except:
        return {'valid': False, 'score': 0, 'reasons': ['Error in calculation']}

def calculate_quantum_ai_score(df, symbol=""):
    if df is None or len(df) < 50:
        return {'quantum_score': 0, 'ai_score': 0, 'recommendation': 'WAIT', 'signal': '⚫', 'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False,
                'atr': 0, 'rsi': 50, 'current_price': 0, 'setup_80': {'valid': False, 'score': 0}, 'order_flow': {}, 'volume_profile': {}, 'sfp_signals': [], 'elliott': {}, 'smc': {}}
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
        quantum_ai = (trend_score * 0.15 + rsi_score * 0.10 + macd_score * 0.10 + volume_score * 0.15 + flow_score * 0.15 + vp_score * 0.10 + sfp_score * 0.10 + elliott_score * 0.05 + smc_score * 0.10)
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
        result = {'quantum_score': round(quantum_score, 1), 'ai_score': round(quantum_ai, 1), 'recommendation': recommendation, 'signal': signal, 'confidence': confidence,
                  'tier': tier, 'whale_alert': order_flow['whale_detected'], 'atr': atr, 'rsi': rsi, 'current_price': current_price,
                  'order_flow': order_flow, 'volume_profile': volume_profile, 'sfp_signals': sfp_signals, 'elliott': elliott, 'smc': smc}
        result['setup_80'] = check_80_percent_setup(df, result)
        return result
    except:
        return {'quantum_score': 0, 'ai_score': 0, 'recommendation': 'ERROR', 'signal': '⚫', 'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False,
                'atr': 0, 'rsi': 50, 'current_price': 0, 'setup_80': {'valid': False, 'score': 0}, 'order_flow': {}, 'volume_profile': {}, 'sfp_signals': [], 'elliott': {}, 'smc': {}}

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

def scan_market_quantum(symbols, min_score=180, min_ai=60, show_progress=True):
    results = []
    scanned = 0
    signals_found = 0
    errors = 0
    if show_progress:
        progress_bar = st.progress(0)
        status = st.empty()
        stats = st.empty()
    for i, symbol in enumerate(symbols):
        if show_progress:
            status.text(f"🔍 Quantum Scanning {symbol}... ({i+1}/{len(symbols)})")
            stats.info(f"📊 Scanned: {scanned} | Signals: {signals_found} | Errors: {errors}")
        try:
            df = get_live_data(symbol, period="3mo", interval="1d")
            scanned += 1
            if df is not None and len(df) > 50:
                analysis = calculate_quantum_ai_score(df, symbol)
                if analysis['quantum_score'] >= min_score and analysis['ai_score'] >= min_ai:
                    signals_found += 1
                    stop, tp1, tp2, tp3 = calculate_targets(analysis['current_price'], analysis['atr'], analysis['ai_score'])
                    results.append({'Time': datetime.now().strftime("%H:%M"), 'Symbol': symbol, 'Quantum': analysis['quantum_score'], 'AI': analysis['ai_score'],
                                    'Tier': analysis['tier'], 'Signal': analysis['signal'], 'Entry': round(analysis['current_price'], 2), 'Stop': stop, 'TP1': tp1, 'TP2': tp2, 'TP3': tp3,
                                    'RSI': round(analysis['rsi'], 1), 'Flow': analysis['order_flow']['imbalance'], 'Whale': '🐋' if analysis['whale_alert'] else '',
                                    '80% Setup': '⭐' if analysis['setup_80']['valid'] else '', 'Setup Score': analysis['setup_80']['score'], 'Recommendation': analysis['recommendation']})
        except Exception as e:
            errors += 1
        if show_progress:
            progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.05)
    if show_progress:
        progress_bar.empty()
        status.empty()
        stats.empty()
        st.success(f"✅ Scan Complete! Scanned: {scanned} | Signals Found: {signals_found} | Errors: {errors}")
    return pd.DataFrame(results)

def get_current_price(symbol):
    df = get_live_data(symbol, '5d', '1d')
    if df is not None and len(df) > 0:
        return float(df['Close'].iloc[-1])
    return 0

def update_signal_status():
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

def run_backtest(symbol, period="6mo", initial_capital=10000):
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
                        position = {'entry': current_price, 'stop': stop, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'entry_date': date, 'qty': qty, 'ai_score': analysis['ai_score'], 'tier': analysis['tier']}
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
                trades.append({'Entry Date': position['entry_date'].strftime('%Y-%m-%d'), 'Exit Date': date.strftime('%Y-%m-%d'), 'Entry': round(position['entry'], 2),
                               'Exit': round(exit_price, 2), 'Qty': position['qty'], 'Exit Reason': exit_reason, 'Tier': position['tier'], 'AI Score': position['ai_score'],
                               'P&L': round(pnl, 2), 'P&L%': round(pnl_pct, 2), 'Result': 'WIN' if pnl > 0 else 'LOSS'})
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
        stats = {'Total Trades': len(trades_df), 'Wins': wins, 'Losses': losses, 'Win Rate': win_rate, 'Profit Factor': profit_factor, 'Total P&L': total_pnl,
                 'Total P&L%': total_pnl_pct, 'Avg Win': avg_win, 'Avg Loss': avg_loss, 'Max Drawdown': max_drawdown, 'Sharpe Ratio': sharpe, 'Final Balance': balance}
        return {'trades': trades_df, 'stats': stats, 'equity_curve': equity_curve, 'dates': df.index[:len(equity_curve)]}
    return None

def add_day_position(symbol, qty, entry):
    st.session_state.day_positions.append({'symbol': symbol, 'qty': qty, 'entry': entry, 'date': datetime.now().strftime('%Y-%m-%d %H:%M')})

def add_long_position(symbol, qty, entry):
    st.session_state.long_positions.append({'symbol': symbol, 'qty': qty, 'entry': entry, 'date': datetime.now().strftime('%Y-%m-%d')})

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
            results.append({'Symbol': pos['symbol'], 'Qty': pos['qty'], 'Entry': f"${pos['entry']:.2f}", 'Current': f"${current_price:.2f}",
                            'Stop': f"${stop:.2f}", 'P&L': f"${pnl:.2f}", 'P&L%': f"{pnl_pct:+.2f}%", 'Status': status, 'Date': pos['date']})
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
    return {'total_value': total_current, 'total_pnl': total_pnl, 'total_pnl_pct': total_pnl_pct}

def run_auto_scan():
    symbols = MARKETS["⭐ TOP 50 US"]
    results = scan_market_quantum(symbols, min_score=240, min_ai=80, show_progress=False)
    if len(results) > 0:
        st.session_state.auto_scan_results = results
        st.session_state.last_auto_scan = datetime.now()

def create_chart_analysis(df, symbol, analysis):
    """Crée un graphique d'analyse technique complet"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.20],
        subplot_titles=(f'{symbol} - Price Action', 'RSI', 'MACD', 'Volume')
    )
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    ema9 = df['Close'].ewm(span=9).mean()
    ema21 = df['Close'].ewm(span=21).mean()
    ema50 = df['Close'].ewm(span=50).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=ema9, name='EMA9', line=dict(color='cyan', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA21', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA50', line=dict(color='magenta', width=1)), row=1, col=1)
    
    # Volume Profile POC
    if analysis['volume_profile']['poc'] > 0:
        fig.add_hline(y=analysis['volume_profile']['poc'], line_dash="dash", line_color="yellow", 
                     annotation_text="POC", row=1, col=1)
    
    # RSI
    rsi_values = [calculate_rsi(df.iloc[:i+1], 14) for i in range(13, len(df))]
    rsi_dates = df.index[13:]
    fig.add_trace(go.Scatter(x=rsi_dates, y=rsi_values, name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color='gray'), row=3, col=1)
    
    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        title_text=f"📊 {symbol} - Quantum Analysis"
    )
    
    return fig

# ==================== THEME ====================
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0a0a0a 0%, #1a1410 100%); color: #ff8c00; font-family: 'Courier New', monospace;}
    div[data-testid="stMetricValue"] {font-size: 28px; color: #ff8c00; font-weight: bold; text-shadow: 0 0 10px #ff8c00;}
    .diamond-signal {background: linear-gradient(135deg, rgba(0,255,255,0.3) 0%, rgba(138,43,226,0.4) 100%); border: 4px solid cyan;
                      padding: 20px; border-radius: 15px; color: white; font-weight: bold; margin: 10px 0; box-shadow: 0 0 20px cyan;}
    .setup-80 {background: linear-gradient(135deg, rgba(255,215,0,0.3) 0%, rgba(255,140,0,0.4) 100%); border: 4px solid gold;
               padding: 20px; border-radius: 15px; color: white; font-weight: bold; margin: 10px 0; box-shadow: 0 0 20px gold;}
    .whale-alert {background: linear-gradient(135deg, rgba(255,0,255,0.3) 0%, rgba(255,100,255,0.4) 100%); border: 3px solid magenta;
                  padding: 15px; border-radius: 10px; color: white; font-weight: bold; margin: 10px 0; animation: pulse 2s infinite;}
    @keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.7;}}
    .metric-card {background: rgba(255,140,0,0.1); border: 2px solid #ff8c00; border-radius: 10px; padding: 15px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 6, 2])
with col1:
    st.markdown("# 🥓")
with col2:
    st.markdown("# BACON TRADER PRO")
    st.caption("⚡ QUANTUM ULTIMATE v7.0 | ⭐ Setup 80% | 📊 Signal Tracker | 🧪 Backtest | 🧠 SMC | 📈 Charts")
with col3:
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now <= market_close and now.weekday() < 5:
        st.metric("Market", "OPEN 🟢")
    else:
        st.metric("Market", "CLOSED 🔴")

st.markdown("---")

# ==================== AUTO-REFRESH ====================
current_time = time.time()
if current_time - st.session_state.last_update > 30:
    update_signal_status()
    st.session_state.last_update = current_time

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ QUANTUM CONTROL")
    st.session_state.telegram_enabled = st.toggle("📱 Telegram Alerts", value=st.session_state.telegram_enabled)
    if st.session_state.telegram_enabled:
        st.success("✅ Telegram ON")
    else:
        st.info("📴 Telegram OFF")
    
    st.markdown("---")
    st.subheader("🎯 SCAN PARAMETERS")
    min_quantum = st.slider("Min Quantum Score", 0, 300, 180, 10, help="Minimum Quantum score pour filtrer les signaux")
    min_ai = st.slider("Min AI Score", 0, 100, 60, 5, help="Minimum AI score pour filtrer les signaux")
    min_setup_score = st.slider("Min Setup 80% Score", 0, 100, 0, 10, help="Minimum score pour le setup 80%")
    only_setup_80 = st.checkbox("⭐ Only 80% Setups", help="Afficher seulement les setups avec score >= 80")
    only_premium = st.checkbox("💎 Only Premium (Diamond/Platinum)", help="Afficher seulement les tiers premium")
    
    st.info(f"📊 Q≥{min_quantum} | AI≥{min_ai} | Setup≥{min_setup_score}")
    
    st.markdown("---")
    st.subheader("🔄 AUTO-SCAN (30min)")
    st.session_state.auto_scan_enabled = st.toggle("Enable Auto-Scan", value=st.session_state.auto_scan_enabled)
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
    st.markdown("**➕ Add Day Position:**")
    day_symbol = st.text_input("Symbol", "", placeholder="AAPL", key="day_sym").upper()
    day_qty = st.number_input("Quantity", min_value=1, value=10, key="day_qty")
    day_entry = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01, key="day_entry")
    if st.button("➕ Add Day Position", use_container_width=True):
        if day_symbol and len(day_symbol) > 0:
            add_day_position(day_symbol, day_qty, day_entry)
            st.success(f"✅ Added {day_qty} {day_symbol} @ ${day_entry}")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("❌ Enter a symbol!")
    
    st.markdown("---")
    lt_stats = calculate_portfolio_stats('long')
    st.subheader("📊 LONG-TERM (CÉLI)")
    st.metric("Value", f"${lt_stats['total_value']:,.2f}")
    st.metric("P&L", f"${lt_stats['total_pnl']:+,.2f}", f"{lt_stats['total_pnl_pct']:+.2f}%")
    st.markdown("**➕ Add Long Position:**")
    lt_symbol = st.text_input("Symbol ", "", placeholder="MSFT", key="lt_sym").upper()
    lt_qty = st.number_input("Quantity ", min_value=1, value=50, key="lt_qty")
    lt_entry = st.number_input("Entry Price ", min_value=0.01, value=150.00, step=0.01, key="lt_entry")
    if st.button("➕ Add Long Position", use_container_width=True):
        if lt_symbol and len(lt_symbol) > 0:
            add_long_position(lt_symbol, lt_qty, lt_entry)
            st.success(f"✅ Added {lt_qty} {lt_symbol} @ ${lt_entry}")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("❌ Enter a symbol!")
    
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
    st.caption("🥓 Bacon Trader Pro v7.0")
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

if st.session_state.auto_scan_enabled and st.session_state.last_auto_scan:
    time_since_scan = (datetime.now() - st.session_state.last_auto_scan).total_seconds()
    if time_since_scan >= 1800:
        run_auto_scan()

# ==================== TABS ====================
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
    
    if len(st.session_state.active_signals) > 0:
        st.subheader("📊 Active Signals Summary")
        for idx, signal in enumerate(st.session_state.active_signals):
            current = get_current_price(signal['symbol'])
            if current > 0:
                pnl = (current - signal['entry_price'])
                pnl_pct = ((current / signal['entry_price']) - 1) * 100
                
                col1, col2, col3, col4 = st.columns([2, 3, 3, 2])
                
                with col1:
                    st.markdown(f"### {signal['tier']} {signal['symbol']}")
                    st.caption(f"Entry: {signal['entry_date']}")
                
                with col2:
                    st.metric("Entry", f"${signal['entry_price']:.2f}")
                    st.metric("Current", f"${current:.2f}")
                
                with col3:
                    st.metric("TP1", f"${signal['tp1']:.2f}", "✅" if current >= signal['tp1'] else "")
                    st.metric("Stop", f"${signal['stop']:.2f}")
                
                with col4:
                    st.metric("Unrealized P&L", f"${pnl:+.2f}", f"{pnl_pct:+.1f}%")
                    progress = min(max((current - signal['entry_price']) / (signal['tp3'] - signal['entry_price']), 0), 1)
                    st.progress(progress)
                
                st.markdown("---")
    else:
        st.info("📊 No active signals. Start scanning to find opportunities!")

# TAB 2: PORTFOLIO DAY
with tab2:
    st.header("💼 DAY TRADING PORTFOLIO")
    day_portfolio = get_portfolio_with_live_prices('day')
    if len(day_portfolio) > 0:
        st.dataframe(day_portfolio, use_container_width=True, hide_index=True)
        st.subheader("🗑️ Remove Position")
        for idx, pos in enumerate(st.session_state.day_positions):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{pos['symbol']} | Qty: {pos['qty']} | Entry: ${pos['entry']:.2f}")
            with col2:
                if st.button("Remove", key=f"remove_day_{idx}"):
                    remove_position('day', idx)
                    st.rerun()
    else:
        st.info("No day trading positions. Add one in the sidebar!")

# TAB 3: PORTFOLIO LT
with tab3:
    st.header("📈 LONG-TERM PORTFOLIO (CÉLI)")
    lt_portfolio = get_portfolio_with_live_prices('long')
    if len(lt_portfolio) > 0:
        st.dataframe(lt_portfolio, use_container_width=True, hide_index=True)
        st.subheader("🗑️ Remove Position")
        for idx, pos in enumerate(st.session_state.long_positions):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{pos['symbol']} | Qty: {pos['qty']} | Entry: ${pos['entry']:.2f}")
            with col2:
                if st.button("Remove", key=f"remove_long_{idx}"):
                    remove_position('long', idx)
                    st.rerun()
    else:
        st.info("No long-term positions. Add one in the sidebar!")

# TAB 4: QUANTUM SCANNER
with tab4:
    st.header("🔍 QUANTUM SCANNER")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("⚙️ Configuration")
        market = st.selectbox("Select Market", list(MARKETS.keys()))
    with col2:
        st.metric("Symbols", len(MARKETS[market]))
        st.metric("Min Q", min_quantum)
        st.metric("Min AI", min_ai)
    
    if st.button("🔥 QUANTUM SCAN", type="primary", use_container_width=True):
        st.markdown("---")
        st.subheader(f"💎 Results: {market}")
        with st.spinner(f"🔍 Scanning {len(MARKETS[market])} symbols..."):
            symbols = MARKETS[market]
            results = scan_market_quantum(symbols, min_quantum, min_ai, show_progress=True)
        
        if len(results) > 0:
            # APPLY FILTERS
            filtered_results = results.copy()
            
            if only_setup_80:
                filtered_results = filtered_results[filtered_results['Setup Score'] >= 80]
            elif min_setup_score > 0:
                filtered_results = filtered_results[filtered_results['Setup Score'] >= min_setup_score]
            
            if only_premium:
                filtered_results = filtered_results[filtered_results['Tier'].isin(['💎 DIAMOND', '🥇 PLATINUM'])]
            
            st.session_state.scan_results = filtered_results
            st.success(f"✅ Found {len(filtered_results)} signals! ({len(results)} total before filters)")
            st.rerun()
        else:
            st.session_state.scan_results = None
            st.warning("📊 No signals found!")
    
    if st.session_state.scan_results is not None and len(st.session_state.scan_results) > 0:
        results = st.session_state.scan_results
        
        st.success(f"✅ {len(results)} signals found!")
        
        # STATISTICS
        setup_80_count = len(results[results['Setup Score'] >= 80])
        diamond_count = len(results[results['Tier'] == '💎 DIAMOND'])
        platinum_count = len(results[results['Tier'] == '🥇 PLATINUM'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Signals", len(results))
        col2.metric("⭐ 80% Setups", setup_80_count)
        col3.metric("💎 Diamond", diamond_count)
        col4.metric("🥇 Platinum", platinum_count)
        col5.metric("Avg Quantum", f"{results['Quantum'].mean():.0f}")
        
        st.markdown("---")
        
        # QUICK TRACK BUTTONS
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 TRACK ALL", use_container_width=True, type="secondary", key="track_all_btn"):
                tracked = 0
                for _, row in results.iterrows():
                    new_signal = {
                        'symbol': str(row['Symbol']),
                        'entry_price': float(row['Entry']),
                        'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'stop': float(row['Stop']),
                        'tp1': float(row['TP1']),
                        'tp2': float(row['TP2']),
                        'tp3': float(row['TP3']),
                        'quantum_score': float(row['Quantum']),
                        'ai_score': float(row['AI']),
                        'tier': str(row['Tier']),
                        'flow': str(row['Flow']),
                        'rsi': float(row['RSI']),
                        'status': 'ACTIVE',
                        'exit_price': None,
                        'exit_reason': None,
                        'exit_date': None,
                        'pnl': 0,
                        'pnl_pct': 0
                    }
                    exists = any(s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' for s in st.session_state.active_signals)
                    if not exists:
                        st.session_state.active_signals.append(new_signal)
                        tracked += 1
                
                if tracked > 0:
                    st.success(f"✅ Tracked {tracked} signals!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("All already tracked!")
        
        with col2:
            if st.button("💎 TRACK PREMIUM", use_container_width=True, type="secondary", key="track_premium_btn"):
                tracked = 0
                premium = results[results['Tier'].isin(['💎 DIAMOND', '🥇 PLATINUM'])]
                for _, row in premium.iterrows():
                    new_signal = {
                        'symbol': str(row['Symbol']),
                        'entry_price': float(row['Entry']),
                        'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'stop': float(row['Stop']),
                        'tp1': float(row['TP1']),
                        'tp2': float(row['TP2']),
                        'tp3': float(row['TP3']),
                        'quantum_score': float(row['Quantum']),
                        'ai_score': float(row['AI']),
                        'tier': str(row['Tier']),
                        'flow': str(row['Flow']),
                        'rsi': float(row['RSI']),
                        'status': 'ACTIVE',
                        'exit_price': None,
                        'exit_reason': None,
                        'exit_date': None,
                        'pnl': 0,
                        'pnl_pct': 0
                    }
                    exists = any(s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' for s in st.session_state.active_signals)
                    if not exists:
                        st.session_state.active_signals.append(new_signal)
                        tracked += 1
                        if st.session_state.telegram_enabled:
                            msg = f"🥓 {row['Tier']} {row['Symbol']}\nQ:{row['Quantum']:.0f} AI:{row['AI']:.0f}\nEntry:${row['Entry']:.2f}"
                            send_telegram_alert(msg)
                
                if tracked > 0:
                    st.success(f"✅ Tracked {tracked} premium!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("All already tracked!")
        
        with col3:
            if st.button("⭐ TRACK 80% SETUPS", use_container_width=True, type="secondary", key="track_80_btn"):
                tracked = 0
                setup_80 = results[results['Setup Score'] >= 80]
                for _, row in setup_80.iterrows():
                    new_signal = {
                        'symbol': str(row['Symbol']),
                        'entry_price': float(row['Entry']),
                        'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'stop': float(row['Stop']),
                        'tp1': float(row['TP1']),
                        'tp2': float(row['TP2']),
                        'tp3': float(row['TP3']),
                        'quantum_score': float(row['Quantum']),
                        'ai_score': float(row['AI']),
                        'tier': str(row['Tier']),
                        'flow': str(row['Flow']),
                        'rsi': float(row['RSI']),
                        'status': 'ACTIVE',
                        'exit_price': None,
                        'exit_reason': None,
                        'exit_date': None,
                        'pnl': 0,
                        'pnl_pct': 0
                    }
                    exists = any(s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' for s in st.session_state.active_signals)
                    if not exists:
                        st.session_state.active_signals.append(new_signal)
                        tracked += 1
                
                if tracked > 0:
                    st.success(f"✅ Tracked {tracked} 80% setups!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("All already tracked!")
        
        with col4:
            csv = results.to_csv(index=False)
            st.download_button("💾 CSV", csv, "signals.csv", "text/csv", use_container_width=True)
        
        st.markdown("---")
        
        # FORM POUR SÉLECTION INDIVIDUELLE
        with st.form("select_signals_form", clear_on_submit=True):
            st.subheader("🎯 Select Individual Signals")
            
            selected_symbols = []
            
            for idx, row in results.iterrows():
                col1, col2, col3, col4, col5 = st.columns([1, 2, 3, 2, 1])
                
                with col1:
                    is_checked = st.checkbox(f"{row['Tier'][:2]}", key=f"check_{idx}_{row['Symbol']}")
                    if is_checked:
                        selected_symbols.append(row['Symbol'])
                
                with col2:
                    st.write(f"**{row['Symbol']}**")
                    if row['80% Setup'] == '⭐':
                        st.caption("⭐ 80% Setup")
                
                with col3:
                    st.write(f"Q:{row['Quantum']:.0f} | AI:{row['AI']:.0f} | Setup:{row['Setup Score']}")
                
                with col4:
                    st.write(f"RSI:{row['RSI']:.1f} | {row['Flow']}")
                
                with col5:
                    st.write(f"${row['Entry']:.2f}")
            
            submitted = st.form_submit_button("📊 TRACK SELECTED", type="primary", use_container_width=True)
            
            if submitted and len(selected_symbols) > 0:
                tracked = 0
                for _, row in results.iterrows():
                    if row['Symbol'] in selected_symbols:
                        new_signal = {
                            'symbol': str(row['Symbol']),
                            'entry_price': float(row['Entry']),
                            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'stop': float(row['Stop']),
                            'tp1': float(row['TP1']),
                            'tp2': float(row['TP2']),
                            'tp3': float(row['TP3']),
                            'quantum_score': float(row['Quantum']),
                            'ai_score': float(row['AI']),
                            'tier': str(row['Tier']),
                            'flow': str(row['Flow']),
                            'rsi': float(row['RSI']),
                            'status': 'ACTIVE',
                            'exit_price': None,
                            'exit_reason': None,
                            'exit_date': None,
                            'pnl': 0,
                            'pnl_pct': 0
                        }
                        exists = any(s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' for s in st.session_state.active_signals)
                        if not exists:
                            st.session_state.active_signals.append(new_signal)
                            tracked += 1
                
                if tracked > 0:
                    st.success(f"✅ Tracked {tracked} selected signals!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        st.dataframe(results, use_container_width=True, hide_index=True)

# TAB 5: SMART MONEY
with tab5:
    st.header("🧠 SMART MONEY CONCEPTS")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol_smc = st.text_input("Enter Symbol for SMC Analysis", "AAPL").upper()
    with col2:
        period_smc = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    if st.button("🔍 ANALYZE SMC", type="primary", use_container_width=True):
        df = get_live_data(symbol_smc, period=period_smc, interval="1d")
        
        if df is not None and len(df) > 50:
            analysis = calculate_quantum_ai_score(df, symbol_smc)
            
            st.markdown("---")
            
            # METRICS
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Quantum Score", f"{analysis['quantum_score']:.0f}/300")
            col2.metric("AI Score", f"{analysis['ai_score']:.0f}/100")
            col3.metric(analysis['tier'], analysis['recommendation'])
            col4.metric("RSI", f"{analysis['rsi']:.1f}")
            col5.metric("Current Price", f"${analysis['current_price']:.2f}")
            
            st.markdown("---")
            
            # 80% SETUP
            if analysis['setup_80']['valid']:
                st.markdown('<div class="setup-80">⭐ 80% WIN RATE SETUP DETECTED!</div>', unsafe_allow_html=True)
                st.subheader("📋 Setup Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Setup Score", f"{analysis['setup_80']['score']}/100")
                    st.metric("Confidence", analysis['setup_80']['confidence'])
                with col2:
                    st.write("**Reasons:**")
                    for reason in analysis['setup_80']['reasons']:
                        st.write(reason)
                st.markdown("---")
            
            # ORDER FLOW & VOLUME PROFILE
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
                st.metric("POC (Point of Control)", f"${analysis['volume_profile']['poc']:.2f}")
                st.metric("VAH (Value Area High)", f"${analysis['volume_profile']['vah']:.2f}")
                st.metric("VAL (Value Area Low)", f"${analysis['volume_profile']['val']:.2f}")
                poc_distance = abs(analysis['current_price'] - analysis['volume_profile']['poc']) / analysis['current_price'] * 100
                st.metric("Distance from POC", f"{poc_distance:.2f}%")
            
            st.markdown("---")
            
            # MARKET STRUCTURE
            st.subheader("🏗️ Market Structure (SMC)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Structure", analysis['smc']['structure'])
            col2.metric("Bias", analysis['smc']['bias'])
            col3.metric("Order Blocks", len(analysis['smc']['order_blocks']))
            
            if len(analysis['smc']['order_blocks']) > 0:
                st.write("**Recent Order Blocks:**")
                for ob in analysis['smc']['order_blocks']:
                    st.write(f"- {ob['type']}: Top ${ob['top']:.2f} | Bottom ${ob['bottom']:.2f}")
            
            # SFP SIGNALS
            if len(analysis['sfp_signals']) > 0:
                st.markdown("---")
                st.subheader("⚡ Swing Failure Patterns (SFP)")
                for sfp in analysis['sfp_signals']:
                    sfp_type = "🟢 BULLISH" if sfp['type'] == 'BULLISH_SFP' else "🔴 BEARISH"
                    st.write(f"{sfp_type} SFP at ${sfp['price']:.2f} (Level: ${sfp['level']:.2f}) - Score: {sfp['score']}")
            
            # ELLIOTT WAVE
            st.markdown("---")
            st.subheader("🌊 Elliott Wave Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Wave Count", analysis['elliott']['wave_count'])
            col2.metric("Direction", analysis['elliott']['direction'])
            col3.metric("Confidence", f"{analysis['elliott']['confidence']}%")
            
            # CHART
            st.markdown("---")
            st.subheader(f"📈 Technical Chart - {symbol_smc}")
            chart = create_chart_analysis(df, symbol_smc, analysis)
            st.plotly_chart(chart, use_container_width=True)
            
        else:
            st.error("❌ Unable to fetch data for this symbol.")

# TAB 6: BACKTEST
with tab6:
    st.header("🧪 BACKTEST ENGINE")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_symbol = st.text_input("Symbol to Backtest", "AAPL").upper()
    with col2:
        bt_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    with col3:
        bt_capital = st.number_input("Initial Capital", min_value=1000, max_value=1000000, value=10000, step=1000)
    with col4:
        bt_risk_pct = st.number_input("Risk %", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    if st.button("🚀 RUN BACKTEST", type="primary", use_container_width=True):
        with st.spinner(f"🔍 Running backtest on {bt_symbol}..."):
            backtest_results = run_backtest(bt_symbol, bt_period, bt_capital)
        
        if backtest_results:
            st.success("✅ Backtest Complete!")
            stats = backtest_results['stats']
            
            st.markdown("---")
            
            # PERFORMANCE METRICS
            st.subheader("📊 Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", stats['Total Trades'])
            col2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
            col3.metric("Profit Factor", f"{stats['Profit Factor']:.2f}")
            col4.metric("Total P&L", f"${stats['Total P&L']:,.2f}", f"{stats['Total P&L%']:.2f}%")
            col5.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Wins", stats['Wins'])
            col2.metric("Losses", stats['Losses'])
            col3.metric("Avg Win", f"${stats['Avg Win']:.2f}")
            col4.metric("Avg Loss", f"${stats['Avg Loss']:.2f}")
            
            col1, col2 = st.columns(2)
            col1.metric("Max Drawdown", f"{stats['Max Drawdown']:.2f}%")
            col2.metric("Final Balance", f"${stats['Final Balance']:,.2f}")
            
            st.markdown("---")
            
            # EQUITY CURVE
            st.subheader("📈 Equity Curve")
            equity_df = pd.DataFrame({
                'Date': backtest_results['dates'],
                'Balance': backtest_results['equity_curve']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['Date'],
                y=equity_df['Balance'],
                mode='lines',
                name='Balance',
                line=dict(color='cyan', width=2)
            ))
            fig.add_hline(y=bt_capital, line_dash="dash", line_color="orange", annotation_text="Initial Capital")
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Balance ($)",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # TRADE HISTORY
            st.subheader("📋 Trade History")
            st.dataframe(backtest_results['trades'], use_container_width=True, hide_index=True)
            
        else:
            st.error("❌ Backtest failed. Not enough data or no trades generated.")

# TAB 7: SIGNAL TRACKER
with tab7:
    st.header("📊 SIGNAL TRACKER")
    
    with st.expander("🔍 DEBUG MODE"):
        st.subheader("🧪 Session State Debug")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Signals Count", len(st.session_state.active_signals))
            st.metric("Closed Signals Count", len(st.session_state.closed_signals))
        with col2:
            if st.button("➕ ADD TEST SIGNAL (AAPL)", use_container_width=True):
                test_signal = {
                    'symbol': 'AAPL',
                    'entry_price': 195.50,
                    'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'stop': 190.00,
                    'tp1': 200.00,
                    'tp2': 205.00,
                    'tp3': 210.00,
                    'quantum_score': 250.0,
                    'ai_score': 85.0,
                    'tier': '🥇 PLATINUM',
                    'flow': 'BULLISH',
                    'rsi': 55.0,
                    'status': 'ACTIVE',
                    'exit_price': None,
                    'exit_reason': None,
                    'exit_date': None,
                    'pnl': 0,
                    'pnl_pct': 0
                }
                st.session_state.active_signals.append(test_signal)
                st.success(f"✅ Test signal added! Total: {len(st.session_state.active_signals)}")
                st.balloons()
                time.sleep(1)
                st.rerun()
            
            if st.button("🗑️ CLEAR ALL SIGNALS", use_container_width=True):
                st.session_state.active_signals = []
                st.session_state.closed_signals = []
                st.success("✅ All signals cleared!")
                st.rerun()
        
        st.markdown("---")
        st.write("**Raw Session State:**")
        st.json({
            'active_count': len(st.session_state.active_signals),
            'closed_count': len(st.session_state.closed_signals)
        })
    
    if st.button("🔄 REFRESH", use_container_width=True):
        update_signal_status()
        st.success("✅ Updated!")
        st.rerun()
    
    st.markdown("---")
    tab_active, tab_closed = st.tabs(["🟢 Active", "📝 Closed"])
    
    with tab_active:
        if len(st.session_state.active_signals) > 0:
            for idx, sig in enumerate(st.session_state.active_signals):
                current = get_current_price(sig['symbol'])
                if current > 0:
                    pnl = current - sig['entry_price']
                    pnl_pct = ((current / sig['entry_price']) - 1) * 100
                    
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
                        
                        with col1:
                            st.markdown(f"### {sig['tier']}\n**{sig['symbol']}**")
                            st.caption(f"Entry: {sig['entry_date']}")
                        
                        with col2:
                            st.metric("Entry", f"${sig['entry_price']:.2f}")
                            st.metric("Current", f"${current:.2f}")
                        
                        with col3:
                            st.metric("Stop", f"${sig['stop']:.2f}")
                            tp1_hit = "✅" if current >= sig['tp1'] else ""
                            st.metric("TP1", f"${sig['tp1']:.2f}", tp1_hit)
                            st.caption(f"TP2: ${sig['tp2']:.2f} | TP3: ${sig['tp3']:.2f}")
                        
                        with col4:
                            st.metric("Unrealized P&L", f"${pnl:+.2f}", f"{pnl_pct:+.1f}%")
                            progress = min(max((current - sig['entry_price']) / (sig['tp3'] - sig['entry_price']), 0), 1)
                            st.progress(progress)
                            if st.button("❌ Close", key
                            if st.button("❌ Close", key=f"close_{idx}"):
                                sig['exit_price'] = current
                                sig['exit_reason'] = 'MANUAL CLOSE'
                                sig['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                                sig['pnl'] = pnl
                                sig['pnl_pct'] = pnl_pct
                                sig['status'] = 'CLOSED'
                                st.session_state.closed_signals.append(sig)
                                st.session_state.active_signals.pop(idx)
                                st.success(f"✅ Closed {sig['symbol']}")
                                st.rerun()
                        
                        st.markdown("---")
        else:
            st.info("📊 No active signals. Start scanning to track opportunities!")
    
    with tab_closed:
        if len(st.session_state.closed_signals) > 0:
            closed_df = pd.DataFrame(st.session_state.closed_signals)
            
            # STATISTICS
            wins = len(closed_df[closed_df['pnl'] > 0])
            losses = len(closed_df[closed_df['pnl'] <= 0])
            total = len(closed_df)
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = closed_df['pnl'].sum()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", total)
            col2.metric("Wins", wins)
            col3.metric("Losses", losses)
            col4.metric("Win Rate", f"{win_rate:.1f}%")
            col5.metric("Total P&L", f"${total_pnl:+,.2f}")
            
            st.markdown("---")
            
            # TRADE HISTORY
            for sig in reversed(st.session_state.closed_signals):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
                    
                    with col1:
                        st.markdown(f"### {sig['tier']}\n**{sig['symbol']}**")
                        st.caption(f"Entry: {sig['entry_date']}")
                        st.caption(f"Exit: {sig['exit_date']}")
                    
                    with col2:
                        st.metric("Entry", f"${sig['entry_price']:.2f}")
                        st.metric("Exit", f"${sig['exit_price']:.2f}")
                    
                    with col3:
                        st.write(f"**Exit Reason:** {sig['exit_reason']}")
                        st.caption(f"Stop: ${sig['stop']:.2f}")
                        st.caption(f"TP1: ${sig['tp1']:.2f} | TP2: ${sig['tp2']:.2f} | TP3: ${sig['tp3']:.2f}")
                    
                    with col4:
                        color = "normal" if sig['pnl'] > 0 else "inverse"
                        st.metric("Realized P&L", f"${sig['pnl']:+.2f}", f"{sig['pnl_pct']:+.1f}%", delta_color=color)
                        result = "🟢 WIN" if sig['pnl'] > 0 else "🔴 LOSS"
                        st.write(f"**Result:** {result}")
                    
                    st.markdown("---")
            
            st.markdown("---")
            
            # EXPORT
            if st.button("💾 EXPORT CLOSED TRADES", use_container_width=True):
                csv = closed_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "closed_trades.csv", "text/csv", use_container_width=True)
        
        else:
            st.info("📊 No closed trades yet. Tracked signals will appear here when closed.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #ff8c00; padding: 20px;'>
    <h3>🥓 BACON TRADER PRO v7.0 ULTIMATE 🥓</h3>
    <p>⚡ Quantum AI Engine | 🧠 Smart Money Concepts | ⭐ 80% Win Rate Setups</p>
    <p>📊 Signal Tracker | 🧪 Backtesting | 📈 Advanced Charting | 💎 Premium Signals</p>
    <p><strong>Developed with 🥓 by Bacon Team</strong></p>
</div>
""", unsafe_allow_html=True)
