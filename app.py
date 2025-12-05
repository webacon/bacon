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
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

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
    "₿ CRYPTO TOP 30": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
                        "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD",
                        "UNI-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "VET-USD",
                        "ICP-USD", "FIL-USD", "HBAR-USD", "APT-USD", "OP-USD", "ARB-USD",
                        "NEAR-USD", "STX-USD", "IMX-USD", "INJ-USD", "TIA-USD", "RUNE-USD"]
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
            return {'delta': 0, 'cumulative_delta': 0, 'imbalance': 'NEUTRAL', 
                    'whale_detected': False, 'flow_score': 50, 'vol_ratio': 1}
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

# ==================== QUANTUM AI SCORE ====================
def calculate_quantum_ai_score(df, symbol=""):
    if df is None or len(df) < 50:
        return {'quantum_score': 0, 'ai_score': 0, 'recommendation': 'WAIT', 'signal': '⚫',
                'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False, 'atr': 0, 'rsi': 50, 'current_price': 0}
    try:
        atr = calculate_atr(df, 14)
        rsi = calculate_rsi(df, 14)
        macd, macd_signal, macd_hist = calculate_macd(df)
        current_price = float(df['Close'].iloc[-1])
        order_flow = calculate_order_flow_advanced(df)
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
        quantum_ai = (trend_score * 0.25 + rsi_score * 0.15 + macd_score * 0.15 + volume_score * 0.20 + flow_score * 0.25)
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
        return {'quantum_score': round(quantum_score, 1), 'ai_score': round(quantum_ai, 1),
                'recommendation': recommendation, 'signal': signal, 'confidence': confidence, 'tier': tier,
                'whale_alert': order_flow['whale_detected'], 'atr': atr, 'rsi': rsi, 'current_price': current_price,
                'order_flow': order_flow}
    except:
        return {'quantum_score': 0, 'ai_score': 0, 'recommendation': 'ERROR', 'signal': '⚫',
                'confidence': 'LOW', 'tier': 'BRONZE', 'whale_alert': False, 'atr': 0, 'rsi': 50, 'current_price': 0}

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
def scan_market_quantum(symbols, min_score=180, min_ai=60, show_progress=True):
    results = []
    if show_progress:
        progress_bar = st.progress(0)
        status = st.empty()
    for i, symbol in enumerate(symbols):
        if show_progress:
            status.text(f"🔍 Scanning {symbol}... ({i+1}/{len(symbols)})")
        try:
            df = get_live_data(symbol, period="3mo", interval="1d")
            if df is not None and len(df) > 50:
                analysis = calculate_quantum_ai_score(df, symbol)
                if analysis['quantum_score'] >= min_score and analysis['ai_score'] >= min_ai:
                    stop, tp1, tp2, tp3 = calculate_targets(analysis['current_price'], analysis['atr'], analysis['ai_score'])
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
                        'Recommendation': analysis['recommendation']
                    })
        except:
            pass
        if show_progress:
            progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.05)
    if show_progress:
        progress_bar.empty()
        status.empty()
    return pd.DataFrame(results)

# ==================== SIGNAL TRACKER ====================
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
            elif current_price >= signal['tp2']:
                signal['exit_price'] = signal['tp2']
                signal['exit_reason'] = 'TP2 HIT ✅'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['tp2'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['tp2'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
            elif current_price >= signal['tp1']:
                signal['exit_price'] = signal['tp1']
                signal['exit_reason'] = 'TP1 HIT ✅'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['tp1'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['tp1'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
            elif current_price <= signal['stop']:
                signal['exit_price'] = signal['stop']
                signal['exit_reason'] = 'STOP HIT ❌'
                signal['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                signal['pnl'] = signal['stop'] - signal['entry_price']
                signal['pnl_pct'] = ((signal['stop'] / signal['entry_price']) - 1) * 100
                signal['status'] = 'CLOSED'
                to_close.append(idx)
    for idx in reversed(to_close):
        closed_signal = st.session_state.active_signals.pop(idx)
        st.session_state.closed_signals.append(closed_signal)

# ==================== PORTFOLIO ====================
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
            results.append({'Symbol': pos['symbol'], 'Qty': pos['qty'], 'Entry': f"${pos['entry']:.2f}",
                          'Current': f"${current_price:.2f}", 'Stop': f"${stop:.2f}", 'P&L': f"${pnl:.2f}",
                          'P&L%': f"{pnl_pct:+.2f}%", 'Status': status, 'Date': pos['date']})
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

# ==================== THEME ====================
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0a0a0a 0%, #1a1410 100%); color: #ff8c00;}
    div[data-testid="stMetricValue"] {font-size: 28px; color: #ff8c00; font-weight: bold; text-shadow: 0 0 10px #ff8c00;}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 6, 2])
with col1:
    st.markdown("# 🥓")
with col2:
    st.markdown("# BACON TRADER PRO")
    st.caption("⚡ QUANTUM ULTIMATE v5.0 | 📊 Signal Tracker FIXED")
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
    day_stats = calculate_portfolio_stats('day')
    st.subheader("💰 DAY TRADING")
    st.metric("Value", f"${day_stats['total_value']:,.2f}")
    st.metric("P&L", f"${day_stats['total_pnl']:+,.2f}", f"{day_stats['total_pnl_pct']:+.2f}%")
    st.markdown("**➕ Add Day Position:**")
    day_symbol = st.text_input("Symbol", "", placeholder="AAPL").upper()
    day_qty = st.number_input("Quantity", min_value=1, value=10)
    day_entry = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01)
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
    lt_symbol = st.text_input("Symbol ", "", placeholder="MSFT").upper()
    lt_qty = st.number_input("Quantity ", min_value=1, value=50)
    lt_entry = st.number_input("Entry Price ", min_value=0.01, value=150.00, step=0.01)
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
    st.subheader("🎯 QUANTUM SETTINGS")
    min_quantum = st.slider("Min Quantum Score", 0, 300, 180, 10)
    min_ai = st.slider("Min AI Score", 0, 100, 60, 5)
    st.info(f"📊 Current: Q≥{min_quantum} | AI≥{min_ai}")
    st.markdown("---")
    st.caption("🥓 Bacon Trader Pro v5.0")
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dashboard", "🔍 Quantum Scanner", "📊 Signal Tracker", "💼 Portfolios"])

# TAB 1: DASHBOARD
with tab1:
    st.header("🎯 QUANTUM DASHBOARD")
    update_signal_status()
    day_stats = calculate_portfolio_stats('day')
    lt_stats = calculate_portfolio_stats('long')
    total_equity = day_stats['total_value'] + lt_stats['total_value']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity", f"${total_equity:,.0f}")
    col2.metric("Day Trading", f"${day_stats['total_value']:,.0f}", f"{day_stats['total_pnl_pct']:+.2f}%")
    col3.metric("Long-Term", f"${lt_stats['total_value']:,.0f}", f"{lt_stats['total_pnl_pct']:+.2f}%")
    col4.metric("Active Signals", len(st.session_state.active_signals))
    st.markdown("---")
    if len(st.session_state.active_signals) > 0:
        st.subheader("📊 Active Signals Summary")
        for signal in st.session_state.active_signals:
            current = get_current_price(signal['symbol'])
            if current > 0:
                pnl = (current - signal['entry_price'])
                pnl_pct = ((current / signal['entry_price']) - 1) * 100
                col1, col2, col3, col4 = st.columns([2, 3, 3, 2])
                with col1:
                    st.markdown(f"### {signal['tier']} {signal['symbol']}")
                with col2:
                    st.metric("Entry", f"${signal['entry_price']:.2f}")
                    st.metric("Current", f"${current:.2f}")
                with col3:
                    st.metric("TP1", f"${signal['tp1']:.2f}", "✅" if current >= signal['tp1'] else "")
                    st.metric("Stop", f"${signal['stop']:.2f}")
                with col4:
                    st.metric("P&L", f"${pnl:+.2f}", f"{pnl_pct:+.1f}%")
                st.markdown("---")

# TAB 2: QUANTUM SCANNER  
with tab2:
    st.header("🔍 QUANTUM SCANNER")
    col1, col2 = st.columns([3, 1])
    with col1:
        market = st.selectbox("Select Market", list(MARKETS.keys()))
    with col2:
        st.metric("Symbols", len(MARKETS[market]))
    if st.button("🔥 QUANTUM SCAN", type="primary", use_container_width=True):
        st.markdown("---")
        with st.spinner(f"🔍 Scanning {len(MARKETS[market])} symbols..."):
            results = scan_market_quantum(MARKETS[market], min_quantum, min_ai)
        if len(results) > 0:
            st.success(f"✅ Found {len(results)} signals!")
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.subheader("📊 Track Signals")
            for idx, row in results.iterrows():
                with st.expander(f"{row['Tier']} {row['Symbol']} | Q:{row['Quantum']:.0f} AI:{row['AI']:.0f}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entry", f"${row['Entry']:.2f}")
                        st.metric("Stop", f"${row['Stop']:.2f}")
                    with col2:
                        st.metric("TP1", f"${row['TP1']:.2f}")
                        st.metric("TP2", f"${row['TP2']:.2f}")
                    with col3:
                        st.metric("TP3", f"${row['TP3']:.2f}")
                        st.metric("RSI", f"{row['RSI']:.1f}")
                    st.write(f"📈 Flow: {row['Flow']} | 💡 {row['Recommendation']}")
                    if row['Whale'] == '🐋':
                        st.warning("🐋 WHALE DETECTED!")
                    track_button_key = f"track_signal_{row['Symbol']}_{row['Entry']:.2f}_{idx}"
                    if st.button("📊 TRACK THIS SIGNAL", key=track_button_key, type="primary"):
                        new_signal = {
                            'symbol': str(row['Symbol']), 'entry_price': float(row['Entry']),
                            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'stop': float(row['Stop']), 'tp1': float(row['TP1']), 'tp2': float(row['TP2']), 'tp3': float(row['TP3']),
                            'quantum_score': float(row['Quantum']), 'ai_score': float(row['AI']),
                            'tier': str(row['Tier']), 'flow': str(row['Flow']), 'rsi': float(row['RSI']),
                            'status': 'ACTIVE', 'exit_price': None, 'exit_reason': None, 'exit_date': None, 'pnl': 0, 'pnl_pct': 0
                        }
                        exists = any(s['symbol'] == row['Symbol'] and s['status'] == 'ACTIVE' and abs(s['entry_price'] - float(row['Entry'])) < 0.01 for s in st.session_state.active_signals)
                        if not exists:
                            st.session_state.active_signals.append(new_signal)
                            st.balloons()
                            st.success(f"✅ {row['Symbol']} tracked! Total: {len(st.session_state.active_signals)}")
                        else:
                            st.warning(f"⚠️ {row['Symbol']} already tracked!")
        else:
            st.warning("📊 No signals found!")

# TAB 3: SIGNAL TRACKER
with tab3:
    st.header("📊 SIGNAL TRACKER")
    with st.expander("🧪 MANUAL TEST"):
        if st.button("➕ Add Test Signal (AAPL)"):
            st.session_state.active_signals.append({
                'symbol': 'AAPL', 'entry_price': 195.50, 'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'stop': 190.00, 'tp1': 200.00, 'tp2': 205.00, 'tp3': 210.00,
                'quantum_score': 250.0, 'ai_score': 85.0, 'tier': '🥇 PLATINUM',
                'flow': 'BULLISH', 'rsi': 55.0, 'status': 'ACTIVE',
                'exit_price': None, 'exit_reason': None, 'exit_date': None, 'pnl': 0, 'pnl_pct': 0
            })
            st.success(f"✅ Test added! Total: {len(st.session_state.active_signals)}")
    with st.expander("🔧 DEBUG"):
        st.write(f"Active: {len(st.session_state.active_signals)}")
        for sig in st.session_state.active_signals:
            st.write(f"- {sig['symbol']} @ ${sig['entry_price']:.2f}")
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
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
                    with col1:
                        st.markdown(f"### {sig['tier']}\n**{sig['symbol']}**")
                    with col2:
                        st.metric("Entry", f"${sig['entry_price']:.2f}")
                        st.metric("Current", f"${current:.2f}")
                    with col3:
                        st.metric("Stop", f"${sig['stop']:.2f}")
                        st.metric("TP1", f"${sig['tp1']:.2f}", "✅" if current >= sig['tp1'] else "")
                    with col4:
                        st.metric("Unrealized P&L", f"${pnl:+.2f}", f"{pnl_pct:+.1f}%")
                        if st.button("❌ Close", key=f"close_{idx}"):
                            sig['exit_price'] = current
                            sig['exit_reason'] = 'MANUAL'
                            sig['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                            sig['pnl'] = pnl
                            sig['pnl_pct'] = pnl_pct
                            sig['status'] = 'CLOSED'
                            st.session_state.closed_signals.append(sig)
                            st.session_state.active_signals.pop(idx)
                            st.rerun()
                    st.markdown("---")
        else:
            st.info("No active signals")
    with tab_closed:
        if len(st.session_state.closed_signals) > 0:
            closed_df = pd.DataFrame(st.session_state.closed_signals)
            total_pnl = closed_df['pnl'].sum()
            wins = len(closed_df[closed_df['pnl'] > 0])
            win_rate = (wins / len(closed_df)) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(closed_df))
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total P&L", f"${total_pnl:+,.2f}")
            display_df = pd.DataFrame({
                'Symbol': closed_df['symbol'], 'Tier': closed_df['tier'],
                'Entry': closed_df['entry_price'].apply(lambda x: f"${x:.2f}"),
                'Exit': closed_df['exit_price'].apply(lambda x: f"${x:.2f}"),
                'Exit Reason': closed_df['exit_reason'],
                'P&L': closed_df['pnl'].apply(lambda x: f"${x:+.2f}"),
                'P&L%': closed_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%"),
                'Result': closed_df['pnl'].apply(lambda x: '🟢 WIN' if x > 0 else '🔴 LOSS')
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades")

# TAB 4: PORTFOLIOS
with tab4:
    st.header("💼 PORTFOLIOS")
    tab_day, tab_lt = st.tabs(["💰 Day Trading", "📈 Long-Term"])
    with tab_day:
        day_portfolio = get_portfolio_with_live_prices('day')
        if len(day_portfolio) > 0:
            st.dataframe(day_portfolio, use_container_width=True, hide_index=True)
            st.subheader("🗑️ Remove")
            for idx, pos in enumerate(st.session_state.day_positions):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{pos['symbol']} | {pos['qty']} @ ${pos['entry']:.2f}")
                with col2:
                    if st.button("Remove", key=f"rm_day_{idx}"):
                        remove_position('day', idx)
                        st.rerun()
        else:
            st.info("No day positions")
    with tab_lt:
        lt_portfolio = get_portfolio_with_live_prices('long')
        if len(lt_portfolio) > 0:
            st.dataframe(lt_portfolio, use_container_width=True, hide_index=True)
            st.subheader("🗑️ Remove")
            for idx, pos in enumerate(st.session_state.long_positions):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{pos['symbol']} | {pos['qty']} @ ${pos['entry']:.2f}")
                with col2:
                    if st.button("Remove", key=f"rm_lt_{idx}"):
                        remove_position('long', idx)
                        st.rerun()
        else:
            st.info("No long positions")

st.markdown("---")
st.caption("🥓 Bacon Trader Pro v5.0 FINAL | Signal Tracker WORKING")
st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
