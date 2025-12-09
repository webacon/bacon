import os
import time
from datetime import datetime

import streamlit as st


# -----------------------------
#  Config de la page
# -----------------------------
st.set_page_config(
    page_title="🥓 BaconAlgo - Premium Trading Platform",
    page_icon="🥓",
    layout="wide",
)


# -----------------------------
#  Style global (CSS)
# -----------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #2b150a 0, #050505 45%, #000000 100%);
        color: #ffffff;
    }

    /* En-tête principal */
    .bacon-header {
        background: linear-gradient(135deg, #D2691E, #FF8C00);
        padding: 32px 40px;
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }

    .bacon-header::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 60%);
        top: -80px;
        right: -40px;
        opacity: 0.9;
    }

    .bacon-title {
        font-size: 56px;
        font-weight: 900;
        margin: 0;
        color: #ffffff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
    }

    .bacon-subtitle {
        margin-top: 10px;
        font-size: 20px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
    }

    /* Carte / container */
    .bacon-card {
        background: rgba(15, 15, 15, 0.9);
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.6);
    }

    .bacon-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(0, 0, 0, 0.25);
        color: #ffe9d1;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .bacon-section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Boutons Streamlit */
    .stButton > button {
        background: linear-gradient(135deg, #FF8C00, #FF4500);
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 12px 26px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.7);
        transition: all 0.18s ease-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.85);
        cursor: pointer;
    }

    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
    }

    .metric-delta-pos {
        font-size: 13px;
        font-weight: 600;
        color: #4ade80;
    }

    .metric-delta-neg {
        font-size: 13px;
        font-weight: 600;
        color: #f97373;
    }

    .bacon-footer {
        text-align: center;
        font-size: 12px;
        color: rgba(255, 255, 255, 0.55);
        margin-top: 36px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
#  Header
# -----------------------------
with st.container():
    st.markdown(
        """
        <div class="bacon-header">
            <div class="bacon-pill">🥓  BaconAlgo · Live platform</div>
            <h1 class="bacon-title">BaconAlgo</h1>
            <p class="bacon-subtitle">
                Premium Trading Platform with live signal metrics, win‑rate tracking
                and trader activity monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
#  Métriques principales
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="bacon-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Signals today</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">8</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-delta-pos">+3 vs. yesterday</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="bacon-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Win rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">73%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-delta-pos">+5% last 7 days</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="bacon-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Active traders</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">1,247</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-delta-pos">+89 new today</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")  # petit espace


# -----------------------------
#  Section centrale
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown(
        """
        <div class="bacon-card">
            <div class="bacon-section-title">🚀 Live Trading Dashboard</div>
            <p>
                This is a static preview of the BaconAlgo trading environment.
                Use this area later for your real‑time order flow, signal lists,
                charts or whatever you want to expose to clients.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    start = st.button("🔍 Start Trading Session", use_container_width=True)
    if start:
        st.balloons()
        st.success("Trading session started — dashboard components will plug in here.")

with right:
    st.markdown(
        """
        <div class="bacon-card">
            <div class="bacon-section-title">🔥 System status</div>
        """,
        unsafe_allow_html=True,
    )

    st.success("Core engine: online")
    st.info("Latency: ~32 ms to primary exchange")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Last refresh: {now}")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
#  Footer
# -----------------------------
st.markdown(
    """
    <div class="bacon-footer">
        🛡️ Protected by Cloudflare &nbsp;·&nbsp;
        ⚡ Served from Streamlit Cloud &nbsp;·&nbsp;
        🥓 BaconAlgo © {year}
    </div>
    """.format(year=datetime.utcnow().year),
    unsafe_allow_html=True,
)
