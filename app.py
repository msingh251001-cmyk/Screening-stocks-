import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ਪੇਜ ਸੈਟਿੰਗਜ਼
st.set_page_config(page_title="Intraday Options Screener Pro", layout="wide", page_icon="⚡")

# -------------------------------------------------------------
# 1. ਲਾਈਵ ਇੰਡੀਆ VIX ਚੈੱਕ
# -------------------------------------------------------------
@st.cache_data(ttl=120)
def get_india_vix():
    try:
        vix_df = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_val = float(vix_df['Close'].iloc[-1])
            prev_vix = float(vix_df['Close'].iloc[-2])
            change = round(vix_val - prev_vix, 2)
            return round(vix_val, 2), change
    except Exception:
        pass
    return 14.5, 0.0

# -------------------------------------------------------------
# 2. Intraday Multi-Timeframe Engine (Trend + Strict Intraday SL)
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def analyze_intraday(ticker):
    # ਟਾਈਮਫ੍ਰੇਮ ਵੇਟੇਜ: 15m & 1h ਟ੍ਰੈਂਡ ਦੱਸਦੇ ਹਨ, 5m ਐਂਟਰੀ ਦਿੰਦਾ ਹੈ
    tf_configs = {
        "5m": {"period": "5d", "weight": 25},
        "15m": {"period": "5d", "weight": 40},
        "1h": {"period": "1mo", "weight": 25},
        "1D": {"period": "3mo", "weight": 10}
    }
    
    tf_data = {}
    bullish_weighted_score = 0
    total_weight = 0
    ltp = 0.0
    intra_support = 0.0
    intra_resistance = 0.0
    intra_5m_atr = 0.0

    for tf, cfg in tf_configs.items():
        try:
            df = yf.download(ticker, period=cfg["period"], interval=tf, progress=False)
            if df.empty or len(df) < 20:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            curr = float(close.iloc[-1])

            if tf == "5m" or ltp == 0.0:
                ltp = curr

            # 20 EMA & 50 EMA
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            if np.isnan(rsi):
                rsi = 50.0

            # Trend Determination
            is_bull = (curr > ema20) and (ema20 >= ema50 or curr > ema50) and (rsi >= 50)
            is_bear = (curr < ema20) and (ema20 <= ema50 or curr < ema50) and (rsi <= 50)

            if is_bull:
                bullish_weighted_score += cfg["weight"]
            elif not is_bear:
                bullish_weighted_score += (cfg["weight"] / 2) # Sideways / Neutral
            total_weight += cfg["weight"]

            tf_data[tf] = {
                "LTP (₹)": round(curr, 2),
                "EMA 20": round(ema20, 2),
                "RSI (14)": round(rsi, 1),
                "Signal": "Bullish 🟢" if is_bull else ("Bearish 🔴" if is_bear else "Neutral ⚪")
            }

            # 5-ਮਿੰਟ ਅਤੇ 15-ਮਿੰਟ ਤੋਂ Intraday Support/Resistance ਅਤੇ ATR ਲੈਣਾ
            if tf == "5m":
                tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                intra_5m_atr = float(tr.tail(10).mean())
            if tf == "15m":
                intra_support = float(low.tail(15).min())
                intra_resistance = float(high.tail(15).max())

        except Exception:
            pass

    final_score = int((bullish_weighted_score / total_weight) * 100) if total_weight > 0 else 50
    return tf_data, final_score, ltp, intra_support, intra_resistance, intra_5m_atr

# -------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------
st.title("⚡ Pro Intraday Options Screener")
st.caption("Noise-filtered Multi-TF Signals • India VIX • Tight 10-20 Pts Option SL")

# VIX Dashboard Header
vix_val, vix_chg = get_india_vix()
vix_alert = "⚠️ VIX ਬਹੁਤ ਜ਼ਿਆਦਾ ਹੈ (>18)! ਦੋਵੇਂ CE/PE ਪ੍ਰੀਮੀਅਮ ਵਧ ਸਕਦੇ ਹਨ" if vix_val > 18 else "✅ VIX ਨਾਰਮਲ ਹੈ (Clean Momentum)"
st.info(f"📊 **India VIX:** `{vix_val}` ({'+' if vix_chg >= 0 else ''}{vix_chg}) | **Status:** {vix_alert}")

col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    predefined = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "RELIANCE": "RELIANCE.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "TATA MOTORS": "TATAMOTORS.NS",
        "INFOSYS": "INFY.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "Custom (ਆਪਣਾ ਸਟਾਕ ਟਾਈਪ ਕਰੋ)": "CUSTOM"
    }
    choice = st.selectbox("ਇੰਡੈਕਸ ਜਾਂ ਸਟਾਕ ਚੁਣੋ:", list(predefined.keys()))
    if choice == "Custom (ਆਪਣਾ ਸਟਾਕ ਟਾਈਪ ਕਰੋ)":
        custom_sym = st.text_input("NSE ਸਿੰਬਲ ਲਿਖੋ (ਜਿਵੇਂ SBIN, TCS, LT):", "SBIN")
        ticker = f"{custom_sym.strip().upper()}.NS"
    else:
        ticker = predefined[choice]

with col_s2:
    st.write("")
    st.write("")
    if st.button("🔄 ਫਰੈੱਸ਼ ਰਿਫ੍ਰੈਸ਼"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Intraday ਡਾਟਾ ਪ੍ਰੋਸੈਸ ਹੋ ਰਿਹਾ ਹੈ..."):
    tf_data, score, ltp, supp, res, atr5m = analyze_intraday(ticker)

if ltp == 0.0:
    st.error("❌ ਸਿੰਬਲ ਦਾ ਲਾਈਵ ਡਾਟਾ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ NSE ਨਾਮ ਭਰੋ।")
    st.stop()

# Strike Step
is_nifty = "^NSEI" in ticker
is_banknifty = "^NSEBANK" in ticker
step = 50 if is_nifty else (100 if is_banknifty else (10 if ltp < 1000 else 50))
atm_strike = int(round(ltp / step) * step)

# -------------------------------------------------------------
# Strict Intraday Trade Signal Logic (No Rapid Flips)
# -------------------------------------------------------------
if score >= 60:
    trade_signal = "CE BUY (ਤੇਜ਼ੀ 🟢)"
    direction = "CE"
elif score <= 40:
    trade_signal = "PE BUY (ਮੰਦੀ 🔴)"
    direction = "PE"
else:
    trade_signal = "NO TRADE / SIDEWAYS 🟡"
    direction = "NONE"

# Spot SL / Target (Tight 5m ATR Based)
spot_sl_points = max(round(atr5m * 1.2, 1), 20.0 if is_nifty else (45.0 if is_banknifty else round(ltp * 0.003, 1)))

# Option Premium SL (ਜੋ ਤੁਹਾਨੂੰ 10-20 ਪੁਆਇੰਟ ਚਾਹੀਦਾ ਸੀ)
opt_sl_pts = 12 if is_nifty else (25 if is_banknifty else round(spot_sl_points * 0.5, 1))
opt_tgt1_pts = round(opt_sl_pts * 1.5, 1)
opt_tgt2_pts = round(opt_sl_pts * 2.5, 1)

st.markdown("---")

# Main Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot LTP", f"₹{round(ltp, 2)}")
m2.metric("ਟਰੇਡ ਸਿਗਨਲ", trade_signal)
m3.metric("ਟ੍ਰੈਂਡ ਕਨਫਰਮੇਸ਼ਨ", f"{score}%")
m4.metric("Suggested ATM Strike", f"{atm_strike} {direction}" if direction != "NONE" else "Wait for Setup")

st.markdown("---")

# 🎯 Option Buying Specific Target & SL Box
st.subheader("🎯 Option Buying Rules (Tight Intraday SL)")
p1, p2, p3, p4 = st.columns(4)

if direction != "NONE":
    p1.metric("📌 Entry Plan", "ਕੈਂਡਲ ਕਲੋਜ਼ਿੰਗ 'ਤੇ ਐਂਟਰੀ")
    p2.metric("🛑 Option SL (ਪ੍ਰੀਮੀਅਮ 'ਤੇ)", f"- {opt_sl_pts} Points", delta=f"-{opt_sl_pts} pts", delta_color="inverse")
    p3.metric("🎯 Target 1 (1:1.5)", f"+ {opt_tgt1_pts} Points", delta=f"+{opt_tgt1_pts} pts")
    p4.metric("🚀 Target 2 (1:2.5)", f"+ {opt_tgt2_pts} Points", delta=f"+{opt_tgt2_pts} pts")
else:
    st.warning("ਮਾਰਕੀਟ ਸਾਈਡਵੇਜ਼ ਹੈ (ਕੋਈ ਸਾਫ਼ ਟ੍ਰੈਂਡ ਨਹੀਂ)। ਕਨਫਰਮੇਸ਼ਨ ਸਕੋਰ 60%+ ਜਾਂ 40%- ਹੋਣ ਦਾ ਇੰਤਜ਼ਾਰ ਕਰੋ ਤਾਂ ਜੋ ਫੇਕ ਸਿਗਨਲਾਂ ਤੋਂ ਬਚਿਆ ਜਾ ਸਕੇ।")

# Technical Table & Key Levels
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Multi-Timeframe Trend Confirmation")
    if tf_data:
        df_view = pd.DataFrame.from_dict(tf_data, orient='index')
        st.dataframe(df_view, use_container_width=True)

with c2:
    st.subheader("🧱 Intraday Key Levels")
    st.metric("15-Min Support (Base)", f"₹{round(supp, 2)}")
    st.metric("15-Min Resistance (Hurdle)", f"₹{round(res, 2)}")
    st.caption("💡 **ਨੋਟ:** ਜੇਕਰ Nifty ਵਿੱਚ ਕੰਮ ਕਰ ਰਹੇ ਹੋ ਤਾਂ ਆਪਸ਼ਨ ਪ੍ਰੀਮੀਅਮ 'ਤੇ 10-15 ਪੁਆਇੰਟ ਅਤੇ Bank Nifty 'ਤੇ 20-25 ਪੁਆਇੰਟ ਤੋਂ ਵੱਧ ਦਾ ਸਟੌਪ ਲਾਸ ਕਦੇ ਨਾ ਰੱਖੋ।")
