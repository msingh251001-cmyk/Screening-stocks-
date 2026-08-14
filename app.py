import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ਪੇਜ ਸੈਟਿੰਗਜ਼
st.set_page_config(page_title="AI Options & Stock Screener Pro", layout="wide", page_icon="⚡")

# -------------------------------------------------------------
# 1. ਲਾਈਵ ਇੰਡੀਆ VIX ਚੈੱਕ ਕਰਨਾ
# -------------------------------------------------------------
@st.cache_data(ttl=120)
def get_india_vix():
    try:
        vix_df = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        if not vix_df.empty:
            vix_val = float(vix_df['Close'].iloc[-1].item() if hasattr(vix_df['Close'].iloc[-1], 'item') else vix_df['Close'].iloc[-1])
            prev_vix = float(vix_df['Close'].iloc[-2].item() if hasattr(vix_df['Close'].iloc[-2], 'item') else vix_df['Close'].iloc[-2])
            change = round(vix_val - prev_vix, 2)
            return round(vix_val, 2), change
    except Exception:
        pass
    return 14.5, 0.0

# -------------------------------------------------------------
# 2. Multi-Timeframe Technical Engine (5m, 15m, 1h, 1d)
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def analyze_stock(ticker):
    intervals = {"5m": ("5d", "5m"), "15m": ("5d", "15m"), "1h": ("1mo", "1h"), "1D": ("6mo", "1d")}
    tf_data = {}
    bullish_votes = 0
    total_valid_tf = 0
    latest_close = 0.0
    support = 0.0
    resistance = 0.0
    atr_val = 0.0

    for tf, (prd, intr) in intervals.items():
        try:
            df = yf.download(ticker, period=prd, interval=intr, progress=False)
            if df.empty or len(df) < 25:
                continue

            # Multi-index ਕਾਲਮ ਸਾਫ਼ ਕਰਨਾ
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            
            curr_price = float(close.iloc[-1])
            if tf == "5m" or latest_close == 0.0:
                latest_close = curr_price

            # Indicators (EMA 20, EMA 50)
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

            is_bullish = curr_price > ema20 and ema20 >= ema50 and rsi > 50
            if is_bullish:
                bullish_votes += 1
            total_valid_tf += 1

            tf_data[tf] = {
                "LTP (₹)": round(curr_price, 2),
                "EMA 20": round(ema20, 2),
                "RSI": round(rsi, 1),
                "Trend": "Bullish 🟢" if is_bullish else "Bearish 🔴"
            }

            # 1D ਜਾਂ 1h ਤੋਂ Support / Resistance / ATR ਲੈਣਾ
            if tf in ["15m", "1h", "1D"]:
                support = float(low.tail(20).min())
                resistance = float(high.tail(20).max())
                # ATR ਅੰਦਾਜ਼ਾ
                tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                atr_val = float(tr.tail(14).mean())

        except Exception:
            pass

    score = int((bullish_votes / total_valid_tf) * 100) if total_valid_tf > 0 else 50
    return tf_data, score, latest_close, support, resistance, atr_val

# -------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------
st.title("⚡ AI Options & Stock Screener (Live Flow)")
st.caption("Auto Technicals + India VIX + Strike + Entry/Target/SL Calculator")

# VIX Header Bar
vix_val, vix_chg = get_india_vix()
vix_status = "High Volatility (CE/PE ਦੋਵੇਂ ਵਧਣ ਦਾ ਰਿਸਕ)" if vix_val > 18 else "Normal Volatility"
st.info(f"📊 **India VIX:** `{vix_val}` ({'+' if vix_chg >= 0 else ''}{vix_chg}) | **ਸਥਿਤੀ:** {vix_status}")

# ਸਰਚ ਅਤੇ ਸਿਲੈਕਸ਼ਨ ਸੈਕਸ਼ਨ
col_search, col_btn = st.columns([3, 1])

with col_search:
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
        custom_input = st.text_input("NSE ਸਟਾਕ ਸਿੰਬਲ ਲਿਖੋ (ਜਿਵੇਂ SBIN, TCS, ITC):", "SBIN")
        ticker = f"{custom_input.strip().upper()}.NS"
    else:
        ticker = predefined[choice]

with col_btn:
    st.write("")
    st.write("")
    if st.button("🔄 ਫਰੈੱਸ਼ ਡਾਟਾ ਲਿਆਓ"):
        st.cache_data.clear()
        st.rerun()

# ਡਾਟਾ ਫੈਚ ਕਰਨਾ
with st.spinner("ਡਾਟਾ ਅਤੇ ਟੈਕਨੀਕਲ ਲੈਵਲ ਕੈਲਕੁਲੇਟ ਹੋ ਰਹੇ ਹਨ..."):
    tf_data, confidence_score, ltp, supp, res, atr = analyze_stock(ticker)

if ltp == 0.0:
    st.error("❌ ਸਿੰਬਲ ਦਾ ਡਾਟਾ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ NSE ਨਾਮ ਭਰੋ।")
    st.stop()

# ਟਰੇਡ ਡਾਇਰੈਕਸ਼ਨ
is_buy_ce = confidence_score >= 50
trade_type = "CE BUY (ਤੇਜ਼ੀ 🟢)" if is_buy_ce else "PE BUY (ਮੰਦੀ 🔴)"

# Strike Price (ਸਭ ਤੋਂ ਨੇੜਲੀ ATM Strike)
step = 50 if "NSEI" in ticker else (100 if "NSEBANK" in ticker else 10 if ltp < 1000 else 50)
atm_strike = int(round(ltp / step) * step)

# Entry, SL, Target Calculations
risk_buffer = atr if atr > 0 else (ltp * 0.006)
if is_buy_ce:
    entry_price = round(ltp, 2)
    stop_loss = round(ltp - (risk_buffer * 1.0), 2)
    target_1 = round(ltp + (risk_buffer * 1.5), 2)
    target_2 = round(ltp + (risk_buffer * 2.5), 2)
else:
    entry_price = round(ltp, 2)
    stop_loss = round(ltp + (risk_buffer * 1.0), 2)
    target_1 = round(ltp - (risk_buffer * 1.5), 2)
    target_2 = round(ltp - (risk_buffer * 2.5), 2)

st.markdown("---")

# Main Metric Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot LTP", f"₹{round(ltp, 2)}")
m2.metric("ਟਰੇਡ ਸਿਗਨਲ", trade_type)
m3.metric("ਸੰਭਾਵਨਾ (Confidence)", f"{confidence_score}%")
m4.metric("ਸਿਫਾਰਿਸ਼ ਕੀਤੀ Strike", f"{atm_strike} {'CE' if is_buy_ce else 'PE'}")

st.markdown("---")

# Entry / Target / SL Dashboard
st.subheader("🎯 ਟਰੇਡ ਪਲੈਨ (Entry, SL, Targets)")
p1, p2, p3, p4 = st.columns(4)
p1.metric("📍 Entry Point", f"₹{entry_price}")
p2.metric("🛑 Stop Loss (SL)", f"₹{stop_loss}")
p3.metric("🎯 Target 1 (1:1.5)", f"₹{target_1}")
p4.metric("🚀 Target 2 (1:2.5)", f"₹{target_2}")

# Timeframe Analysis & S/R
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Multi-Timeframe Signals")
    if tf_data:
        df_view = pd.DataFrame.from_dict(tf_data, orient='index')
        st.dataframe(df_view, use_container_width=True)

with c2:
    st.subheader("🧱 Key Levels (Support & Resistance)")
    st.metric("Strong Resistance (R1/Peak)", f"₹{round(res, 2)}")
    st.metric("Strong Support (S1/Base)", f"₹{round(supp, 2)}")
    if vix_val > 18:
        st.warning("⚠️ ਧਿਆਨ ਦਿਓ: VIX ਵਧਿਆ ਹੋਇਆ ਹੈ। ਸਟ੍ਰਿਕਟ ਸਟੌਪ-ਲਾਸ (SL) ਦੀ ਵਰਤੋਂ ਕਰੋ!")
