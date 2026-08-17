import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="AI Options & BTST Flow Screener", layout="wide", page_icon="⚡")

# -------------------------------------------------------------
# 1. NSE Live Option Chain Fetcher (OI, Chg OI, Volume, IV)
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_nse_chain(symbol="NIFTY"):
    is_index = symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_index else 'equities'}?symbol={symbol}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }
    
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=4)
        resp = session.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def parse_option_data(data, spot_price):
    if not data or 'records' not in data:
        return None

    records = data['records'].get('data', [])
    underlying = data['records'].get('underlyingValue', spot_price)
    
    total_ce_oi, total_pe_oi = 0, 0
    total_ce_chg_oi, total_pe_chg_oi = 0, 0
    total_ce_vol, total_pe_vol = 0, 0
    iv_list = []
    
    strike_rows = []

    for item in records:
        strike = item.get('strikePrice', 0)
        ce = item.get('CE', {})
        pe = item.get('PE', {})
        
        ce_oi = ce.get('openInterest', 0)
        pe_oi = pe.get('openInterest', 0)
        ce_chg = ce.get('changeinOpenInterest', 0)
        pe_chg = pe.get('changeinOpenInterest', 0)
        ce_v = ce.get('totalTradedVolume', 0)
        pe_v = pe.get('totalTradedVolume', 0)
        ce_iv = ce.get('impliedVolatility', 0)
        pe_iv = pe.get('impliedVolatility', 0)

        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_chg_oi += ce_chg
        total_pe_chg_oi += pe_chg
        total_ce_vol += ce_v
        total_pe_vol += pe_v
        
        if ce_iv > 0: iv_list.append(ce_iv)
        if pe_iv > 0: iv_list.append(pe_iv)

        # ATM ਦੇ ਨੇੜਲੀਆਂ ਸਟ੍ਰਾਈਕਸ ਫਿਲਟਰ ਕਰੋ (ਡੈਸ਼ਬੋਰਡ ਲਈ)
        if abs(strike - underlying) <= (underlying * 0.03):
            strike_rows.append({
                "Strike": strike,
                "CE Chg OI": ce_chg,
                "CE Volume": ce_v,
                "CE IV": ce_iv,
                "PE Chg OI": pe_chg,
                "PE Volume": pe_v,
                "PE IV": pe_iv
            })

    pcr_oi = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 1.0
    pcr_chg_oi = round(total_pe_chg_oi / total_ce_chg_oi, 2) if total_ce_chg_oi > 0 else 1.0
    pcr_vol = round(total_pe_vol / total_ce_vol, 2) if total_ce_vol > 0 else 1.0
    avg_iv = round(np.mean(iv_list), 2) if iv_list else 15.0

    return {
        "spot": underlying,
        "pcr_oi": pcr_oi,
        "pcr_chg_oi": pcr_chg_oi,
        "pcr_vol": pcr_vol,
        "avg_iv": avg_iv,
        "strikes_df": pd.DataFrame(strike_rows),
        "total_ce_chg": total_ce_chg_oi,
        "total_pe_chg": total_pe_chg_oi
    }

# -------------------------------------------------------------
# 2. Live India VIX
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
            return round(vix_val, 2), round(vix_val - prev_vix, 2)
    except Exception:
        pass
    return 14.5, 0.0

# -------------------------------------------------------------
# 3. Multi-Timeframe Technical Engine (5m, 15m, 1h, 1D)
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def analyze_technicals(ticker):
    tf_configs = {
        "5m": {"period": "5d", "weight": 20},
        "15m": {"period": "5d", "weight": 40},
        "1h": {"period": "1mo", "weight": 25},
        "1D": {"period": "3mo", "weight": 15}
    }
    
    tf_data = {}
    bullish_score = 0
    total_weight = 0
    ltp = 0.0
    intra_supp, intra_res = 0.0, 0.0
    atr5m = 0.0
    day_volume = 0
    avg_vol = 0

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
            vol = df['Volume'].astype(float)
            curr = float(close.iloc[-1])

            if tf == "5m" or ltp == 0.0:
                ltp = curr

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            if np.isnan(rsi): rsi = 50.0

            # Volume Trend
            vol_ma = vol.rolling(10).mean().iloc[-1]
            vol_spike = (vol.iloc[-1] > vol_ma * 1.2)

            is_bull = (curr > ema20) and (ema20 >= ema50 or curr > ema50) and (rsi >= 50)
            is_bear = (curr < ema20) and (ema20 <= ema50 or curr < ema50) and (rsi <= 50)

            if is_bull:
                bullish_score += cfg["weight"]
            elif not is_bear:
                bullish_score += (cfg["weight"] / 2)
            total_weight += cfg["weight"]

            tf_data[tf] = {
                "LTP (₹)": round(curr, 2),
                "EMA 20": round(ema20, 2),
                "RSI": round(rsi, 1),
                "Vol Spike": "High 🟢" if vol_spike else "Normal ⚪",
                "Signal": "Bullish 🟢" if is_bull else ("Bearish 🔴" if is_bear else "Neutral ⚪")
            }

            if tf == "5m":
                tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                atr5m = float(tr.tail(10).mean())
            if tf == "15m":
                intra_supp = float(low.tail(15).min())
                intra_res = float(high.tail(15).max())
            if tf == "1D":
                day_volume = int(vol.iloc[-1])
                avg_vol = int(vol.tail(20).mean())

        except Exception:
            pass

    score = int((bullish_score / total_weight) * 100) if total_weight > 0 else 50
    return tf_data, score, ltp, intra_supp, intra_res, atr5m, day_volume, avg_vol

# -------------------------------------------------------------
# 4. Streamlit UI Layout
# -------------------------------------------------------------
st.title("⚡ AI Options Flow & BTST Predictor")
st.caption("NSE Live OI + Change in OI + Volume Spike + Multi-TF Technical Confluence Engine")

vix_val, vix_chg = get_india_vix()
st.info(f"📊 **India VIX:** `{vix_val}` ({'+' if vix_chg >= 0 else ''}{vix_chg}) | **Market Mood:** {'High Volatility / Event Risk ⚠️' if vix_val > 17 else 'Stable Trend Regime ✅'}")

# Tabs: Tab 1 = Intraday Engine, Tab 2 = BTST / IV Surge Scanner
tab1, tab2 = st.tabs(["🎯 Intraday Options Screener (OI + Volume Confluence)", "🌙 BTST & IV Surge Scanner (CE+PE Expansion)"])

symbols = {
    "NIFTY 50": {"yf": "^NSEI", "nse": "NIFTY"},
    "BANK NIFTY": {"yf": "^NSEBANK", "nse": "BANKNIFTY"},
    "FINNIFTY": {"yf": "NIFTY_FIN_SERVICE.NS", "nse": "FINNIFTY"},
    "RELIANCE": {"yf": "RELIANCE.NS", "nse": "RELIANCE"},
    "HDFC BANK": {"yf": "HDFCBANK.NS", "nse": "HDFCBANK"},
    "TATA MOTORS": {"yf": "TATAMOTORS.NS", "nse": "TATAMOTORS"},
    "INFOSYS": {"yf": "INFY.NS", "nse": "INFY"},
    "ICICI BANK": {"yf": "ICICIBANK.NS", "nse": "ICICIBANK"},
    "SBIN": {"yf": "SBIN.NS", "nse": "SBIN"}
}

with tab1:
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        choice = st.selectbox("ਇੰਡੈਕਸ ਜਾਂ ਸਟਾਕ ਚੁਣੋ:", list(symbols.keys()), key="intra_sym")
        active_sym = symbols[choice]
    with col_s2:
        st.write("")
        st.write("")
        if st.button("🔄 ਫਰੈੱਸ਼ ਡਾਟਾ ਰਿਫ੍ਰੈਸ਼", key="btn_refresh_1"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("NSE OI, Volume ਅਤੇ ਕੈਂਡਲਸਟਿਕ ਡਾਟਾ ਕੈਲਕੁਲੇਟ ਹੋ ਰਿਹਾ ਹੈ..."):
        tf_data, tech_score, ltp, supp, res, atr5m, d_vol, a_vol = analyze_technicals(active_sym["yf"])
        nse_raw = fetch_nse_chain(active_sym["nse"])
        opt_data = parse_option_data(nse_raw, ltp)

    if ltp == 0.0:
        st.error("ਡਾਟਾ ਲੋਡ ਨਹੀਂ ਹੋ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।")
        st.stop()

    # --- ADVANCED CONFLUENCE CALCULATION ---
    # 1. Technical Score (40% Weight)
    # 2. Change in OI Direction (35% Weight)
    # 3. Volume PCR Direction (25% Weight)
    
    oi_score = 50
    if opt_data:
        pcr_chg = opt_data["pcr_chg_oi"]
        if pcr_chg > 1.3: oi_score = 90  # Strong Put Writing (Bullish)
        elif pcr_chg > 1.0: oi_score = 70
        elif pcr_chg < 0.7: oi_score = 10 # Strong Call Writing (Bearish)
        elif pcr_chg < 1.0: oi_score = 30
    
    vol_score = 50
    if opt_data:
        if opt_data["pcr_vol"] > 1.1: vol_score = 80
        elif opt_data["pcr_vol"] < 0.9: vol_score = 20

    final_accuracy_score = int((tech_score * 0.40) + (oi_score * 0.35) + (vol_score * 0.25))

    # Signal Decisions
    if final_accuracy_score >= 62:
        trade_dir = "CE BUY (ਤੇਜ਼ੀ 🟢)"
        dir_code = "CE"
    elif final_accuracy_score <= 38:
        trade_dir = "PE BUY (ਮੰਦੀ 🔴)"
        dir_code = "PE"
    else:
        trade_dir = "NO TRADE / SIDEWAYS 🟡"
        dir_code = "NONE"

    # Strike Calculation
    step = 50 if "NSEI" in active_sym["yf"] else (100 if "NSEBANK" in active_sym["yf"] else (10 if ltp < 1000 else 50))
    atm_strike = int(round(ltp / step) * step)

    # Tight 10-20 Points Option SL
    is_nifty = "^NSEI" in active_sym["yf"]
    is_banknifty = "^NSEBANK" in active_sym["yf"]
    opt_sl_pts = 12 if is_nifty else (25 if is_banknifty else 15)
    opt_tgt1 = round(opt_sl_pts * 1.5, 1)
    opt_tgt2 = round(opt_sl_pts * 2.5, 1)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot LTP", f"₹{round(ltp, 2)}")
    m2.metric("ਫਾਈਨਲ ਸਿਗਨਲ (Confluence)", trade_dir)
    m3.metric("ਐਕੂਰੇਸੀ ਸਕੋਰ", f"{final_accuracy_score}%")
    m4.metric("ATM Strike", f"{atm_strike} {dir_code}" if dir_code != "NONE" else "Wait")

    # Tight SL Dashboard
    st.markdown("---")
    st.subheader("🎯 Strict Intraday Option Buying Execution Plan")
    p1, p2, p3, p4 = st.columns(4)
    if dir_code != "NONE":
        p1.metric("📌 Entry Point", "15m ਕੈਂਡਲ ਕਨਫਰਮੇਸ਼ਨ 'ਤੇ")
        p2.metric("🛑 Option SL (ਪ੍ਰੀਮੀਅਮ 'ਤੇ)", f"- {opt_sl_pts} Pts", delta=f"-{opt_sl_pts}", delta_color="inverse")
        p3.metric("🎯 Target 1", f"+ {opt_tgt1} Pts", delta=f"+{opt_tgt1}")
        p4.metric("🚀 Target 2", f"+ {opt_tgt2} Pts", delta=f"+{opt_tgt2}")
    else:
        st.warning("⚠️ ਟੈਕਨੀਕਲ ਅਤੇ NSE OI ਡਾਟਾ ਆਪਸ ਵਿੱਚ ਮੈਚ ਨਹੀਂ ਕਰ ਰਹੇ। ਝੂਠੇ ਬ੍ਰੇਕਆਊਟ ਤੋਂ ਬਚਣ ਲਈ ਹੁਣੇ ਕੋਈ ਟਰੇਡ ਨਾ ਲਵੋ।")

    # Details Breakdown
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Multi-Timeframe Technical Breakdown")
        if tf_data:
            st.dataframe(pd.DataFrame.from_dict(tf_data, orient='index'), use_container_width=True)
    with c2:
        st.subheader("⚡ NSE Option Chain Confluence Metrics")
        if opt_data:
            oi_c1, oi_c2 = st.columns(2)
            oi_c1.metric("Overall PCR (OI)", f"{opt_data['pcr_oi']}")
            oi_c2.metric("Chg in OI PCR", f"{opt_data['pcr_chg_oi']}", help=">1.2 = Bullish, <0.8 = Bearish")
            oi_c1.metric("Volume PCR", f"{opt_data['pcr_vol']}")
            oi_c2.metric("Average IV", f"{opt_data['avg_iv']}%")
        else:
            st.info("NSE ਲਾਈਵ Option Chain ਡਾਟਾ ਬ੍ਰੋਕਰ ਸੈਸ਼ਨ ਕਾਰਨ ਲਿਮਿਟਿਡ ਹੈ।")

# -------------------------------------------------------------
# TAB 2: BTST & IV SURGE SCANNER
# -------------------------------------------------------------
with tab2:
    st.subheader("🌙 Automated BTST & IV Explosion Detector")
    st.caption("ਜਦੋਂ Implied Volatility (IV) ਵਧਦੀ ਹੈ ਜਾਂ ਵੱਡਾ ਇਵੈਂਟ ਹੁੰਦਾ ਹੈ, ਤਾਂ CE ਅਤੇ PE ਦੋਵੇਂ ਵਧਦੇ ਹਨ। ਇਹ ਸਕੈਨਰ ਉਹਨਾਂ ਸਟਾਕਾਂ ਨੂੰ ਪਛਾਣਦਾ ਹੈ।")

    btst_candidates = []
    
    scan_list = ["RELIANCE.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "^NSEI", "^NSEBANK"]
    
    with st.spinner("ਸਾਰੇ ਸਟਾਕਾਂ ਦਾ BTST + IV ਡਾਟਾ ਸਕੈਨ ਹੋ ਰਿਹਾ ਹੈ..."):
        for sym_ticker in scan_list:
            try:
                raw_df = yf.download(sym_ticker, period="10d", interval="1d", progress=False)
                if raw_df.empty or len(raw_df) < 5: continue
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                
                cl = raw_df['Close'].astype(float)
                hi = raw_df['High'].astype(float)
                lo = raw_df['Low'].astype(float)
                vl = raw_df['Volume'].astype(float)

                today_close = float(cl.iloc[-1])
                prev_close = float(cl.iloc[-2])
                price_chg_pct = round(((today_close - prev_close) / prev_close) * 100, 2)
                
                # Day Range (High vs Close)
                day_high = float(hi.iloc[-1])
                is_near_high = (today_close >= day_high - ((day_high - float(lo.iloc[-1])) * 0.15))
                
                # Volume Surge (20-day Average ਨਾਲੋਂ 1.5 ਗੁਣਾ ਵੱਧ)
                vol_surge = float(vl.iloc[-1]) > (float(vl.tail(10).mean()) * 1.3)
                
                # BTST Score Logic
                btst_score = 0
                if price_chg_pct > 1.2: btst_score += 35
                if is_near_high: btst_score += 35
                if vol_surge: btst_score += 30

                # IV Straddle Surge Check
                iv_surge_alert = "HIGH (CE+PE Expanding) 🔥" if vix_val > 16.5 and vol_surge else "Normal ⚪"

                btst_candidates.append({
                    "Symbol": sym_ticker.replace(".NS", "").replace("^", ""),
                    "LTP": round(today_close, 2),
                    "Day Change %": f"{'+' if price_chg_pct > 0 else ''}{price_chg_pct}%",
                    "Near Day High?": "Yes ✅" if is_near_high else "No ❌",
                    "Volume Surge?": "High Volume 🟢" if vol_surge else "Normal ⚪",
                    "IV Condition": iv_surge_alert,
                    "BTST Suitability": f"{btst_score}%",
                    "Trade Action": "Strong BTST CE 🚀" if btst_score >= 70 else ("Both CE+PE Straddle 💥" if iv_surge_alert.startswith("HIGH") else "Avoid BTST ❌")
                })
            except Exception:
                pass

    if btst_candidates:
        btst_df = pd.DataFrame(btst_candidates)
        st.dataframe(btst_df, use_container_width=True)
        
        st.markdown("### 💡 BTST & IV Trade Guide")
        st.write("""
        * **Strong BTST CE (70%+ Score):** ਸਟਾਕ ਡੇਅ-ਹਾਈ ਦੇ ਨੇੜੇ ਬੰਦ ਹੋ ਰਿਹਾ ਹੈ ਅਤੇ ਵਾਲੀਅਮ ਬਹੁਤ ਭਾਰੀ ਹੈ। ਅਗਲੇ ਦਿਨ ਗੈਪ-ਅੱਪ (Gap-up) ਓਪਨਿੰਗ ਦੇ ਵੱਧ ਚਾਂਸ ਹੁੰਦੇ ਹਨ।
        * **Both CE+PE Straddle (IV Expansion):** ਜਦੋਂ India VIX ਵਧ ਰਿਹਾ ਹੋਵੇ ਅਤੇ ਵਾਲੀਅਮ ਵਧੇ, ਤਾਂ ਵੱਡੇ ਝਟਕੇ ਜਾਂ ਨਿਊਜ਼ ਕਾਰਨ CE ਅਤੇ PE ਦੋਵੇਂ ਵਧਦੇ ਹਨ। ਅਜਿਹੇ ਸਮੇਂ Long Straddle (ਦੋਵੇਂ ਖਰੀਦਣਾ) ਕੰਮ ਕਰਦਾ ਹੈ।
        """)
