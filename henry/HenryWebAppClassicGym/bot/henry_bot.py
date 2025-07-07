import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

def run_henry_bot(api_key, secret, email, live_trading):
    st.write("✅ Henry bot function started")

    def fetch_demo_data():
        np.random.seed(42)
        steps = 120
        prices = np.cumsum(np.random.randn(steps) * 20 + 20000)
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=steps, freq="5min"),
            "open": prices,
            "high": prices + np.random.rand(steps) * 10,
            "low": prices - np.random.rand(steps) * 10,
            "close": prices + np.random.randn(steps),
            "volume": np.random.rand(steps) * 100
        })
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["ema"] = EMAIndicator(close=df["close"], window=14).ema_indicator()
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        return df.dropna().reset_index(drop=True)

    # Use demo data
    try:
        df = fetch_demo_data()
        st.info("🔧 Running in DEMO mode with synthetic data.")
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return

    if len(df) < 20:
        st.warning("Not enough data to simulate trades.")
        return

    # Inject demo trades
    df["action"] = ""
    df.loc[10, "action"] = "Buy"
    df.loc[30, "action"] = "Buy"
    df.loc[50, "action"] = "Buy"
    df.loc[20, "action"] = "Sell"
    df.loc[60, "action"] = "Sell"
    df.loc[100, "action"] = "Sell"

    df["step"] = range(len(df))

    # Show price chart
    st.subheader("📈 Henry's Demo Price Chart")
    st.line_chart(df.set_index("step")[["close"]])

    # Trade log
    st.subheader("📜 Trade Timeline")
    for _, row in df[df["action"] != ""].iterrows():
        if row["action"] == "Buy":
            st.markdown(f"🟢 **BUY** at step {row['step']} — ${row['close']:.2f}")
        elif row["action"] == "Sell":
            st.markdown(f"🔴 **SELL** at step {row['step']} — ${row['close']:.2f}")

    st.success("✅ Demo complete. Chart and trades displayed.")






