import streamlit as st
from bot.henry_bot import run_henry_bot

st.set_page_config(page_title="Henry Bot", layout="centered")
st.title("🤖 Henry - Autonomous Trading Bot")

symbol = st.text_input("Symbol (e.g., BTC/USD)", "BTC/USD")
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
live_trading = st.checkbox("Enable Live Trading")
api_key = st.secrets["api"]["coinbase_api"] if live_trading else None
secret = st.secrets["api"]["coinbase_secret"] if live_trading else None

if st.button("▶️ Run Henry"):
    run_henry_bot(symbol, timeframe, live_trading, api_key, secret)
