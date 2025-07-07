import streamlit as st
from bot.henry_bot import run_henry_bot

st.title("🤖 Henry the Autonomous Trading Bot")

symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
timeframes = ["1m", "5m", "15m", "1h"]

symbol = st.selectbox("Select Crypto Pair", symbols)
timeframe = st.selectbox("Select Timeframe", timeframes)

live_trading = st.checkbox("Enable Live Trading (Careful!)")
api_key = st.secrets["api"]["coinbase_api"] if live_trading else None
secret = st.secrets["api"]["coinbase_secret"] if live_trading else None

if st.button("▶️ Run Henry Bot"):
    run_henry_bot(symbol, timeframe, live_trading, api_key, secret)
