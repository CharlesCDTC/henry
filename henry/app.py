
import streamlit as st
from bot.henry_bot import run_henry_bot

st.title("🤖 Henry the Trading Bot")

symbol = st.text_input("Symbol", value="BTC/USD")
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)
live_trading = st.checkbox("Enable live trading (real money)", value=False)

if live_trading:
    api_key = st.secrets["api"]["coinbase_api"]
    secret = st.secrets["api"]["coinbase_secret"]
else:
    api_key = None
    secret = None

strategy = st.selectbox("Strategy", ["trend-follow"], index=0)

if st.button("▶️ Run Henry"):
    run_henry_bot(symbol, timeframe, live_trading, api_key, secret, strategy)
