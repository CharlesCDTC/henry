import streamlit as st
from bot.henry_bot import run_henry_bot

st.title("🤖 Henry the Trading Bot")

symbol = st.selectbox("Select Symbol", ["BTC/USD", "ETH/USD", "LTC/USD"])
timeframe = st.selectbox("Select Timeframe", ["1m", "5m", "15m"])
live_trading = st.checkbox("Enable Live Trading")

if live_trading:
    api_key = st.secrets["api"]["coinbase_api"]
    secret = st.secrets["api"]["coinbase_secret"]
else:
    api_key = None
    secret = None

strategy = st.selectbox("Select Strategy", ["trend-follow", "mean-reversion"])

if st.button("▶️ Run Henry"):
    run_henry_bot(symbol, timeframe, live_trading, api_key, secret, strategy)
