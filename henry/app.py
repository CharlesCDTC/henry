import streamlit as st
from bot.henry_bot import run_henry_bot
from bot.henry_gpt import get_gpt_suggestion, explain_strategy

st.title("🤖 Henry the Autonomous Trading Bot")

use_gpt = st.checkbox("Let ChatGPT choose what to trade", value=True)

if use_gpt:
    gpt_info = get_gpt_suggestion()
    symbol = gpt_info["symbol"]
    timeframe = gpt_info["timeframe"]
    strategy = gpt_info["strategy"]
    st.info(f"🤖 GPT chose: {symbol}, {timeframe}, strategy: {strategy}")
else:
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    timeframes = ["1m", "5m", "15m", "1h"]
    symbol = st.selectbox("Select Symbol", symbols)
    timeframe = st.selectbox("Select Timeframe", timeframes)
    strategy = "manual"

live_trading = st.checkbox("Enable Live Trading")
api_key = st.secrets["api"]["coinbase_api"] if live_trading else None
secret = st.secrets["api"]["coinbase_secret"] if live_trading else None

if st.button("▶️ Run Henry"):
    run_henry_bot(symbol, timeframe, live_trading, api_key, secret, strategy)