import streamlit as st
from bot.henry_bot import run_henry_bot

st.set_page_config(page_title="Henry Trading Bot", layout="wide")
st.title("🤖 Henry the Autonomous Trading Bot")

with st.sidebar:
    st.header("Bot Settings")
    use_secrets = st.checkbox("Use Streamlit Secrets (for cloud deploy)", value=True)

    if use_secrets:
        api_key = st.secrets["api"]["coinbase_api"]
        secret = st.secrets["api"]["coinbase_secret"]
        email = st.secrets["api"]["email_alert"]
    else:
        api_key = st.text_input("Coinbase API Key", type="password")
        secret = st.text_input("Coinbase Secret", type="password")
        email = st.text_input("Email for Alerts")

    live_trading = st.checkbox("Enable Live Trading", value=False)
    run = st.button("Run Henry")

if run:
    if not api_key or not secret:
        st.warning("API Key and Secret are required.")
    else:
        st.success("Bot is running...")
        run_henry_bot(api_key, secret, email, live_trading)
