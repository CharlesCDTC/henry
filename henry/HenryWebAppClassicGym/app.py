import streamlit as st
from bot.henry_bot import run_henry_bot

st.title("🤖 Henry the Trading Bot (DEMO)")

# Demo credentials
api_key = "demo"
secret = "demo"
email = ""
live_trading = False

if st.button("▶️ Run Henry Bot"):
    run_henry_bot(api_key, secret, email, live_trading)
