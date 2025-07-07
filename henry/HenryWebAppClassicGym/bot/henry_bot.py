import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import smtplib
from email.message import EmailMessage
import streamlit as st

class HenryTradingEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = 1000.0
        self.starting_balance = 1000.0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = self.starting_balance
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['open'], row['high'], row['low'], row['close'], row['volume'],
            row['rsi'], row['macd'], row['macd_signal'], row['ema'],
            row['bb_high'], row['bb_low'], self.crypto_held, self.usd_balance
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row['close']
        if action == 1 and self.usd_balance >= price:
            self.crypto_held += 1
            self.usd_balance -= price
        elif action == 2 and self.crypto_held >= 1:
            self.crypto_held -= 1
            self.usd_balance += price
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        portfolio_value = self.usd_balance + self.crypto_held * price
        reward = (portfolio_value - self.starting_balance) / self.starting_balance
        return obs, reward, terminated, False, {}

def fetch_data(api_key, secret, symbol="BTC/USD", timeframe="5m", limit=100):
    exchange = ccxt.coinbase({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["ema"] = EMAIndicator(close=df["close"], window=14).ema_indicator()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    return df.dropna().reset_index(drop=True)

def send_email_alert(subject, body, to_email):
    if not to_email:
        return
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = "yourbot@gmail.com"
        msg['To'] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("yourbot@gmail.com", "your_app_password")
            smtp.send_message(msg)
        st.success("📧 Email alert sent.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        
def run_henry_bot(api_key, secret, email, live_trading):
    def fetch_demo_data():
        np.random.seed(42)
        steps = 120
        prices = np.cumsum(np.random.randn(steps) * 20 + 20000)
        df = pd.DataFrame({
            "step": range(steps),
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

    try:
        if api_key == "demo" or secret == "demo":
            df = fetch_demo_data()
            st.info("🔧 Running in DEMO mode with synthetic data.")
        else:
            df = fetch_data(api_key, secret, limit=200)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return

    if len(df) < 20:
        st.warning("Not enough data to simulate trades.")
        return

  # Inject demo trade signals
df["action"] = ""
df.loc[10, "action"] = "Buy"
df.loc[30, "action"] = "Buy"
df.loc[50, "action"] = "Buy"
df.loc[20, "action"] = "Sell"
df.loc[60, "action"] = "Sell"
df.loc[100, "action"] = "Sell"

# ✅ Fix: ensure 'step' column exists
df["step"] = range(len(df))

# Show chart with price only (no markers inline)
st.subheader("📈 Henry's Demo Price Chart")
st.line_chart(df.set_index("step")[["close"]])


    # Trade summary
    st.subheader("📜 Trade Timeline")
    for _, row in df[df["action"] != ""].iterrows():
        if row["action"] == "Buy":
            st.markdown(f"🟢 **BUY** at step {row['step']} - ${row['close']:.2f}")
        elif row["action"] == "Sell":
            st.markdown(f"🔴 **SELL** at step {row['step']} - ${row['close']:.2f}")

    st.success("✅ Demo complete. Chart and trades rendered.")





