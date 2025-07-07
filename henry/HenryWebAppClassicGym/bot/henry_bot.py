import pandas as pd
import numpy as np
import streamlit as st
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from stable_baselines3 import PPO
from gym import Env, spaces
from gym.wrappers import TimeLimit


def fetch_demo_data():
    np.random.seed(42)
    steps = 150
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
    return df.dropna().reset_index(drop=True)


def fetch_real_data(symbol="BTC/USD", timeframe="5m", limit=150):
    exchange = ccxt.coinbase()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df.dropna().reset_index(drop=True)


class HenryEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = 1000.0
        self.starting_balance = 1000.0
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row["close"], row["rsi"], row["macd"],
            row["macd_signal"], self.crypto_held, self.usd_balance
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["close"]

        if action == 1 and self.usd_balance >= price:
            self.crypto_held += 1
            self.usd_balance -= price
        elif action == 2 and self.crypto_held >= 1:
            self.crypto_held -= 1
            self.usd_balance += price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_obs = self._next_observation()
        portfolio_value = self.usd_balance + self.crypto_held * price
        reward = (portfolio_value - self.starting_balance) / self.starting_balance
        return next_obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = self.starting_balance
        return self._next_observation(), {}


def run_henry_bot(api_key, secret, email, live_trading):
    st.write("✅ Henry bot is starting...")

    # Load data
    try:
        if api_key == "demo" or secret == "demo":
            df = fetch_demo_data()
            st.info("📊 Running on demo data")
        else:
            df = fetch_real_data()
            st.success("📡 Fetched real Coinbase data")
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")
        return

    if len(df) < 30:
        st.warning("Not enough data to run PPO model.")
        return

    # Train PPO model
    env = HenryEnv(df)
    env = TimeLimit(env, max_episode_steps=100)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)

    # Simulate trading
    obs, _ = env.reset()
    df["action"] = ""
    portfolio = []

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        step = env.current_step
        if step >= len(df):
            break
        df.loc[step, "action"] = "Buy" if action == 1 else "Sell" if action == 2 else ""
        portfolio.append(env.usd_balance + env.crypto_held * df.loc[step, "close"])
        if done:
            break

    df["step"] = range(len(df))

    # Chart
    st.subheader("📈 Portfolio Value Over Time")
    st.line_chart(portfolio)

    st.subheader("📊 Price Chart with Actions")
    st.line_chart(df.set_index("step")[["close"]])

    # Trade log
    st.subheader("📜 Trade Timeline")
    for _, row in df[df["action"] != ""].iterrows():
        if row["action"] == "Buy":
            st.markdown(f"🟢 **BUY** at step {row['step']} — ${row['close']:.2f}")
        elif row["action"] == "Sell":
            st.markdown(f"🔴 **SELL** at step {row['step']} — ${row['close']:.2f}")

    st.success("✅ Simulation complete.")






