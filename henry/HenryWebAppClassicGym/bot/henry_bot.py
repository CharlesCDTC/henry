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
    try:
        df = fetch_data(api_key, secret)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return

    env = DummyVecEnv([lambda: HenryTradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)
    obs = env.reset()
    done = False
    portfolio = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        price = df.iloc[env.envs[0].current_step]['close']
        value = env.envs[0].usd_balance + env.envs[0].crypto_held * price
        portfolio.append(value)
        if live_trading and action in [1, 2]:
            send_email_alert(f"Henry Trade Action {action}", f"Executed action {action} at {price}", email)

    st.line_chart(portfolio)
    st.success(f"Run complete. Final portfolio value: ${portfolio[-1]:.2f}")
