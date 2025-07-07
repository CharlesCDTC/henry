import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
from gymnasium import Env, spaces
from ta.momentum import RSIIndicator
from ta.trend import MACD
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

class HenryEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = 1000.0
        self.starting_balance = 1000.0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row["close"],
            row["rsi"],
            row["macd"],
            row["macd_signal"],
            self.crypto_held,
            self.usd_balance
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
        obs = self._next_observation()
        portfolio_value = self.usd_balance + self.crypto_held * price
        reward = (portfolio_value - self.starting_balance) / self.starting_balance
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.crypto_held = 0.0
        self.usd_balance = self.starting_balance
        return self._next_observation(), {}

def fetch_data(symbol="BTC/USD", timeframe="5m", limit=200):
    exchange = ccxt.coinbase()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df.dropna().reset_index(drop=True)

def run_henry_bot(symbol, timeframe, live_trading, api_key, secret, strategy):
    df = fetch_data(symbol, timeframe)
    env = HenryEnv(df)
    env = TimeLimit(env, max_episode_steps=100)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)

    obs, _ = env.reset()
    portfolio = []

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        step = env.unwrapped.current_step
        if step >= len(df):
            break
        usd = env.unwrapped.usd_balance
        held = env.unwrapped.crypto_held
        price = df.loc[step, "close"]
        portfolio.append(usd + held * price)
        if done:
            break
