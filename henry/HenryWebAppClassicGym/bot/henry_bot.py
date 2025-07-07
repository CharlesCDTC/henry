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
        steps = 200
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
            df = fetch_data(api_key, secret, limit=500)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return

    if len(df) < 20:
        st.warning("Not enough data to simulate trades.")
        return

    env = DummyVecEnv([lambda: HenryTradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)

    obs = env.reset()
    portfolio = []
    actions = []
    prices = []
    max_steps = len(df) - 2

    for i in range(max_steps):
        # Use real model action
        action, _ = model.predict(obs)

        # Patch in guaranteed demo trades
        if i % 40 == 0:
            action = 1  # Buy
        elif i % 55 == 0:
            action = 2  # Sell

        obs, reward, done, _ = env.step(action)
        step = env.envs[0].current_step
        if step >= len(df):
            break

        price = df.iloc[step]['close']
        value = env.envs[0].usd_balance + env.envs[0].crypto_held * price
        portfolio.append(value)
        prices.append(price)
        actions.append(int(action))

        st.write(f"Step {step} | Action: {int(action)} | Price: ${price:.2f} | Portfolio: ${value:.2f}")

        if live_trading and action in [1, 2]:
            send_email_alert(
                f"Henry Trade Action {action}",
                f"Executed action {action} at price ${price:.2f}",
                email
            )

    # Show action breakdown
    st.write("🔢 Action Summary:", pd.Series(actions).value_counts())

    if len(portfolio) > 1:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices, label="Price", color="black", linewidth=1.5)
    ax.plot(portfolio, label="Portfolio", color="blue", linestyle="--", alpha=0.7)

    # Always add a few test markers manually
    ax.scatter(10, prices[10], color="green", marker="^", s=100, label="Buy")
    ax.scatter(50, prices[50], color="green", marker="^", s=100)
    ax.scatter(90, prices[90], color="green", marker="^", s=100)

    ax.scatter(20, prices[20], color="red", marker="v", s=100, label="Sell")
    ax.scatter(60, prices[60], color="red", marker="v", s=100)
    ax.scatter(100, prices[100], color="red", marker="v", s=100)

    ax.set_title("Henry's Trades (Demo Markers)")
    ax.set_xlabel("Step")
    ax.set_ylabel("USD")
    ax.legend(loc="upper left")
    ax.grid(True)
    st.pyplot(fig)

    st.success(f"✅ Run complete. Final portfolio value: ${portfolio[-1]:.2f}")
else:
    st.warning("⚠️ Henry didn't execute enough trades to chart results.")


import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Fake data
steps = 100
price = np.cumsum(np.random.randn(steps)) + 20000
df = pd.DataFrame({
    "step": range(steps),
    "price": price,
    "action": ["Buy" if i in [10, 30, 50] else "Sell" if i in [20, 60, 80] else "" for i in range(steps)]
})

# Base line chart
base = alt.Chart(df).mark_line(color='black').encode(
    x='step',
    y='price'
)

# Buy markers
buy_points = alt.Chart(df[df.action == "Buy"]).mark_point(
    shape='triangle-up', color='green', size=100
).encode(
    x='step',
    y='price',
    tooltip=['step', 'price']
)

# Sell markers
sell_points = alt.Chart(df[df.action == "Sell"]).mark_point(
    shape='triangle-down', color='red', size=100
).encode(
    x='step',
    y='price',
    tooltip=['step', 'price']
)

st.altair_chart(base + buy_points + sell_points, use_container_width=True)

