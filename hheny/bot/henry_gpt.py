import openai
import streamlit as st

openai.api_key = st.secrets["openai"]["api_key"]

def get_gpt_suggestion():
    try:
        prompt = """
        You are a crypto trading assistant.
        Suggest:
        1. A crypto symbol (like BTC/USD or ETH/USD)
        2. A timeframe (e.g. 5m, 15m)
        3. A strategy (e.g. trend-follow, breakout, scalp)

        Respond only in JSON format like:
        {
          "symbol": "BTC/USD",
          "timeframe": "15m",
          "strategy": "scalp"
        }
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        return eval(content)

    except Exception as e:
        st.warning(f"GPT fallback: {e}")
        return {"symbol": "BTC/USD", "timeframe": "5m", "strategy": "trend-follow"}

def explain_strategy(symbol, timeframe, strategy):
    prompt = f"""
    Write a 3-line summary explaining the trading strategy for:
    Symbol: {symbol}
    Timeframe: {timeframe}
    Strategy: {strategy}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Strategy: {strategy} ({symbol}, {timeframe})"

def summarize_results(actions, pnl):
    buys = sum(1 for a in actions if a == "Buy")
    sells = sum(1 for a in actions if a == "Sell")

    prompt = f"""
    Henry made {buys} buy trades and {sells} sell trades.
    Net portfolio change: {pnl:.2f}%.

    Provide a simple 2-sentence summary of performance.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except:
        return f"Trades: {buys} buys, {sells} sells. Final return: {pnl:.2f}%"