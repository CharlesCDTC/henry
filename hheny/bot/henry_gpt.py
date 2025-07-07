# bot/henry_gpt.py

def get_gpt_suggestion():
    """
    Returns a fallback trading configuration suggestion.
    This replaces GPT logic with simple built-in rules.
    """
    return {
        "symbol": "BTC/USD",
        "timeframe": "15m",
        "strategy": "momentum"
    }

def explain_strategy(symbol, timeframe, strategy):
    """
    Returns a human-readable description of the chosen strategy.
    """
    if strategy == "momentum":
        return (
            f"📊 Strategy: Momentum\n"
            f"Trades {symbol} on the {timeframe} timeframe using RSI and MACD indicators. "
            f"The bot buys into strength and exits when momentum weakens."
        )
    elif strategy == "scalp":
        return (
            f"📊 Strategy: Scalp\n"
            f"Scalps quick price movements for {symbol} on the {timeframe} chart. "
            f"Designed for small, fast trades with tight risk control."
        )
    elif strategy == "breakout":
        return (
            f"📊 Strategy: Breakout\n"
            f"Monitors {symbol} on the {timeframe} chart and enters trades when the price "
            f"breaks above or below established support/resistance zones."
        )
    else:
        return (
            f"📊 Strategy: {strategy.title()}\n"
            f"Trading strategy for {symbol} on the {timeframe} chart."
        )

def summarize_results(actions, pnl):
    """
    Summarizes trading actions and profit/loss.
    """
    buys = sum(1 for a in actions if a == "Buy")
    sells = sum(1 for a in actions if a == "Sell")
    
    return (
        f"📈 Summary:\n"
        f"Total Buy Trades: {buys}\n"
        f"Total Sell Trades: {sells}\n"
        f"Final Portfolio Change: {pnl:.2f}%"
    )


