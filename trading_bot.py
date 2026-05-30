"""
trading_bot.py – Account info, positions, and trade execution helpers.
"""

import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime

def get_account_equity(api):
    """Return current account equity (cash + market value)."""
    try:
        return float(api.get_account().equity)
    except Exception:
        return None

def get_all_positions(api):
    """Return a list of open positions as dicts with Symbol, Qty, Avg Entry Price, etc."""
    try:
        positions = api.list_positions()
        return [
            {
                "Symbol": p.symbol,
                "Qty": float(p.qty),
                "Avg Entry Price": float(p.avg_entry_price),
                "Market Value": float(p.market_value),
                "Unrealized P&L": float(p.unrealized_pl),
            }
            for p in positions
        ]
    except Exception:
        return []

def get_position_for(api, symbol: str):
    """Return dict {'qty': qty, 'avg_price': avg_entry_price} or None if no position."""
    try:
        p = api.get_position(symbol)
        return {"qty": float(p.qty), "avg_price": float(p.avg_entry_price)}
    except tradeapi.rest.APIError:
        return None

def process_ticker_signal(api, model, ticker: str, bars_df, capital_per_trade: float):
    """
    Given a DataFrame of historical bars (including latest row with indicators),
    build the observation, predict action, and execute trade if auto‑trade is on.
    Returns a dict with signal info and optional order message.
    The actual order submission is done here if auto‑trade is enabled.
    """
    if bars_df is None or bars_df.empty:
        return {"signal": None, "price": None, "order_msg": "No data"}

    latest = bars_df.iloc[-1]
    price = float(latest['close'])
    rsi = float(latest['rsi'])
    macd_val = float(latest['macd'])
    atr = float(latest['atr'])

    # Get current position (if any)
    pos = get_position_for(api, ticker)
    shares_held = pos['qty'] if pos else 0.0
    cost_basis = pos['avg_price'] if pos else 0.0
    balance = capital_per_trade
    net_worth = balance + shares_held * price

    obs = np.array([balance, shares_held, cost_basis, net_worth, rsi, macd_val, atr], dtype=np.float32)
    action_arr, _ = model.predict(obs, deterministic=True)
    action = int(action_arr.item())

    signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    signal = signal_map[action]

    # --- Order execution ---
    order_msg = ""
    is_crypto = "/" in ticker

    if action == 1:  # BUY
        if pos is None:
            qty = capital_per_trade / price
            try:
                api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc' if is_crypto else 'day'
                )
                order_msg = f"Bought {qty:.4f} shares"
            except Exception as e:
                order_msg = f"Buy error: {e}"
        else:
            order_msg = "Already holding"
    elif action == 2:  # SELL
        if pos is not None:
            qty = pos['qty']
            try:
                api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc' if is_crypto else 'day'
                )
                order_msg = f"Sold {qty} shares"
            except Exception as e:
                order_msg = f"Sell error: {e}"
        else:
            order_msg = "No shares to sell"

    return {
        "signal": signal,
        "price": price,
        "order_msg": order_msg,
        "action": action,
    }