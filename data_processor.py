"""
data_processor.py – Fetch historical bars from Alpaca and compute technical indicators.
Handles crypto (slash in symbol) and stocks (20‑min delay) correctly.
"""

import pandas as pd
from datetime import datetime, timedelta
import pytz
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

def get_historical_data(api, symbol: str, days: int = 60):
    """
    Fetch daily bars and compute RSI, MACD, ATR.
    Returns a pandas DataFrame with columns: open, high, low, close, volume, rsi, macd, atr.
    Returns None on failure.
    """
    try:
        utc = pytz.UTC
        end = datetime.now(utc)
        is_crypto = "/" in symbol

        # Stocks have a 20‑minute data delay on the free tier
        if not is_crypto:
            end = end - timedelta(minutes=20)

        start = end - timedelta(days=days)
        start_str = start.isoformat()
        end_str = end.isoformat()

        # Alpaca’s bars endpoint expects a slash‑less symbol for crypto
        bars_symbol = symbol.replace("/", "") if is_crypto else symbol

        if is_crypto:
            bars = api.get_crypto_bars(bars_symbol, "1Day", start=start_str, end=end_str).df
        else:
            bars = api.get_bars(bars_symbol, "1Day", start=start_str, end=end_str).df

        if bars.empty:
            return None

        # Rename short column names to full lowercase names
        rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        bars.rename(columns=rename_map, inplace=True)

        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in bars.columns:
                return None
        bars = bars[required].copy()

        # Compute indicators
        bars['rsi'] = RSIIndicator(close=bars['close'], window=14).rsi()
        macd_ind = MACD(close=bars['close'], window_slow=26, window_fast=12, window_sign=9)
        bars['macd'] = macd_ind.macd()
        atr_ind = AverageTrueRange(high=bars['high'], low=bars['low'], close=bars['close'], window=14)
        bars['atr'] = atr_ind.average_true_range()

        bars.dropna(inplace=True)
        bars.reset_index(drop=True, inplace=True)
        return bars

    except Exception as e:
        # Log error in production; for now just return None
        return None