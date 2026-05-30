"""
app.py – Professional AI Day‑Trading Dashboard (Persistent Dynamic Scanner)
- Scanner runs once at startup, saved in st.session_state.scanned_tickers.
- Re‑Scan button clears stored list and reruns scanner.
- Manual tickers can be added separately and persist across reruns.
- 1‑Minute candlestick charts with correct IST timestamps.
- Live order audit log, portfolio, equity popover.
"""

import streamlit as st
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from datetime import datetime, timedelta
import pytz
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from stable_baselines3 import PPO
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import time

# ----------------------------------------------------------------------
# Page config & Premium CSS
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .ticker-badge {
        display: inline-block;
        background-color: #0f1117;
        color: #00ff88;
        border: 2px solid #00ff88;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 2px 5px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .ticker-badge:hover {
        background-color: #00ff8810;
        box-shadow: 0 0 8px #00ff8850;
    }
    .signal-card {
        padding: 16px 20px;
        border-radius: 16px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-left: 6px solid;
        transition: transform 0.2s;
    }
    .signal-card:hover { transform: translateY(-2px); }
    .signal-buy {
        background: linear-gradient(135deg, #e6ffe6 0%, #b3f0b3 100%);
        color: #0a2e0a;
        border-left-color: #00cc66;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ffe6e6 0%, #f5b3b3 100%);
        color: #2e0a0a;
        border-left-color: #cc3333;
    }
    .signal-hold {
        background: linear-gradient(135deg, #fff7e6 0%, #ffd699 100%);
        color: #2e2a0a;
        border-left-color: #cc9900;
    }
    .asset-header {
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .price-text {
        font-size: 24px;
        font-weight: 600;
        margin: 0 0 4px 0;
    }
    .rating-text {
        font-size: 16px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    .equity-metric {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .delta-positive { color: #28a745; font-weight: 600; }
    .delta-negative { color: #dc3545; font-weight: 600; }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #dc3545 !important;
        color: white !important;
        border: 2px solid #a71d2a !important;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .remove-btn button {
        background: transparent;
        border: none;
        color: #ff4d4d;
        font-size: 18px;
        padding: 0;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Initialize Alpaca API & model
# ----------------------------------------------------------------------
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
    BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    api.get_account()
    alpaca_connected = True
except Exception as e:
    st.error(f"Alpaca connection failed: {e}")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return PPO.load("ppo_trading_agent.zip")
    except:
        return None

model = load_model()
if model is None:
    st.error("Trained model 'ppo_trading_agent.zip' not found. Run train.py first.")
    st.stop()

# ----------------------------------------------------------------------
# Dynamic Stock Screener (cached)
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Scanning market for high‑volume stocks...")
def scan_dynamic_watchlist(limit=5):
    """
    Fetch active US equities, rank by current daily volume, return top N symbols.
    """
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        symbols = [a.symbol for a in assets if a.tradable and a.symbol.isalpha() and len(a.symbol) < 6]
    except APIError as e:
        st.warning(f"Could not fetch asset list: {e}")
        return [], []

    if not symbols:
        return [], []

    BATCH_SIZE = 200
    all_pairs = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        try:
            snaps_dict = api.get_snapshots(batch)   # dict {symbol: SnapshotV2}
            all_pairs.extend(snaps_dict.items())
        except APIError as e:
            if '429' in str(e):
                time.sleep(1)
                snaps_dict = api.get_snapshots(batch)
                all_pairs.extend(snaps_dict.items())
            else:
                st.warning(f"Snapshot error: {e}")
                continue

    volume_pairs = []
    for sym, snap in all_pairs:
        try:
            vol = snap.daily_bar.volume
        except AttributeError:
            vol = 0
        volume_pairs.append((sym, vol))

    volume_pairs.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in volume_pairs[:limit]]
    top_volumes = [v for _, v in volume_pairs[:limit]]
    return top_symbols, top_volumes

# ----------------------------------------------------------------------
# Session state – Persistent scanned tickers
# ----------------------------------------------------------------------
if "scanned_tickers" not in st.session_state:
    with st.spinner("Initial market scan..."):
        syms, vols = scan_dynamic_watchlist(limit=5)
        st.session_state.scanned_tickers = syms
        st.session_state.scanned_volumes = vols

if "manual_tickers" not in st.session_state:
    st.session_state.manual_tickers = []

# Build the final monitored list (scanned + manual, unique)
monitored = list(dict.fromkeys(st.session_state.scanned_tickers + st.session_state.manual_tickers))
st.session_state.monitored_tickers = monitored

# Initialize other states
if "per_ticker_enabled" not in st.session_state:
    st.session_state.per_ticker_enabled = {}
if "auto_trade" not in st.session_state:
    st.session_state.auto_trade = False
if "equity_history" not in st.session_state:
    st.session_state.equity_history = []
if "prev_equity" not in st.session_state:
    st.session_state.prev_equity = None
if "execution_log" not in st.session_state:
    st.session_state.execution_log = []
if "custom_equity" not in st.session_state:
    st.session_state.custom_equity = None

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_tradeable(symbol):
    try:
        asset = api.get_asset(symbol)
        return asset.tradable
    except APIError:
        return False

@st.cache_data(ttl=60, show_spinner=False)
def get_historical_data(symbol, days=5, timeframe="1Min"):
    try:
        utc = pytz.UTC
        end = datetime.now(utc)
        is_crypto = "/" in symbol
        if not is_crypto:
            end = end - timedelta(minutes=20)

        start = end - timedelta(days=days)
        bars_symbol = symbol.replace("/", "") if is_crypto else symbol

        if is_crypto:
            bars = api.get_crypto_bars(bars_symbol, timeframe,
                                       start=start.isoformat(),
                                       end=end.isoformat()).df
        else:
            bars = api.get_bars(bars_symbol, timeframe,
                                start=start.isoformat(),
                                end=end.isoformat()).df

        if bars.empty:
            return None

        rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        bars.rename(columns=rename_map, inplace=True)
        for col in ['open','high','low','close','volume']:
            if col not in bars.columns:
                return None
        bars = bars[['open','high','low','close','volume']].copy()

        bars['rsi'] = RSIIndicator(close=bars['close'], window=14).rsi()
        macd_ind = MACD(close=bars['close'], window_slow=26, window_fast=12, window_sign=9)
        bars['macd'] = macd_ind.macd()
        atr_ind = AverageTrueRange(high=bars['high'], low=bars['low'], close=bars['close'], window=14)
        bars['atr'] = atr_ind.average_true_range()

        bars.dropna(inplace=True)
        return bars
    except APIError as e:
        st.warning(f"Alpaca API error for {symbol}: {e}")
        return None
    except Exception:
        return None

def get_account_equity():
    try:
        return float(api.get_account().equity)
    except APIError as e:
        st.warning(f"Could not fetch equity: {e}")
        return None

def get_all_positions():
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
    except APIError as e:
        st.warning(f"Could not fetch positions: {e}")
        return []

def get_position_for(symbol):
    try:
        p = api.get_position(symbol)
        return {"qty": float(p.qty), "avg_price": float(p.avg_entry_price)}
    except:
        return None

def process_signal_and_trade(ticker, bars, capital, do_trade=False):
    if bars is None or bars.empty:
        return {"signal": None, "price": None, "action": None, "order_msg": "No data"}

    latest = bars.iloc[-1]
    price = float(latest['close'])
    rsi = float(latest['rsi'])
    macd_val = float(latest['macd'])
    atr = float(latest['atr'])

    pos = get_position_for(ticker)
    shares_held = pos['qty'] if pos else 0.0
    cost_basis = pos['avg_price'] if pos else 0.0
    net_worth = capital + shares_held * price

    obs = np.array([capital, shares_held, cost_basis, net_worth, rsi, macd_val, atr], dtype=np.float32)
    action_arr, _ = model.predict(obs, deterministic=True)
    action = int(action_arr.item())

    signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    signal = signal_map[action]

    order_msg = ""
    if do_trade and action != 0:
        is_crypto = "/" in ticker
        try:
            if action == 1:   # BUY
                if pos is None:
                    qty = capital / price
                    api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='gtc' if is_crypto else 'day'
                    )
                    order_msg = f"Bought {qty:.4f} shares"
                else:
                    order_msg = "Already holding"
            elif action == 2:  # SELL
                if pos is not None:
                    qty = pos['qty']
                    api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc' if is_crypto else 'day'
                    )
                    order_msg = f"Sold {qty} shares"
                else:
                    order_msg = "No shares to sell"
        except APIError as e:
            order_msg = f"Trade error: {e}"

    return {"signal": signal, "price": price, "action": action, "order_msg": order_msg}

# ----------------------------------------------------------------------
# Sidebar – Persistent Scanner UI
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 AI Day‑Trading Cockpit")

    # Scanner section
    st.subheader("🔍 AI Market Scanner")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Today's Top Volume (scanned)**")
    with col2:
        if st.button("🔄 Re‑Scan", help="Clear cached scan and run fresh"):
            if "scanned_tickers" in st.session_state:
                del st.session_state.scanned_tickers
            if "scanned_volumes" in st.session_state:
                del st.session_state.scanned_volumes
            st.cache_data.clear()
            st.rerun()

    if st.session_state.scanned_tickers:
        for sym, vol in zip(st.session_state.scanned_tickers, st.session_state.scanned_volumes):
            st.markdown(f"🟢 {sym} – Vol: {vol:,.0f}")
    else:
        st.info("No scanned tickers yet.")

    st.markdown("---")

    # Manual add
    st.subheader("➕ Manual Add")
    new_ticker = st.text_input("Symbol (e.g., AAPL)")
    if st.button("Add"):
        clean = new_ticker.strip().upper()
        if clean and clean not in st.session_state.manual_tickers and is_tradeable(clean):
            st.session_state.manual_tickers.append(clean)
            st.session_state.per_ticker_enabled[clean] = False
            st.success(f"Added {clean}")
            st.rerun()
        elif clean:
            st.info("Already in list or invalid.")

    st.subheader("📊 Active Tickers")
    monitored = list(dict.fromkeys(st.session_state.scanned_tickers + st.session_state.manual_tickers))
    if monitored:
        for ticker in monitored:
            col_badge, col_toggle, col_remove = st.columns([4, 2, 1])
            badge_html = f'<span class="ticker-badge">{ticker}</span>'
            col_badge.markdown(badge_html, unsafe_allow_html=True)

            auto_enabled = st.session_state.per_ticker_enabled.get(ticker, False)
            new_auto = col_toggle.checkbox("Auto", value=auto_enabled, key=f"auto_{ticker}")
            if new_auto != auto_enabled:
                st.session_state.per_ticker_enabled[ticker] = new_auto
                st.rerun()

            if col_remove.button("❌", key=f"remove_{ticker}"):
                if ticker in st.session_state.manual_tickers:
                    st.session_state.manual_tickers.remove(ticker)
                    if ticker in st.session_state.per_ticker_enabled:
                        del st.session_state.per_ticker_enabled[ticker]
                    st.rerun()
    else:
        st.info("No tickers. Click 'Re‑Scan' or add manually.")

    st.markdown("---")
    capital_per_trade = st.number_input("Capital per Trade ($)", min_value=10, value=1000, step=100)

    auto_trade = st.toggle("Enable Autonomous Execution", value=st.session_state.auto_trade)
    st.session_state.auto_trade = auto_trade

    if auto_trade:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[60, 300, 600, 1800, 3600],
            index=1,
            format_func=lambda x: f"{x//60} min" if x < 3600 else f"{x//3600} hr"
        )
        st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")
    else:
        st_autorefresh(interval=86400000, key="manual")

    st.markdown("---")
    if st.button("🛑 EMERGENCY STOP", type="primary", use_container_width=True):
        st.session_state.auto_trade = False
        st.session_state.per_ticker_enabled = {t: False for t in st.session_state.monitored_tickers}
        st.success("All trading halted.")

# ----------------------------------------------------------------------
# Main Dashboard
# ----------------------------------------------------------------------
st.title("🚀 Professional AI Day‑Trading Dashboard")

if not st.session_state.monitored_tickers:
    st.info("👈 Scan for stocks or add a ticker to begin.")
else:
    st.subheader("📡 Live Signals & 1‑Minute Price Charts (IST)")
    for ticker in st.session_state.monitored_tickers:
        if not is_tradeable(ticker):
            st.error(f"**{ticker}** – Not tradable on Alpaca")
            continue

        bars = get_historical_data(ticker, days=5, timeframe="1Min")
        if bars is None:
            st.warning(f"**{ticker}** – No 1‑min data available (market closed or unsupported asset)")
            continue

        do_trade = (
            st.session_state.auto_trade and
            st.session_state.per_ticker_enabled.get(ticker, False)
        )
        result = process_signal_and_trade(ticker, bars, capital_per_trade, do_trade)
        signal = result["signal"]
        price = result["price"]
        order_msg = result["order_msg"]

        if signal and price:
            css_class = {"HOLD": "signal-hold", "BUY": "signal-buy", "SELL": "signal-sell"}[signal]
            rating = {"BUY": "STRONG BUY", "SELL": "STRONG SELL", "HOLD": "HOLD"}[signal]
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}[signal]

            card_html = f"""
            <div class="signal-card {css_class}">
                <div class="asset-header">{emoji} {ticker}</div>
                <div class="price-text">${price:,.2f}</div>
                <div class="rating-text">AI Model Rating: {rating}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            if order_msg:
                if "error" in order_msg.lower():
                    st.error(order_msg)
                elif order_msg not in ["Already holding", "No shares to sell"]:
                    st.success(order_msg)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.execution_log.append(f"{timestamp} | {ticker}: {signal} -> {order_msg}")
                else:
                    st.info(order_msg)
        else:
            st.warning(f"**{ticker}** – Unable to generate signal")

        # ---------- CANDLESTICK CHART (IST) ----------
        if bars is not None and len(bars) >= 2:
            chart_data = bars.copy()
            chart_data.columns = [col.lower() for col in chart_data.columns]

            if isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data['timestamp'] = chart_data.index.tz_convert('Asia/Kolkata')
            else:
                st.error("Unexpected index type – cannot plot dates.")
                continue

            required_ohlc = ['open', 'high', 'low', 'close']
            if all(col in chart_data.columns for col in required_ohlc):
                fig = go.Figure(data=[go.Candlestick(
                    x=chart_data['timestamp'],
                    open=chart_data['open'],
                    high=chart_data['high'],
                    low=chart_data['low'],
                    close=chart_data['close'],
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350',
                )])
                fig.update_layout(
                    title=f"{ticker} – 1‑Minute Chart (5 Days, IST)",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_rangeslider_visible=False,
                    dragmode='pan',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("OHLC data missing for candlestick chart.")
        else:
            st.caption("Not enough data for candlestick chart (need at least 2 bars).")

# ------------------------------
# Account Equity (persistent popover)
# ------------------------------
st.subheader("📈 Account Equity")
real_equity = get_account_equity()
display_equity = st.session_state.custom_equity if st.session_state.custom_equity is not None else (real_equity or 100000.0)

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown(f"""
    <div class="equity-metric">
        <div>
            <div style="font-size:18px; color:#333; font-weight:600;">Current Equity</div>
            <div style="font-size:28px; font-weight:700; color:#000;">${display_equity:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    with st.popover("Change Balance"):
        new_val = st.number_input(
            "Enter custom capital ($)",
            min_value=0.0,
            value=float(display_equity),
            step=1000.0,
            format="%.2f"
        )
        if st.button("Save", use_container_width=True):
            st.session_state.custom_equity = new_val
            st.session_state.equity_history = [{"Time": datetime.now().strftime("%H:%M:%S"), "Equity": new_val}]
            st.session_state.prev_equity = None
            st.rerun()

# Delta display
prev_equity = st.session_state.prev_equity
delta_str = ""
delta_class = ""
if prev_equity is not None and prev_equity != 0 and display_equity != prev_equity:
    change = display_equity - prev_equity
    pct = (change / prev_equity) * 100
    if change > 0:
        delta_str = f"▲ ${change:,.2f} ({pct:+.2f}%)"
        delta_class = "delta-positive"
    elif change < 0:
        delta_str = f"▼ ${-change:,.2f} ({pct:+.2f}%)"
        delta_class = "delta-negative"

if delta_str:
    st.markdown(f"""
    <div class="equity-metric">
        <div style="text-align:right; width:100%;">
            <div style="font-size:18px; color:#333; font-weight:600;">Change</div>
            <div style="font-size:24px;"><span class="{delta_class}">{delta_str}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Update equity history
now = datetime.now().strftime("%H:%M:%S")
if not st.session_state.equity_history or st.session_state.equity_history[-1]["Equity"] != display_equity:
    st.session_state.equity_history.append({"Time": now, "Equity": display_equity})
    if len(st.session_state.equity_history) > 100:
        st.session_state.equity_history.pop(0)
st.session_state.prev_equity = display_equity

# Equity line chart
if st.session_state.equity_history:
    df_equity = pd.DataFrame(st.session_state.equity_history)
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=df_equity["Time"],
        y=df_equity["Equity"],
        mode='lines',
        line=dict(color='#4fc3f7', width=2),
        fill='tozeroy',
        fillcolor='rgba(79, 195, 247, 0.1)',
        name='Account Equity'
    ))
    fig_equity.update_layout(
        title="Account Equity Over Time",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    st.plotly_chart(fig_equity, use_container_width=True)
else:
    st.info("Waiting for first equity snapshot...")

# ------------------------------
# Portfolio Holdings
# ------------------------------
st.subheader("💼 Current Holdings")
positions = get_all_positions()
if positions:
    df_pos = pd.DataFrame(positions)
    def color_pl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    st.dataframe(
        df_pos.style.format({
            "Avg Entry Price": "${:,.2f}",
            "Market Value": "${:,.2f}",
            "Unrealized P&L": "${:,.2f}"
        }).map(color_pl, subset=['Unrealized P&L']),
        use_container_width=True
    )
else:
    st.info("No open positions.")

# ------------------------------
# Live Order Audit Log
# ------------------------------
st.subheader("📜 Live Order Audit Log")
try:
    orders = api.list_orders(status='all', limit=20)
    if orders:
        orders_data = []
        for o in orders:
            ts = o.submitted_at
            if ts:
                ist_ts = pd.to_datetime(ts).tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')
            else:
                ist_ts = ""
            orders_data.append({
                "Timestamp (IST)": ist_ts,
                "Symbol": o.symbol,
                "Side": o.side.capitalize(),
                "Qty": o.qty,
                "Type": o.type.capitalize(),
                "Status": o.status.capitalize()
            })
        df_orders = pd.DataFrame(orders_data)
        st.dataframe(df_orders, use_container_width=True)
    else:
        st.info("No recent orders found.")
except APIError as e:
    st.warning(f"Could not fetch order audit: {e}")
except Exception as e:
    st.warning(f"Unexpected error fetching orders: {e}")

# ------------------------------
# Execution Log
# ------------------------------
with st.expander("📋 Execution Log (last 20 entries)"):
    if st.session_state.execution_log:
        st.text("\n".join(st.session_state.execution_log[-20:]))
    else:
        st.text("No orders yet.")