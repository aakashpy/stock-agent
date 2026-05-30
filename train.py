"""
train.py – GPU-accelerated PPO training for Risk-First Stock Trading Bot.
Downloads data with yfinance, computes indicators, trains on GPU (if available).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import SmartTradingEnv

# -------------------------------
# 1. Download data
# -------------------------------
TICKER = "AAPL"          # Change to "RELIANCE.NS" if needed
PERIOD = "10y"           # 10 years of daily data
INTERVAL = "1d"

print(f"Downloading {TICKER} data...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)

# Flatten multi-level columns if any
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0].lower() if col[1] == '' else col[0].lower() for col in df.columns]
else:
    df.columns = [c.lower() for c in df.columns]

# Keep only required price columns
df = df[['open', 'high', 'low', 'close', 'volume']].copy()

# Ensure index is DatetimeIndex
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# -------------------------------
# 2. Add technical indicators
# -------------------------------
print("Computing indicators (RSI, MACD, ATR)...")
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
df['macd'] = macd_indicator.macd()
atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
df['atr'] = atr_indicator.average_true_range()

# Preserve date column before dropping NaN
df['date'] = df.index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Data shape after cleaning: {df.shape}")

# -------------------------------
# 3. Chronological train/test split (80%-20%)
# -------------------------------
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# -------------------------------
# 4. Create environments
# -------------------------------
def make_env(dataframe):
    return SmartTradingEnv(
        df=dataframe,
        initial_balance=10000.0,
        slippage_tolerance=0.0005,       # 0.05%
        transaction_fee_rate=0.001,      # 0.1%
        timestamp_lag_threshold=1.0,
    )

train_env = DummyVecEnv([lambda: make_env(train_df)])
test_env = DummyVecEnv([lambda: make_env(test_df)])

# -------------------------------
# 5. Device selection for GPU/CPU
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "="*50)
print(f"🚀 Training is running on: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*50 + "\n")

# -------------------------------
# 6. Initialize PPO model
# -------------------------------
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./ppo_trading_tensorboard/",
    device=DEVICE,          # <-- Explicit device assignment
)

# -------------------------------
# 7. Train the agent
# -------------------------------
TOTAL_TIMESTEPS = 100_000   # increased for GPU training
print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("ppo_trading_agent")
print("Model saved as ppo_trading_agent.zip")

# -------------------------------
# 8. Evaluate on test set
# -------------------------------
print("\nEvaluating on test data...")
obs = test_env.reset()
done = False
total_reward = 0
step = 0
final_net_worth = None
max_drawdown = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward[0]
    net_worth = info[0].get('net_worth', 0)
    drawdown = info[0].get('drawdown', 0)
    if drawdown > max_drawdown:
        max_drawdown = drawdown
    step += 1

final_net_worth = net_worth
print(f"Test completed after {step} steps.")
print(f"Total Reward: {total_reward:.2f}")
print(f"Final Net Worth: ${final_net_worth:,.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")

# Optional: render final state
test_env.env_method('render')