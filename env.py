"""
SmartTradingEnv - A Risk-First Reinforcement Learning Environment for Stock Trading.
Built on Gymnasium. Features discrete actions (Hold, Buy, Sell), slippage tolerance,
timestamp lag validation, and capital-protection reward with max drawdown penalty.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Tuple, Dict, Any


class SmartTradingEnv(gym.Env):
    """
    Custom Trading Environment with risk-first design:
    - Action space: 0=Hold, 1=Buy, 2=Sell (Discrete)
    - Observation: [Balance, Shares Held, Cost Basis, Net Worth, RSI, MACD, ATR]
    - Slippage: random execution price within ± slippage_tolerance of current close
    - Timestamp lag: checks if data interval > threshold, forces hold if lag detected
    - Reward: ΔNetWorth - transaction_fee - exponential_drawdown_penalty (if drawdown > 5%)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        slippage_tolerance: float = 0.0005,    # 0.05%
        transaction_fee_rate: float = 0.001,   # 0.1% per trade value
        timestamp_lag_threshold: float = 1.0,  # seconds, for live data freshness check
    ):
        super().__init__()

        if df.empty:
            raise ValueError("DataFrame must not be empty")
        required_cols = {'close', 'rsi', 'macd', 'atr'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.df = df.reset_index(drop=True).copy()
        self.initial_balance = initial_balance
        self.slippage_tolerance = slippage_tolerance
        self.transaction_fee_rate = transaction_fee_rate
        self.timestamp_lag_threshold = timestamp_lag_threshold

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Internal state
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0       # average cost per share
        self.current_step = 0
        self.net_worth = self.balance
        self.peak_net_worth = self.balance
        self.prev_net_worth = self.balance
        self.done = False
        self.info: Dict[str, Any] = {}

        # Seed for reproducibility of slippage randomness
        self._np_random = np.random.default_rng()

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # Reset state
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0
        self.current_step = 0
        self.net_worth = self.balance
        self.peak_net_worth = self.balance
        self.prev_net_worth = self.balance
        self.done = False
        self.info = {}
        # Reset random generator if seed provided
        self._np_random = np.random.default_rng(seed)

        obs = self._next_observation()
        return obs, self.info

    def _next_observation(self) -> np.ndarray:
        """Extract observation from current data row."""
        row = self.df.iloc[self.current_step]
        self.current_price = float(row["close"])
        self.current_time = row.get("date", None)  # optional timestamp

        rsi = float(row["rsi"])
        macd = float(row["macd"])
        atr = float(row["atr"])

        obs = np.array(
            [
                self.balance,
                self.shares_held,
                self.cost_basis,
                self.net_worth,
                rsi,
                macd,
                atr,
            ],
            dtype=np.float32,
        )
        return obs

    def _calculate_net_worth(self, price: float = None) -> float:
        if price is None:
            price = self.current_price
        return self.balance + self.shares_held * price

    def _check_timestamp_lag(self) -> bool:
        """
        Validate data freshness: return True if time gap between consecutive steps
        exceeds timestamp_lag_threshold (seconds). Works only if 'date' column is
        present and contains pandas Timestamps.
        """
        if self.current_step > 0 and "date" in self.df.columns:
            prev_time = self.df.iloc[self.current_step - 1]["date"]
            curr_time = self.df.iloc[self.current_step]["date"]
            if isinstance(prev_time, pd.Timestamp) and isinstance(curr_time, pd.Timestamp):
                lag = (curr_time - prev_time).total_seconds()
                return lag > self.timestamp_lag_threshold
        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done, call reset() before stepping again.")

        # ---- Timestamp lag guard ----
        lag_detected = self._check_timestamp_lag()
        if lag_detected:
            # In live trading lag could invalidate trade decisions.
            # Here we force hold and add a warning flag.
            action = 0
            self.info["lag_warning"] = True
        else:
            self.info["lag_warning"] = False

        prev_net_worth = self.net_worth
        transaction_fee = 0.0

        # ---- Simulate slippage for execution price ----
        slippage_factor = self._np_random.uniform(
            -self.slippage_tolerance, self.slippage_tolerance
        )
        execution_price = self.current_price * (1 + slippage_factor)

        # ---- Process action ----
        if action == 1:  # Buy
            if self.balance > 1e-6:
                # Maximum shares purchasable given balance and fee
                # balance = shares * execution_price * (1 + fee_rate)
                shares_to_buy = self.balance / (
                    execution_price * (1 + self.transaction_fee_rate)
                )
                if shares_to_buy > 0:
                    # Update average cost basis
                    total_cost = (
                        self.shares_held * self.cost_basis
                        + shares_to_buy * execution_price
                    )
                    self.shares_held += shares_to_buy
                    self.cost_basis = total_cost / self.shares_held if self.shares_held > 0 else 0.0

                    trade_value = shares_to_buy * execution_price
                    transaction_fee = trade_value * self.transaction_fee_rate
                    self.balance -= (trade_value + transaction_fee)

        elif action == 2:  # Sell
            if self.shares_held > 0:
                trade_value = self.shares_held * execution_price
                transaction_fee = trade_value * self.transaction_fee_rate
                self.balance += (trade_value - transaction_fee)
                self.shares_held = 0.0
                self.cost_basis = 0.0

        # Action 0 (Hold) does nothing

        # ---- Update net worth using current market price (after trade) ----
        self.net_worth = self._calculate_net_worth(self.current_price)

        # ---- Reward: capital protection + fee penalty + drawdown guard ----
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change - transaction_fee

        # Update peak net worth
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth

        # Exponential drawdown penalty if drop > 5% from peak
        drawdown = 0.0
        if self.peak_net_worth > 0:
            drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth

        if drawdown > 0.05:
            # Exponential penalty: starts small at 5% drawdown and grows rapidly
            penalty = np.exp(10 * (drawdown - 0.05))
            reward -= penalty

        # ---- Advance to next time step ----
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        self.done = terminated

        # Prepare next observation
        if not terminated:
            obs = self._next_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Build info dictionary
        info = {
            "net_worth": self.net_worth,
            "drawdown": drawdown,
            "transaction_fee": transaction_fee,
            "lag_warning": self.info.get("lag_warning", False),
        }
        self.info = info
        self.prev_net_worth = self.net_worth

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step} | "
            f"Net Worth: {self.net_worth:,.2f} | "
            f"Shares: {self.shares_held:.4f} | "
            f"Drawdown: {self.info.get('drawdown', 0)*100:.2f}%"
        )

    def close(self):
        pass