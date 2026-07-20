"""
动量轮动策略 v2 — 多维度改进
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

DEFAULTS = {
    "lookback_periods": [5, 10, 20, 60],
    "weights": [0.15, 0.20, 0.35, 0.30],
    "top_n": 2,
    "rebalance_freq": "weekly",
    "trend_ma": 20,
    "trend_filter": True,
    "vol_adjust": True,
    "vol_period": 20,
    "stop_loss_pct": -0.08,
    "portfolio_stop": -0.10,
    "risk_off_codes": ["518880"],
    "risk_off_threshold": -0.01,
}


class MomentumRotationV2:
    def __init__(self, config: dict = None):
        self.config = {**DEFAULTS, **(config or {})}

    def _trend_ok(self, close: pd.Series) -> bool:
        ma = self.config["trend_ma"]
        if len(close) < ma:
            return False
        return close.iloc[-1] > close.rolling(ma).mean().iloc[-1]

    def _momentum_score(self, close: pd.Series) -> float:
        periods = self.config["lookback_periods"]
        weights = self.config["weights"]
        if len(close) < max(periods):
            return np.nan
        score = 0.0
        for p, w in zip(periods, weights):
            if len(close) >= p:
                score += w * (close.iloc[-1] / close.iloc[-p] - 1)
        return score

    def _volatility(self, close: pd.Series) -> float:
        period = self.config["vol_period"]
        if len(close) < period:
            return np.nan
        return close.pct_change().dropna().tail(period).std() * np.sqrt(252)

    def _risk_adjusted_score(self, close: pd.Series) -> float:
        mom = self._momentum_score(close)
        vol = self._volatility(close)
        if np.isnan(mom) or np.isnan(vol) or vol == 0:
            return np.nan
        return mom / vol

    def generate_signals(self, all_data, date, current_holdings=None, entry_prices=None):
        scores = {}
        trend_pass = {}

        for code, df in all_data.items():
            df_d = df[df["日期"] <= date]
            if df_d.empty:
                continue
            close = df_d["收盘"].astype(float)

            trend_pass[code] = self._trend_ok(close)

            if self.config["vol_adjust"]:
                score = self._risk_adjusted_score(close)
            else:
                score = self._momentum_score(close)
            if not np.isnan(score):
                scores[code] = score

        # 个股止损
        stop_loss_codes = []
        if current_holdings and entry_prices:
            for code in current_holdings:
                if code in all_data:
                    df_d = all_data[code][all_data[code]["日期"] <= date]
                    if not df_d.empty:
                        cur = float(df_d["收盘"].iloc[-1])
                        ent = entry_prices.get(code, cur)
                        if cur / ent - 1 < self.config["stop_loss_pct"]:
                            stop_loss_codes.append(code)

        # 避险判断
        avg_mom = np.mean(list(scores.values())) if scores else 0
        risk_off = avg_mom < self.config["risk_off_threshold"]

        if risk_off:
            risk_codes = self.config["risk_off_codes"]
            signals = [(c, 1.0/len(risk_codes)) for c in risk_codes if c in all_data]
            return {"signals": signals, "stop_loss": stop_loss_codes, "risk_off": True}

        # 正常模式
        eligible = {c: s for c, s in scores.items()
                    if (not self.config["trend_filter"] or trend_pass.get(c, False))
                    and c not in stop_loss_codes}

        if not eligible:
            eligible = {c: s for c, s in scores.items() if c not in stop_loss_codes}
        if not eligible:
            return {"signals": [], "stop_loss": stop_loss_codes, "risk_off": False}

        sorted_e = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_e[:self.config["top_n"]]
        w = 1.0 / len(selected)
        return {"signals": [(c, w) for c, _ in selected], "stop_loss": stop_loss_codes, "risk_off": False}

    def should_rebalance(self, current_date, last_rebalance):
        freq = self.config["rebalance_freq"]
        cur = pd.Timestamp(current_date)
        last = pd.Timestamp(last_rebalance)
        if freq == "daily":
            return True
        elif freq == "weekly":
            return cur.weekday() == 4 or (cur - last).days >= 7
        elif freq == "biweekly":
            return (cur - last).days >= 14
        return False
