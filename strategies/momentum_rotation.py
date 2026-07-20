"""
ETF 动量轮动策略
核心逻辑：计算多周期动量得分，选择 Top N 持有
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MomentumRotationStrategy:
    """ETF 动量轮动策略"""

    def __init__(self, config: dict = None):
        self.config = config or {
            "lookback_periods": [5, 10, 20],    # 动量回看周期
            "weights": [0.3, 0.3, 0.4],          # 各周期权重
            "top_n": 2,                           # 持有Top N只
            "rebalance_freq": "weekly",           # 调仓频率: daily/weekly/biweekly
            "risk_off_code": "518880",            # 避险标的（黄金ETF）
            "risk_off_threshold": -0.02,          # 动量跌破此值切换避险
            "min_momentum": 0.0,                  # 最低动量阈值
        }

    def calc_momentum_score(self, close_series: pd.Series) -> float:
        """
        计算动量得分
        得分 = Σ(weight_i × N日涨幅_i)
        """
        weights = self.config["weights"]
        periods = self.config["lookback_periods"]

        if len(close_series) < max(periods):
            return np.nan

        score = 0.0
        for period, weight in zip(periods, weights):
            if len(close_series) >= period:
                ret = (close_series.iloc[-1] / close_series.iloc[-period]) - 1
                score += weight * ret

        return score

    def calc_volatility_score(self, close_series: pd.Series, period: int = 20) -> float:
        """计算波动率（用于风险调整）"""
        if len(close_series) < period:
            return np.nan
        returns = close_series.pct_change().dropna().tail(period)
        return returns.std() * np.sqrt(252)

    def calc_risk_adjusted_momentum(self, close_series: pd.Series) -> float:
        """风险调整后的动量得分 = 动量 / 波动率"""
        momentum = self.calc_momentum_score(close_series)
        volatility = self.calc_volatility_score(close_series)
        if volatility == 0 or np.isnan(volatility) or np.isnan(momentum):
            return np.nan
        return momentum / volatility

    def generate_signals(self, all_data: Dict[str, pd.DataFrame],
                          date: str) -> List[Tuple[str, float]]:
        """
        生成交易信号
        返回: [(etf_code, target_weight), ...]
        """
        scores = {}

        for code, df in all_data.items():
            df_date = df[df["日期"] <= date]
            if df_date.empty:
                continue

            close = df_date["收盘"].astype(float)
            score = self.calc_risk_adjusted_momentum(close)

            if not np.isnan(score):
                scores[code] = score

        if not scores:
            return []

        # 排序并选择 Top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_n = self.config["top_n"]
        risk_off_code = self.config["risk_off_code"]
        risk_off_threshold = self.config["risk_off_threshold"]
        min_momentum = self.config["min_momentum"]

        selected = []
        for code, score in sorted_scores:
            if code == risk_off_code:
                continue
            if score < min_momentum:
                break
            if len(selected) < top_n:
                selected.append((code, score))

        # 如果所有标的动量都很差，切换到避险
        if not selected or all(s < risk_off_threshold for _, s in selected):
            if risk_off_code in scores:
                return [(risk_off_code, 1.0)]
            return []

        # 等权分配
        weight = 1.0 / len(selected)
        signals = [(code, weight) for code, _ in selected]

        return signals

    def should_rebalance(self, current_date: str, last_rebalance: str) -> bool:
        """判断是否需要调仓"""
        freq = self.config["rebalance_freq"]
        current = pd.Timestamp(current_date)
        last = pd.Timestamp(last_rebalance)

        if freq == "daily":
            return True
        elif freq == "weekly":
            # 每周五调仓
            return current.weekday() == 4 or (current - last).days >= 7
        elif freq == "biweekly":
            return (current - last).days >= 14

        return False
