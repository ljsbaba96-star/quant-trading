"""
回测引擎 v2 - 修复资金跟踪和信号生成逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.momentum_rotation import MomentumRotationStrategy


class BacktestEngine:
    def __init__(self, config: dict = None):
        self.config = config or {
            "initial_capital": 20000,
            "commission_rate": 0.000085,   # 万0.85
            "stamp_tax_rate": 0.0005,      # 万5（卖出）
            "min_commission": 0,           # 免五
        }
        self.strategy = MomentumRotationStrategy()

    def _calc_cost(self, amount: float, is_sell: bool) -> float:
        """计算交易成本"""
        commission = max(amount * self.config["commission_rate"],
                        self.config["min_commission"])
        stamp = amount * self.config["stamp_tax_rate"] if is_sell else 0
        return commission + stamp

    def run(self, all_data: Dict[str, pd.DataFrame],
            start_date: str = "2022-01-01",
            end_date: str = None) -> dict:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 获取交易日序列
        all_dates = set()
        for df in all_data.values():
            dates_in_range = df[df["日期"] >= start_date]["日期"].tolist()
            all_dates.update(dates_in_range)
        trade_dates = sorted(all_dates)

        if not trade_dates:
            print("无交易日数据")
            return {}

        # 状态
        cash = self.config["initial_capital"]
        holdings = {}  # {code: {"shares": int, "avg_cost": float}}
        last_rebalance_date = None

        # 记录
        daily_values = []
        trades_log = []

        for date in trade_dates:
            # ---- 1. 计算当日总资产 ----
            portfolio_value = cash
            for code, h in holdings.items():
                price = self._price(all_data, code, date)
                if price > 0:
                    portfolio_value += h["shares"] * price

            daily_values.append({
                "date": date,
                "value": portfolio_value,
                "cash": cash,
            })

            # ---- 2. 判断是否调仓 ----
            is_first_day = (last_rebalance_date is None)
            should_rebal = self.strategy.should_rebalance(date, last_rebalance_date or date)

            if not (is_first_day or should_rebal):
                continue

            # ---- 3. 生成信号 ----
            signals = self.strategy.generate_signals(all_data, date)
            if not signals:
                last_rebalance_date = date
                continue

            target_codes = {s[0] for s in signals}

            # ---- 4. 卖出不在目标中的持仓 ----
            for code in list(holdings.keys()):
                if code not in target_codes:
                    price = self._price(all_data, code, date)
                    if price <= 0:
                        continue
                    shares = holdings[code]["shares"]
                    sell_amount = shares * price
                    cost = self._calc_cost(sell_amount, is_sell=True)
                    cash += sell_amount - cost
                    trades_log.append({
                        "date": date, "code": code, "action": "SELL",
                        "shares": shares, "price": price, "cost": cost,
                    })
                    del holdings[code]

            # ---- 5. 买入目标持仓 ----
            total_value = cash  # 当前可用资金
            for code, weight in signals:
                price = self._price(all_data, code, date)
                if price <= 0:
                    continue
                # 检查是否已持有
                if code in holdings:
                    total_value += holdings[code]["shares"] * price

            for code, weight in signals:
                price = self._price(all_data, code, date)
                if price <= 0:
                    continue

                target_amount = total_value * weight
                current_amount = 0
                if code in holdings:
                    current_amount = holdings[code]["shares"] * price

                diff = target_amount - current_amount

                if diff > 100:  # 需要买入
                    buy_amount = min(diff, cash * 0.98)
                    shares_to_buy = int(buy_amount / price / 100) * 100
                    if shares_to_buy > 0:
                        cost = self._calc_cost(shares_to_buy * price, is_sell=False)
                        cash -= shares_to_buy * price + cost
                        if code in holdings:
                            old = holdings[code]
                            total_shares = old["shares"] + shares_to_buy
                            avg_cost = (old["avg_cost"] * old["shares"] + price * shares_to_buy) / total_shares
                            holdings[code] = {"shares": total_shares, "avg_cost": avg_cost}
                        else:
                            holdings[code] = {"shares": shares_to_buy, "avg_cost": price}
                        trades_log.append({
                            "date": date, "code": code, "action": "BUY",
                            "shares": shares_to_buy, "price": price, "cost": cost,
                        })

            last_rebalance_date = date

        # ---- 计算绩效 ----
        return self._calc_performance(daily_values, trades_log)

    def _price(self, all_data, code, date):
        df = all_data.get(code)
        if df is None:
            return 0
        row = df[df["日期"] == date]
        if row.empty:
            return 0
        return float(row["收盘"].iloc[0])

    def _calc_performance(self, daily_values, trades_log):
        if not daily_values:
            return {}

        values = pd.Series(
            [d["value"] for d in daily_values],
            index=pd.to_datetime([d["date"] for d in daily_values])
        )

        initial = values.iloc[0]
        final = values.iloc[-1]
        total_return = final / initial - 1
        days = (values.index[-1] - values.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        daily_ret = values.pct_change().dropna()
        annual_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        max_dd = drawdown.min()

        # Sortino
        down_ret = daily_ret[daily_ret < 0]
        down_std = down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else 0
        sortino = (annual_return - 0.02) / down_std if down_std > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # 胜率（按调仓周期算）
        rebal_values = []
        prev_val = initial
        for d in daily_values:
            rebal_values.append(d["value"])

        # 用滚动周收益算胜率
        weekly_ret = values.resample("W").last().pct_change().dropna()
        win_rate = (weekly_ret > 0).sum() / len(weekly_ret) if len(weekly_ret) > 0 else 0

        # 总交易成本
        total_cost = sum(t["cost"] for t in trades_log)

        return {
            "初始资金": initial,
            "期末资金": final,
            "总收益率": total_return,
            "年化收益率": annual_return,
            "年化波动率": annual_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "最大回撤": max_dd,
            "交易次数": len(trades_log),
            "总交易成本": total_cost,
            "周胜率": win_rate,
            "回测天数": days,
            "daily_values": daily_values,
            "trades_log": trades_log,
        }
