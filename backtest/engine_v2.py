"""
回测引擎 v2 — 支持止损、避险、基准对比
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.momentum_v2 import MomentumRotationV2


class BacktestEngineV2:
    def __init__(self, config: dict = None):
        self.config = config or {
            "initial_capital": 20000,
            "commission_rate": 0.000085,
            "stamp_tax_rate": 0.0005,
            "min_commission": 0,
        }
        self.strategy = MomentumRotationV2()

    def _cost(self, amount, is_sell):
        c = max(amount * self.config["commission_rate"], self.config["min_commission"])
        c += amount * self.config["stamp_tax_rate"] if is_sell else 0
        return c

    def _price(self, data, code, date):
        df = data.get(code)
        if df is None: return 0
        r = df[df["日期"] == date]
        return float(r["收盘"].iloc[0]) if not r.empty else 0

    def run(self, all_data: Dict[str, pd.DataFrame],
            start_date: str = "2022-01-01",
            end_date: str = None) -> dict:

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 交易日序列
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df[df["日期"] >= start_date]["日期"].tolist())
        trade_dates = sorted(all_dates)
        if not trade_dates:
            return {}

        # 状态
        cash = self.config["initial_capital"]
        holdings = {}       # {code: {"shares": int, "avg_cost": float}}
        entry_prices = {}   # {code: float} 买入价
        last_rebal = None
        portfolio_stop_triggered = False

        # 记录
        daily_values = []
        trades_log = []
        rebal_count = 0
        risk_off_days = 0

        for date in trade_dates:
            # ---- 计算总资产 ----
            port_val = cash
            for code, h in holdings.items():
                p = self._price(all_data, code, date)
                if p > 0:
                    port_val += h["shares"] * p

            daily_values.append({"date": date, "value": port_val, "cash": cash})

            # ---- 组合止损 ----
            if self.config.get("initial_capital"):
                dd = port_val / self.config["initial_capital"] - 1
                if dd < self.strategy.config["portfolio_stop"] and not portfolio_stop_triggered:
                    # 清仓
                    for code in list(holdings.keys()):
                        p = self._price(all_data, code, date)
                        if p > 0:
                            shares = holdings[code]["shares"]
                            sell_amt = shares * p
                            cost = self._cost(sell_amt, True)
                            cash += sell_amt - cost
                            trades_log.append({"date": date, "code": code, "action": "STOP_SELL",
                                              "shares": shares, "price": p})
                    holdings.clear()
                    entry_prices.clear()
                    portfolio_stop_triggered = True
                    last_rebal = date
                    continue

            # ---- 判断调仓 ----
            is_first = last_rebal is None
            should_rebal = self.strategy.should_rebalance(date, last_rebal or date)
            if not (is_first or should_rebal):
                continue

            # ---- 生成信号 ----
            result = self.strategy.generate_signals(
                all_data, date,
                current_holdings=list(holdings.keys()),
                entry_prices=entry_prices,
            )
            signals = result["signals"]
            stop_codes = result["stop_loss"]
            is_risk_off = result["risk_off"]

            if is_risk_off:
                risk_off_days += 1

            if not signals and not stop_codes:
                last_rebal = date
                continue

            target_codes = {s[0] for s in signals}

            # ---- 止损卖出 ----
            for code in stop_codes:
                if code in holdings:
                    p = self._price(all_data, code, date)
                    if p > 0:
                        shares = holdings[code]["shares"]
                        sell_amt = shares * p
                        cost = self._cost(sell_amt, True)
                        cash += sell_amt - cost
                        trades_log.append({"date": date, "code": code, "action": "STOP_LOSS",
                                          "shares": shares, "price": p})
                        del holdings[code]
                        del entry_prices[code]

            # ---- 常规卖出 ----
            for code in list(holdings.keys()):
                if code not in target_codes:
                    p = self._price(all_data, code, date)
                    if p > 0:
                        shares = holdings[code]["shares"]
                        sell_amt = shares * p
                        cost = self._cost(sell_amt, True)
                        cash += sell_amt - cost
                        trades_log.append({"date": date, "code": code, "action": "SELL",
                                          "shares": shares, "price": p})
                        del holdings[code]
                        del entry_prices[code]

            # ---- 计算可用资金 ----
            total_val = cash
            for code, h in holdings.items():
                p = self._price(all_data, code, date)
                if p > 0:
                    total_val += h["shares"] * p

            # ---- 买入 ----
            for code, weight in signals:
                p = self._price(all_data, code, date)
                if p <= 0:
                    continue
                target_amt = total_val * weight
                cur_amt = holdings.get(code, {}).get("shares", 0) * p
                diff = target_amt - cur_amt
                if diff > 100 and cash > 0:
                    buy_amt = min(diff, cash * 0.98)
                    shares = int(buy_amt / p / 100) * 100
                    if shares > 0:
                        cost = self._cost(shares * p, False)
                        cash -= shares * p + cost
                        if code in holdings:
                            old = holdings[code]
                            total_s = old["shares"] + shares
                            avg = (old["avg_cost"] * old["shares"] + p * shares) / total_s
                            holdings[code] = {"shares": total_s, "avg_cost": avg}
                        else:
                            holdings[code] = {"shares": shares, "avg_cost": p}
                            entry_prices[code] = p
                        trades_log.append({"date": date, "code": code, "action": "BUY",
                                          "shares": shares, "price": p})

            last_rebal = date
            rebal_count += 1

        return self._calc_performance(daily_values, trades_log, rebal_count, risk_off_days)

    def _calc_performance(self, daily_values, trades_log, rebal_count, risk_off_days):
        if not daily_values:
            return {}

        vals = pd.Series([d["value"] for d in daily_values],
                        index=pd.to_datetime([d["date"] for d in daily_values]))

        init_val = vals.iloc[0]
        final_val = vals.iloc[-1]
        total_ret = final_val / init_val - 1
        days = (vals.index[-1] - vals.index[0]).days
        ann_ret = (1 + total_ret) ** (365 / max(days, 1)) - 1

        daily_ret = vals.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0

        peak = vals.expanding().max()
        dd = (vals - peak) / peak
        max_dd = dd.min()

        # Calmar & Sortino
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        down_ret = daily_ret[daily_ret < 0]
        down_std = down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else 0
        sortino = (ann_ret - 0.02) / down_std if down_std > 0 else 0

        # 周胜率
        weekly = vals.resample("W").last().pct_change().dropna()
        win_rate = (weekly > 0).sum() / len(weekly) if len(weekly) > 0 else 0

        # 最大回撤恢复天数
        max_dd_end = dd.idxmin()
        recovery = vals[max_dd_end:]
        recovery_date = None
        for d, v in recovery.items():
            if v >= peak[max_dd_end]:
                recovery_date = d
                break
        recovery_days = (recovery_date - max_dd_end).days if recovery_date else None

        # 交易成本
        total_cost = sum(t.get("cost", 0) for t in trades_log)

        # 盈亏交易统计
        buy_prices = {}
        pnl_list = []
        for t in trades_log:
            if t["action"] in ("BUY",):
                buy_prices[t["code"]] = t["price"]
            elif t["action"] in ("SELL", "STOP_SELL", "STOP_LOSS"):
                bp = buy_prices.get(t["code"])
                if bp:
                    pnl_list.append(t["price"] / bp - 1)

        win_trades = [p for p in pnl_list if p > 0]
        lose_trades = [p for p in pnl_list if p <= 0]
        trade_win_rate = len(win_trades) / len(pnl_list) if pnl_list else 0
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_lose = np.mean(lose_trades) if lose_trades else 0
        profit_loss_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else 0

        # 止损次数
        stop_count = sum(1 for t in trades_log if t["action"] in ("STOP_LOSS", "STOP_SELL"))

        return {
            "初始资金": init_val,
            "期末资金": final_val,
            "总收益率": total_ret,
            "年化收益率": ann_ret,
            "年化波动率": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "最大回撤": max_dd,
            "回撤恢复天数": recovery_days,
            "交易次数": len(trades_log),
            "调仓次数": rebal_count,
            "止损次数": stop_count,
            "避险天数": risk_off_days,
            "总交易成本": total_cost,
            "周胜率": win_rate,
            "交易胜率": trade_win_rate,
            "盈亏比": profit_loss_ratio,
            "平均盈利": avg_win,
            "平均亏损": avg_lose,
            "回测天数": days,
            "daily_values": daily_values,
            "trades_log": trades_log,
        }
