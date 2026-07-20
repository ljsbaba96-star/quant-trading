"""
策略优化 v3 — 现金避险 + 参数搜索
只用已有的5只ETF，但通过现金避险控制回撤
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data.fetcher_v2 import fetch_etf_pool
from backtest.engine_v2 import BacktestEngineV2
from strategies.momentum_v2 import MomentumRotationV2


def fmt_pct(v): return f"{v*100:.2f}%"
def fmt_money(v): return f"¥{v:,.2f}"


# 现金避险策略：不买任何ETF，直接返回空信号
class CashRiskOffStrategy(MomentumRotationV2):
    """当需要避险时，不买任何标的，直接持有现金"""

    def generate_signals(self, all_data, date, current_holdings=None, entry_prices=None):
        result = super().generate_signals(all_data, date, current_holdings, entry_prices)
        if result["risk_off"]:
            # 避险模式：返回空信号，持有现金
            return {"signals": [], "stop_loss": result["stop_loss"], "risk_off": True}
        return result


def run_test(all_data, config, strat_config, label):
    engine = BacktestEngineV2(config)
    engine.strategy = CashRiskOffStrategy(strat_config)
    r = engine.run(all_data, start_date="2022-01-01")
    if r:
        r["_label"] = label
    return r


def main():
    print("=" * 70)
    print("  ETF动量轮动 v3 — 现金避险 + 参数优化")
    print("=" * 70)

    core_codes = ["510300", "510500", "588000", "159928", "512010"]
    print(f"\n📊 获取数据...")
    all_data = fetch_etf_pool(start_date="20210101", codes=core_codes)
    print(f"✅ {len(all_data)} 只\n")

    if len(all_data) < 3:
        print("数据不足"); return

    base = {
        "initial_capital": 20000,
        "commission_rate": 0.000085,
        "stamp_tax_rate": 0.0005,
        "min_commission": 0,
    }

    # ============================================================
    # 参数网格搜索
    # ============================================================
    tests = [
        # 基准
        ("基准: 纯动量", {
            "lookback_periods": [5, 10, 20], "weights": [0.3, 0.3, 0.4],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": False, "vol_adjust": False,
            "stop_loss_pct": -0.50, "portfolio_stop": -0.99,
            "risk_off_threshold": -0.99,  # 永不避险
        }),
        # 现金避险
        ("现金避险(动量<0)", {
            "lookback_periods": [5, 10, 20], "weights": [0.3, 0.3, 0.4],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": False, "vol_adjust": False,
            "stop_loss_pct": -0.50, "portfolio_stop": -0.99,
            "risk_off_threshold": 0.0,  # 动量为负就持现金
        }),
        # 趋势过滤 + 现金避险
        ("趋势+避险", {
            "lookback_periods": [5, 10, 20, 60], "weights": [0.15, 0.20, 0.35, 0.30],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": True, "trend_ma": 20, "vol_adjust": False,
            "stop_loss_pct": -0.08, "portfolio_stop": -0.12,
            "risk_off_threshold": 0.0,
        }),
        # 波动率调整 + 现金避险
        ("波动率调整+避险", {
            "lookback_periods": [5, 10, 20, 60], "weights": [0.15, 0.20, 0.35, 0.30],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": False, "vol_adjust": True, "vol_period": 20,
            "stop_loss_pct": -0.50, "portfolio_stop": -0.99,
            "risk_off_threshold": 0.0,
        }),
        # 严格止损 + 现金避险
        ("严格止损+避险", {
            "lookback_periods": [5, 10, 20, 60], "weights": [0.15, 0.20, 0.35, 0.30],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": True, "trend_ma": 20, "vol_adjust": True, "vol_period": 20,
            "stop_loss_pct": -0.05, "portfolio_stop": -0.08,
            "risk_off_threshold": 0.0,
        }),
        # Top1 集中
        ("Top1集中+避险", {
            "lookback_periods": [5, 10, 20], "weights": [0.3, 0.3, 0.4],
            "top_n": 1, "rebalance_freq": "weekly",
            "trend_filter": True, "trend_ma": 20, "vol_adjust": True, "vol_period": 20,
            "stop_loss_pct": -0.06, "portfolio_stop": -0.10,
            "risk_off_threshold": 0.0,
        }),
        # 双周调仓 + Top3
        ("双周+Top3+避险", {
            "lookback_periods": [5, 10, 20, 60], "weights": [0.15, 0.20, 0.35, 0.30],
            "top_n": 3, "rebalance_freq": "biweekly",
            "trend_filter": True, "trend_ma": 20, "vol_adjust": True, "vol_period": 20,
            "stop_loss_pct": -0.08, "portfolio_stop": -0.12,
            "risk_off_threshold": -0.01,
        }),
        # 短周期
        ("短周期日频+避险", {
            "lookback_periods": [3, 5, 10], "weights": [0.4, 0.35, 0.25],
            "top_n": 1, "rebalance_freq": "daily",
            "trend_filter": True, "trend_ma": 10, "vol_adjust": True, "vol_period": 10,
            "stop_loss_pct": -0.03, "portfolio_stop": -0.06,
            "risk_off_threshold": 0.0,
        }),
        # 长周期稳健
        ("长周期月频+避险", {
            "lookback_periods": [10, 20, 60, 120], "weights": [0.10, 0.20, 0.35, 0.35],
            "top_n": 2, "rebalance_freq": "biweekly",
            "trend_filter": True, "trend_ma": 60, "vol_adjust": True, "vol_period": 60,
            "stop_loss_pct": -0.10, "portfolio_stop": -0.15,
            "risk_off_threshold": -0.01,
        }),
        # 最小回撤
        ("最小回撤模式", {
            "lookback_periods": [5, 10, 20, 60], "weights": [0.15, 0.20, 0.35, 0.30],
            "top_n": 2, "rebalance_freq": "weekly",
            "trend_filter": True, "trend_ma": 60, "vol_adjust": True, "vol_period": 20,
            "stop_loss_pct": -0.03, "portfolio_stop": -0.05,
            "risk_off_threshold": 0.02,  # 动量低于2%就避险
        }),
    ]

    results = []
    for label, cfg in tests:
        print(f"🔄 {label}")
        r = run_test(all_data, base, cfg, label)
        if r:
            results.append(r)
            # 年度收益
            daily = r.get("daily_values", [])
            yearly = {}
            for d in daily:
                y = d["date"][:4]
                if y not in yearly: yearly[y] = {"f": d["value"], "l": d["value"]}
                yearly[y]["l"] = d["value"]
            yr_str = " | ".join(f"{y}:{fmt_pct(v['l']/v['f']-1)}" for y, v in sorted(yearly.items()))
            print(f"   收益:{fmt_pct(r['总收益率'])} 年化:{fmt_pct(r['年化收益率'])} "
                  f"回撤:{fmt_pct(r['最大回撤'])} Sharpe:{r['Sharpe']:.3f} "
                  f"胜率:{fmt_pct(r['交易胜率'])} 避险:{r['避险天数']}天")
            print(f"   年度: {yr_str}")
        print()

    # 排名
    if results:
        results.sort(key=lambda x: x.get("Sharpe", -999), reverse=True)
        print("=" * 70)
        print("  📊 综合排名")
        print("=" * 70)
        print(f"\n{'#':<3} {'策略':<25} {'总收益':>8} {'年化':>8} {'回撤':>8} "
              f"{'Sharpe':>8} {'胜率':>8} {'避险天':>6}")
        print("-" * 78)
        for i, r in enumerate(results, 1):
            print(f" {i:<3} {r['_label']:<25} {fmt_pct(r['总收益率']):>8} "
                  f"{fmt_pct(r['年化收益率']):>8} {fmt_pct(r['最大回撤']):>8} "
                  f"{r['Sharpe']:>8.3f} {fmt_pct(r['交易胜率']):>8} {r['避险天数']:>6}")

        best = results[0]
        print(f"\n🏆 最优: {best['_label']}")
        print(f"   {fmt_money(best['初始资金'])} → {fmt_money(best['期末资金'])}")

        # 最优策略详细年度
        daily = best.get("daily_values", [])
        if daily:
            print(f"\n📅 年度明细:")
            yearly = {}
            for d in daily:
                y = d["date"][:4]
                if y not in yearly: yearly[y] = {"f": d["value"], "l": d["value"]}
                yearly[y]["l"] = d["value"]
            for y, v in sorted(yearly.items()):
                ret = v["l"]/v["f"] - 1
                print(f"   {y}: {fmt_pct(ret):>10} → {fmt_money(v['l'])}")


if __name__ == "__main__":
    main()
