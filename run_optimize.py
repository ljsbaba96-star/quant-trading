"""
策略优化对比 — 跑多组参数，找出最优配置
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher_v2 import fetch_etf_pool, ETF_POOL
from backtest.engine_v2 import BacktestEngineV2
from strategies.momentum_v2 import MomentumRotationV2
import json


def fmt_pct(v): return f"{v*100:.2f}%"
def fmt_money(v): return f"¥{v:,.2f}"


def run_single(all_data, config, strategy_config, label=""):
    """运行单次回测"""
    engine = BacktestEngineV2(config)
    engine.strategy = MomentumRotationV2(strategy_config)
    results = engine.run(all_data, start_date="2022-01-01")
    if results:
        results["_label"] = label
    return results


def print_result(r):
    if not r:
        print("  ❌ 无结果")
        return
    print(f"  {r['_label']}")
    print(f"    资金: {fmt_money(r['初始资金'])} → {fmt_money(r['期末资金'])}  "
          f"收益: {fmt_pct(r['总收益率'])}  年化: {fmt_pct(r['年化收益率'])}")
    print(f"    Sharpe: {r['Sharpe']:.3f}  Sortino: {r['Sortino']:.3f}  "
          f"最大回撤: {fmt_pct(r['最大回撤'])}  周胜率: {fmt_pct(r['周胜率'])}")
    print(f"    交易胜率: {fmt_pct(r['交易胜率'])}  盈亏比: {r['盈亏比']:.2f}  "
          f"止损: {r['止损次数']}次  交易成本: {fmt_money(r['总交易成本'])}")


def main():
    print("=" * 70)
    print("  A股量化 — ETF动量轮动策略 v2 优化对比")
    print("=" * 70)

    # 获取数据（用核心池，避免API限流）
    core_codes = ["510300", "510500", "588000", "159928", "512010",
                  "510050", "512690", "512880", "518880", "512890"]
    print(f"\n📊 获取 {len(core_codes)} 只 ETF 数据...")
    all_data = fetch_etf_pool(start_date="20210101", codes=core_codes)
    print(f"✅ 成功获取 {len(all_data)} 只\n")

    if len(all_data) < 3:
        print("数据不足，退出")
        return

    base_config = {
        "initial_capital": 20000,
        "commission_rate": 0.000085,
        "stamp_tax_rate": 0.0005,
        "min_commission": 0,
    }

    # ============================================================
    # 参数组合测试
    # ============================================================
    configs = [
        {
            "label": "A: 基础动量(周频, Top2, 无止损)",
            "strategy": {
                "lookback_periods": [5, 10, 20],
                "weights": [0.3, 0.3, 0.4],
                "top_n": 2,
                "rebalance_freq": "weekly",
                "trend_filter": False,
                "vol_adjust": False,
                "stop_loss_pct": -0.50,   # 实质上不停损
                "portfolio_stop": -0.99,
                "risk_off_codes": ["518880"],
                "risk_off_threshold": -0.10,
            },
        },
        {
            "label": "B: 趋势过滤 + 波动率调整",
            "strategy": {
                "lookback_periods": [5, 10, 20, 60],
                "weights": [0.15, 0.20, 0.35, 0.30],
                "top_n": 2,
                "rebalance_freq": "weekly",
                "trend_filter": True,
                "trend_ma": 20,
                "vol_adjust": True,
                "vol_period": 20,
                "stop_loss_pct": -0.08,
                "portfolio_stop": -0.12,
                "risk_off_codes": ["518880"],
                "risk_off_threshold": -0.01,
            },
        },
        {
            "label": "C: 严格止损 + 避险切换",
            "strategy": {
                "lookback_periods": [5, 10, 20, 60],
                "weights": [0.15, 0.20, 0.35, 0.30],
                "top_n": 2,
                "rebalance_freq": "weekly",
                "trend_filter": True,
                "trend_ma": 20,
                "vol_adjust": True,
                "vol_period": 20,
                "stop_loss_pct": -0.05,    # 严格止损
                "portfolio_stop": -0.08,    # 组合-8%清仓
                "risk_off_codes": ["518880"],
                "risk_off_threshold": 0.0,  # 动量为负就避险
            },
        },
        {
            "label": "D: 双周调仓 + Top3 + 宽松止损",
            "strategy": {
                "lookback_periods": [5, 10, 20, 60],
                "weights": [0.15, 0.20, 0.35, 0.30],
                "top_n": 3,
                "rebalance_freq": "biweekly",
                "trend_filter": True,
                "trend_ma": 20,
                "vol_adjust": True,
                "vol_period": 20,
                "stop_loss_pct": -0.10,
                "portfolio_stop": -0.15,
                "risk_off_codes": ["518880"],
                "risk_off_threshold": -0.02,
            },
        },
        {
            "label": "E: 短周期动量(日频, Top1, 黄金避险)",
            "strategy": {
                "lookback_periods": [3, 5, 10],
                "weights": [0.4, 0.35, 0.25],
                "top_n": 1,
                "rebalance_freq": "daily",
                "trend_filter": True,
                "trend_ma": 10,
                "vol_adjust": True,
                "vol_period": 10,
                "stop_loss_pct": -0.03,
                "portfolio_stop": -0.06,
                "risk_off_codes": ["518880"],
                "risk_off_threshold": -0.005,
            },
        },
        {
            "label": "F: 长周期稳健(月频, Top2, 红利+黄金)",
            "strategy": {
                "lookback_periods": [10, 20, 60, 120],
                "weights": [0.10, 0.20, 0.35, 0.35],
                "top_n": 2,
                "rebalance_freq": "biweekly",
                "trend_filter": True,
                "trend_ma": 60,
                "vol_adjust": True,
                "vol_period": 60,
                "stop_loss_pct": -0.10,
                "portfolio_stop": -0.15,
                "risk_off_codes": ["518880", "512890"],  # 黄金+红利低波
                "risk_off_threshold": -0.02,
            },
        },
    ]

    # 运行所有配置
    results = []
    for cfg in configs:
        print(f"🔄 测试: {cfg['label']}")
        r = run_single(all_data, base_config, cfg["strategy"], cfg["label"])
        if r:
            results.append(r)
            print_result(r)
            print()

    # ============================================================
    # 汇总排名
    # ============================================================
    if not results:
        print("所有配置均无结果")
        return

    print("\n" + "=" * 70)
    print("  📊 综合排名（按 Sharpe 排序）")
    print("=" * 70)

    results.sort(key=lambda x: x.get("Sharpe", -999), reverse=True)

    print(f"\n{'排名':<4} {'策略':<35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'胜率':>8}")
    print("-" * 75)
    for i, r in enumerate(results, 1):
        label = r["_label"][:33]
        print(f"  {i:<3} {label:<35} {fmt_pct(r['年化收益率']):>8} "
              f"{fmt_pct(r['最大回撤']):>8} {r['Sharpe']:>8.3f} {fmt_pct(r['周胜率']):>8}")

    # 最优策略详情
    best = results[0]
    print(f"\n🏆 最优策略: {best['_label']}")
    print(f"   {fmt_money(best['初始资金'])} → {fmt_money(best['期末资金'])}")
    print(f"   年化: {fmt_pct(best['年化收益率'])}  最大回撤: {fmt_pct(best['最大回撤'])}")
    print(f"   Sharpe: {best['Sharpe']:.3f}  交易胜率: {fmt_pct(best['交易胜率'])}  盈亏比: {best['盈亏比']:.2f}")

    # 年度收益
    daily = best.get("daily_values", [])
    if daily:
        print(f"\n📅 最优策略年度收益:")
        yearly = {}
        for d in daily:
            y = d["date"][:4]
            if y not in yearly: yearly[y] = {"f": d["value"], "l": d["value"]}
            yearly[y]["l"] = d["value"]
        for y, v in sorted(yearly.items()):
            ret = v["l"] / v["f"] - 1
            print(f"   {y}: {fmt_pct(ret):>10}  →  {fmt_money(v['l'])}")

    # 最近交易
    trades = best.get("trades_log", [])
    if trades:
        print(f"\n📋 最近5笔交易:")
        for t in trades[-5:]:
            print(f"   {t['date']} {t['action']:<10} {t['code']} {t['shares']}股 @{t['price']:.3f}")

    print("\n" + "=" * 70)
    print("  优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
