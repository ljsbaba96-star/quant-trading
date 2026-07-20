"""
主运行脚本 - ETF 动量轮动策略回测
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_all_etf_data, ETF_POOL
from backtest.engine import BacktestEngine


def fmt_pct(v): return f"{v*100:.2f}%"
def fmt_money(v): return f"¥{v:,.2f}"


def run_backtest():
    print("=" * 60)
    print("  A股量化交易系统 - ETF动量轮动策略回测")
    print("=" * 60)

    print("\n📊 正在获取 ETF 历史数据...")
    all_data = fetch_all_etf_data(start_date="20210101")
    if not all_data:
        print("❌ 未获取到数据"); return

    print(f"\n✅ 成功获取 {len(all_data)} 只 ETF")
    for code, df in all_data.items():
        info = ETF_POOL.get(code, {})
        print(f"   {info.get('name',code)}: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}, {len(df)}条")

    config = {
        "initial_capital": 20000,
        "commission_rate": 0.000085,
        "stamp_tax_rate": 0.0005,
        "min_commission": 0,
    }

    print(f"\n🔄 回测中...")
    print(f"   资金: {fmt_money(config['initial_capital'])} | 佣金: 万0.85(免五) | 印花税: 万5")
    print(f"   调仓: 每周 | 持仓: Top 2 ETF")

    engine = BacktestEngine(config)
    results = engine.run(all_data, start_date="2022-01-01")
    if not results:
        print("❌ 回测失败"); return

    # 结果
    print("\n" + "=" * 60)
    print("  📈 回测结果")
    print("=" * 60)
    print(f"\n{'指标':<20} {'数值':>15}")
    print("-" * 37)
    print(f"{'初始资金':<20} {fmt_money(results['初始资金']):>15}")
    print(f"{'期末资金':<20} {fmt_money(results['期末资金']):>15}")
    print(f"{'总收益率':<20} {fmt_pct(results['总收益率']):>15}")
    print(f"{'年化收益率':<20} {fmt_pct(results['年化收益率']):>15}")
    print(f"{'年化波动率':<20} {fmt_pct(results['年化波动率']):>15}")
    print(f"{'Sharpe':<20} {results['Sharpe']:>15.3f}")
    print(f"{'Sortino':<20} {results['Sortino']:>15.3f}")
    print(f"{'Calmar':<20} {results['Calmar']:>15.3f}")
    print(f"{'最大回撤':<20} {fmt_pct(results['最大回撤']):>15}")
    print(f"{'交易次数':<20} {results['交易次数']:>15}")
    print(f"{'交易成本':<20} {fmt_money(results['总交易成本']):>15}")
    print(f"{'周胜率':<20} {fmt_pct(results['周胜率']):>15}")

    # 年度收益
    daily = results.get("daily_values", [])
    if daily:
        print(f"\n📅 年度收益:")
        yearly = {}
        for d in daily:
            y = d["date"][:4]
            if y not in yearly: yearly[y] = {"first": d["value"], "last": d["value"]}
            yearly[y]["last"] = d["value"]
        for y, v in sorted(yearly.items()):
            ret = v["last"]/v["first"] - 1
            print(f"   {y}: {fmt_pct(ret):>10}  →  {fmt_money(v['last'])}")

    # 当前持仓
    if daily:
        last = daily[-1]
        print(f"\n📋 最新净值: {fmt_money(last['value'])}")
        print(f"   现金: {fmt_money(last['cash'])}")

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    run_backtest()
