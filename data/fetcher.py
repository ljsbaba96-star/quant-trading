"""
数据获取模块 - 基于 AkShare
支持 ETF、股票、指数的历史行情数据获取
"""

import akshare as ak
import pandas as pd
import time
import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "market_data.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline (
            code TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (code, date)
        )
    """)
    conn.commit()
    return conn


def save_to_db(code: str, df: pd.DataFrame):
    """保存行情数据到本地 SQLite"""
    conn = get_db_conn()
    records = []
    for _, row in df.iterrows():
        records.append((
            code,
            str(row["日期"]),
            float(row["开盘"]),
            float(row["最高"]),
            float(row["最低"]),
            float(row["收盘"]),
            float(row["成交量"]),
            float(row["成交额"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO kline VALUES (?,?,?,?,?,?,?,?)",
        records,
    )
    conn.commit()
    conn.close()


def load_from_db(code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从本地数据库加载行情数据"""
    conn = get_db_conn()
    query = "SELECT * FROM kline WHERE code = ?"
    params = [code]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty:
        df.rename(columns={
            "code": "代码", "date": "日期", "open": "开盘",
            "high": "最高", "low": "最低", "close": "收盘",
            "volume": "成交量", "amount": "成交额",
        }, inplace=True)
    return df


def fetch_etf_hist(symbol: str, start_date: str = "20200101",
                   end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
    """
    获取 ETF 历史 K 线数据
    symbol: ETF 代码，如 "510300"
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if use_cache:
        cached = load_from_db(symbol, start_date, end_date)
        if not cached.empty:
            last_date = cached["日期"].iloc[-1]
            # 如果缓存足够新（1天内），直接返回
            if last_date >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                return cached

    print(f"  正在获取 {symbol} 行情数据...")
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                  start_date=start_date, end_date=end_date,
                                  adjust="qfq")
        if df is not None and not df.empty:
            save_to_db(symbol, df)
            time.sleep(0.5)  # 避免频率限制
            return df
    except Exception as e:
        print(f"  获取 {symbol} 失败: {e}")

    # 如果在线获取失败，返回缓存
    return load_from_db(symbol, start_date, end_date)


def fetch_stock_hist(symbol: str, start_date: str = "20200101",
                     end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
    """获取股票历史 K 线数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if use_cache:
        cached = load_from_db(symbol, start_date, end_date)
        if not cached.empty:
            last_date = cached["日期"].iloc[-1]
            if last_date >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                return cached

    print(f"  正在获取 {symbol} 行情数据...")
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start_date, end_date=end_date,
                                 adjust="qfq")
        if df is not None and not df.empty:
            save_to_db(symbol, df)
            time.sleep(0.5)
            return df
    except Exception as e:
        print(f"  获取 {symbol} 失败: {e}")

    return load_from_db(symbol, start_date, end_date)


def fetch_index_hist(symbol: str, start_date: str = "20200101",
                     end_date: str = None) -> pd.DataFrame:
    """获取指数历史数据（用于基准对比）"""
    print(f"  正在获取指数 {symbol} 行情数据...")
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is not None and not df.empty:
            df = df.rename(columns={"date": "日期", "open": "开盘",
                                     "high": "最高", "low": "最低",
                                     "close": "收盘", "volume": "成交量"})
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
            if start_date:
                df = df[df["日期"] >= start_date]
            if end_date:
                df = df[df["日期"] <= end_date]
            time.sleep(0.5)
            return df
    except Exception as e:
        print(f"  获取指数 {symbol} 失败: {e}")
    return pd.DataFrame()


# ============================================================
# ETF 标的池定义
# ============================================================

ETF_POOL = {
    "510300": {"name": "沪深300ETF", "category": "宽基"},
    "510500": {"name": "中证500ETF", "category": "宽基"},
    "159915": {"name": "创业板ETF", "category": "宽基"},
    "588000": {"name": "科创50ETF", "category": "宽基"},
    "510880": {"name": "红利ETF", "category": "风格"},
    "159928": {"name": "消费ETF", "category": "行业"},
    "512010": {"name": "医药ETF", "category": "行业"},
    "516160": {"name": "新能源ETF", "category": "行业"},
    "512880": {"name": "证券ETF", "category": "行业"},
    "518880": {"name": "黄金ETF", "category": "避险"},
}


def fetch_all_etf_data(start_date: str = "20200101") -> dict:
    """批量获取所有 ETF 标的历史数据"""
    data = {}
    for code, info in ETF_POOL.items():
        df = fetch_etf_hist(code, start_date=start_date)
        if not df.empty:
            data[code] = df
            print(f"  ✓ {info['name']}({code}): {len(df)} 条记录")
        else:
            print(f"  ✗ {info['name']}({code}): 无数据")
    return data


if __name__ == "__main__":
    print("=" * 50)
    print("A股量化交易系统 - 数据获取测试")
    print("=" * 50)
    data = fetch_all_etf_data("20230101")
    print(f"\n共获取 {len(data)} 只 ETF 数据")
