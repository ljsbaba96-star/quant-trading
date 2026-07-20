"""
数据获取 v2 — 增加重试、备用接口、ETF扩展池
"""

import akshare as ak
import pandas as pd
import time
import os
import sqlite3
from datetime import datetime, timedelta

# 绕过代理（本地直连）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

DB_PATH = os.path.join(os.path.dirname(__file__), "market_data.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS kline (
        code TEXT, date TEXT, open REAL, high REAL, low REAL,
        close REAL, volume REAL, amount REAL,
        PRIMARY KEY (code, date))""")
    conn.commit()
    return conn


def save_to_db(code, df):
    conn = get_db_conn()
    records = []
    for _, row in df.iterrows():
        records.append((code, str(row["日期"]), float(row["开盘"]),
                        float(row["最高"]), float(row["最低"]),
                        float(row["收盘"]), float(row["成交量"]),
                        float(row["成交额"])))
    conn.executemany("INSERT OR REPLACE INTO kline VALUES (?,?,?,?,?,?,?,?)", records)
    conn.commit()
    conn.close()


def load_from_db(code, start_date=None, end_date=None):
    conn = get_db_conn()
    q = "SELECT * FROM kline WHERE code=?"
    p = [code]
    if start_date: q += " AND date>=?"; p.append(start_date)
    if end_date: q += " AND date<=?"; p.append(end_date)
    q += " ORDER BY date"
    df = pd.read_sql_query(q, conn, params=p)
    conn.close()
    if not df.empty:
        df.rename(columns={"code": "代码", "date": "日期", "open": "开盘",
                           "high": "最高", "low": "最低", "close": "收盘",
                           "volume": "成交量", "amount": "成交额"}, inplace=True)
    return df


def fetch_etf(symbol, start_date="20200101", end_date=None, retries=3):
    """带重试的ETF数据获取"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 先查缓存
    cached = load_from_db(symbol, start_date, end_date)
    if not cached.empty:
        last = cached["日期"].iloc[-1]
        if last >= (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"):
            return cached

    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                      start_date=start_date, end_date=end_date,
                                      adjust="qfq")
            if df is not None and not df.empty:
                save_to_db(symbol, df)
                time.sleep(1.5)  # 更长的间隔避免限流
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  ✗ {symbol}: {e}")

    return load_from_db(symbol, start_date, end_date)


def fetch_stock(symbol, start_date="20200101", end_date=None, retries=3):
    """带重试的股票数据获取"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cached = load_from_db(symbol, start_date, end_date)
    if not cached.empty:
        last = cached["日期"].iloc[-1]
        if last >= (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"):
            return cached

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                     start_date=start_date, end_date=end_date,
                                     adjust="qfq")
            if df is not None and not df.empty:
                save_to_db(symbol, df)
                time.sleep(1.5)
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  ✗ {symbol}: {e}")

    return load_from_db(symbol, start_date, end_date)


# ============================================================
# 扩展ETF池（覆盖更多风格和行业）
# ============================================================
ETF_POOL = {
    # 宽基
    "510300": {"name": "沪深300ETF", "cat": "宽基"},
    "510500": {"name": "中证500ETF", "cat": "宽基"},
    "159915": {"name": "创业板ETF", "cat": "宽基"},
    "588000": {"name": "科创50ETF", "cat": "宽基"},
    "510050": {"name": "上证50ETF", "cat": "宽基"},
    "512100": {"name": "中证1000ETF", "cat": "宽基"},
    # 行业
    "159928": {"name": "消费ETF", "cat": "行业"},
    "512010": {"name": "医药ETF", "cat": "行业"},
    "516160": {"name": "新能源ETF", "cat": "行业"},
    "512880": {"name": "证券ETF", "cat": "行业"},
    "515030": {"name": "新能源车ETF", "cat": "行业"},
    "512690": {"name": "酒ETF", "cat": "行业"},
    "159869": {"name": "游戏ETF", "cat": "行业"},
    "512660": {"name": "军工ETF", "cat": "行业"},
    "515790": {"name": "光伏ETF", "cat": "行业"},
    # 风格
    "510880": {"name": "红利ETF", "cat": "风格"},
    "512890": {"name": "红利低波ETF", "cat": "风格"},
    # 避险
    "518880": {"name": "黄金ETF", "cat": "避险"},
    "511010": {"name": "国债ETF", "cat": "避险"},
}


def fetch_etf_pool(start_date="20210101", codes=None) -> dict:
    """批量获取ETF数据"""
    target = codes or list(ETF_POOL.keys())
    data = {}
    for code in target:
        info = ETF_POOL.get(code, {})
        df = fetch_etf(code, start_date=start_date)
        if not df.empty:
            data[code] = df
            print(f"  ✓ {info.get('name', code)}({code}): {len(df)}条")
        else:
            print(f"  ✗ {info.get('name', code)}({code}): 无数据")
    return data
