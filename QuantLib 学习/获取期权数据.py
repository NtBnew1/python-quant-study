"""
获取股票期权数据（Calls 和 Puts）
支持多个股票代码，每个股票单独保存到 Excel 文件
"""

import pandas as pd
from yahooquery import Ticker
import datetime as dt

# ==============================
# 1️⃣ 输入股票代码
# ==============================
# 输入格式示例: AAPL, TSLA, MSFT
symbol_input = input("请输入期权股票代码 (用逗号分开): ")
symbols = [s.strip().upper() for s in symbol_input.split(",")]  # 转大写 & 去空格

# 定义今天日期
today = dt.datetime.today()

# ==============================
# 2️⃣ 循环处理每个股票
# ==============================
for symbol in symbols:
    print(f"\n========= 处理 {symbol} =========")
    ticker = Ticker(symbol)

    try:
        # -----------------------------
        # 获取期权链（option_chain）
        # 返回 DataFrame，索引是 (symbol, expiration, optionType)
        # 列包括 strike, lastPrice, bid, ask, volume, openInterest 等
        # -----------------------------
        opt_chain = ticker.option_chain
        if opt_chain is None or opt_chain.empty:
            print(f"⚠️ {symbol} 没有获取到期权数据")
            continue

        print("期权链前5行数据:")
        print(opt_chain.head())

        # -----------------------------
        # 筛选 Calls 和 Puts
        # -----------------------------
        calls = opt_chain[opt_chain.index.get_level_values('optionType') == 'calls']
        puts = opt_chain[opt_chain.index.get_level_values('optionType') == 'puts']

        # -----------------------------
        # 筛选未来 3 个月以上到期的期权
        # -----------------------------
        future_options = opt_chain[
            opt_chain.index.get_level_values("expiration") > (today + pd.Timedelta(days=90))
        ]

        # -----------------------------
        # 选择一个示例期权（第一条未来期权）
        # -----------------------------
        if not future_options.empty:
            first_row = future_options.iloc[0]   # 取第一行
            idx = first_row.name                 # 索引是 (symbol, expiration, optionType)
            expiration = pd.to_datetime(idx[1])  # 到期日
            option_type = idx[2]                 # calls 或 puts
            K = first_row['strike']              # 执行价
            market_price = first_row['lastPrice']  # 市场价格
            print(f"示例期权: {symbol} {option_type} 到期日: {expiration.date()}, "
                  f"执行价: {K}, 市场价: {market_price}")
        else:
            print(f"⚠️ 没有找到未来 90 天以上的期权")

        # -----------------------------
        # 保存到 Excel
        # 每个股票单独一个文件，包含两个 Sheet (Calls / Puts)
        # -----------------------------
        output_file = f"{symbol}_options.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            calls.reset_index().to_excel(writer, sheet_name='Calls', index=False)
            puts.reset_index().to_excel(writer, sheet_name='Puts', index=False)

        print(f"✅ {symbol} 所有期权数据已保存到 {output_file}")

    except Exception as e:
        print(f"❌ 获取 {symbol} 期权数据失败: {e}")


'''
=========================总结=================
这段代码能批量获取用户输入股票的期权数据，
分别提取 Calls 和 Puts，并保存到每个股票的独立 Excel 文件中。
同时，它会筛选未来 90 天以上的期权，并展示一个示例（到期日、执行价、市场价）。
'''