'''
第1天：
学习投资组合的基础理论，理解预期收益和风险（标准差）的计算。
练习：用pandas-datareader 获取多资产历史价格数据，计算日收益率及协方差矩阵。
'''


import pandas as pd
import numpy as np
from pandas_datareader import data as web
import datetime as dt

# 用pandas-dataread 来获取数据比较容易.  yfinance获取更加容易只是这个库不是很稳定. 很经常被封锁IP.
def portfolio_analysis(tickers, start_date, end_date):
    # 想获取数据
    # 将输入的起始日期和结束日期转换为pandas的日期时间格式
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # 下载数据
    # 创建一个空的DataFrame用于存储价格数据
    price_data = pd.DataFrame()
    # 遍历每个股票代码
    for ticker in tickers:
        try:
            # 从stooq数据源获取股票数据
            df = web.DataReader(ticker, data_source='stooq', start=start, end=end)
            # 将收盘价数据添加到price_data中
            price_data[ticker] = df['Close']        # 只获取股价
        except Exception as e:
            # 如果获取数据失败，打印错误信息
            print(f"无法获取: {e}")

    # 如果为空则退出
    # 检查是否成功获取到任何数据
    if price_data.empty:
        print(f"未获取任何股票数据, 请检查代码或者日期范围.")
        return

    # stooq 返回倒序数据, 需要反转
    # 将数据顺序反转，使其按时间正序排列
    price_data = price_data[::-1]

    # 计算每日收益率
    # 使用pct_change计算每日价格变动百分比，然后删除包含NaN的行
    returns = price_data.pct_change().dropna()

    # 年化收益与协方差矩阵
    # 计算年化预期收益率（假设一年有252个交易日）
    expected_returns = returns.mean() * 252
    # 计算年化协方差矩阵
    cov_matrix = returns.cov() * 252

    # 假设等权重
    # 创建等权重投资组合（每只股票权重相同）
    weights = np.array([1/len(tickers)] * len(tickers))

    # 投资组合收益与风险
    # 计算投资组合的预期收益：权重向量与预期收益向量的点积
    portfolio_return = np.dot(weights, expected_returns)
    # 计算投资组合的波动率：sqrt(w^T * Σ * w)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 输出
    print(f"===投资组合分析结果===")
    print(f"股票列表: {tickers}")
    print(f"时间范围: {start_date} -> {end_date}")
    print(f"\n 资产预期收益率 (年化): {expected_returns.round(4)}")
    print(f"\n协方差矩阵 (年化): {cov_matrix.round(5)}")
    print(f"\n投资组合预期收益: {portfolio_return:.4f}")
    print(f"\n投资组合风险 (标准差): {portfolio_volatility:.4f}")

    # 返回计算结果
    return expected_returns, cov_matrix, portfolio_return, portfolio_volatility

if __name__ == "__main__":
    # 获取用户输入的股票代码列表（用逗号分隔）
    tickers = input(f"请输入股票代码").split(',')
    # 获取用户输入的起始日期
    start_date = input(f"请输入开始日期: (格式: YYYY-MM-DD):")
    # 获取用户输入的结束日期
    end_date = input(f"请输入结束日期: (格式: YYYY-MM-DD):")

    # 调用投资组合分析函数
    # 对输入的股票代码进行处理：去除空格并转换为大写
    portfolio_analysis([t.strip().upper() for t in tickers], start_date, end_date)

'''
===========================总结=======================
在代码中，我使用了 pandas-datareader 从 Stooq 数据源 获取多只股票的历史价格数据。
相比 yfinance，它不容易被封锁，数据也较为稳定。

获取到收盘价后，我计算了每日收益率，并进一步求出了：

年化预期收益率：表示每只资产在一年的平均收益水平；

协方差矩阵：反映不同资产之间的相关性，是衡量组合风险的重要工具。

在假设每只股票等权重的情况下，利用矩阵运算计算出了投资组合的整体预期收益与风险（标准差）。
从结果中可以看出，投资组合的风险并不是各个资产风险的简单平均，而是受到它们之间相关性影响的。
若资产之间的相关性较低或为负，可以有效降低整体波动，实现"分散风险"的效果。
'''