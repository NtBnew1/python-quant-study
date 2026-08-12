"""
第1天：获取数据
1. 文件名：1. 获取数据.py
2. 输入：股票列表（AAPL, MSFT, GOOGL, AMZN, TSLA）
3. 时间范围：2020-01-01 到 今天
4. 输出：data/raw/*.csv

5. 步骤：
--创建 data/raw/ 目录
--定义股票列表
--用 yfinance 下载数据
--保存为 CSV

6. 测试：
1. 检查文件是否存在
2. 检查数据行数

"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class DataFetcher:
    """获取股票数据"""
    def __init__(self):
        """初始化设置"""
        print(f'\n 获取数据')
        print('=' * 50)

        # 1. 设置保存数据的目录
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent
        self.raw_dir = self.project_root / "data" / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        print(f" {current_dir}")
        print(f' 数据目录: {self.raw_dir}')

        # 用户输入股票代码
        stock_input = input(f" 请输入股票代码 (用逗号分隔, 如AAPL, NVDA, GOOGL): ")
        self.stocks = [s.strip().upper() for s in stock_input.split(',')]
        print(f" 股票列表: {self.stocks}")

        # 用户输入开始日期
        self.start_date = input(f" 请输入开始日期 (如 2020-01-01): ")

        # 结束日期为当天
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        print(f' 开始: {self.start_date}')
        print(f" 结束: {self.end_date}")

    def run(self):
        """下载所有股票数据"""
        print(f" \n 开始下载........")
        print(f'-' * 50)

        for symbol in self.stocks:
            print(f" \n 下载: {symbol}")
            try:
                df = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if df.empty:
                    print(f" 无数据")
                    continue
                df = df.reset_index()
                file_path = self.raw_dir / f"{symbol}.csv"
                df.to_csv(file_path, index=False)

                print(f" 已保存: {file_path.name} ({len(df)} 行)")
            except Exception as e:
                print(f" 失败: {e}")

        print(f" \n" + "=" * 80)
        print(f" 数据保存完成:{self.raw_dir}")
        print(f" =" * 60)


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run()




