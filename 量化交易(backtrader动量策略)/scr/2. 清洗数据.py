"""
第2天：清洗数据
1. 文件名：2. 清洗数据.py
2. 输入：data/raw/*.csv
3. 输出：data/processed/*.csv

4. 步骤：
--创建 data/processed/ 目录
--读取 CSV 数据
--转换日期格式
--检查缺失值
--计算日收益率
--保存清洗后数据

5. 测试：
    1. 检查缺失值
    2. 检查收益率范围
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    """清洗数据"""
    def __init__(self):
        """初始化"""
        print(f" 清洗数据")
        print(f'=' * 60)

        # 设置目录
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        print(f" 原始数据目录: {self.raw_dir}")
        print(f" 清洗后目录: {self.processed_dir}")

        # 查找原始文件
        self.raw_files = list(self.raw_dir.glob("*.csv"))
        print(f" 找到 {len(self.raw_files)} 个原始文件")

    def clean_stock(self, file_path):
        """清洗单个股票数据"""
        symbol = file_path.stem
        print(f" \n 清洗: {symbol}")

        # 1. 读取数据
        df = pd.read_csv(file_path)
        print(f" 原始数据: {len(df)}行")

        # 2. 转换 Close 列为数字 (解决字符串问题)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # 3. 转换时间
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # 4.. 检查缺失值
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f" 缺失值: {null_count} 个")
            df = df.dropna()
            print(f" 删除后: {len(df)} 行")

        # 5.. 计算日收益率
        df['daily_return'] = df['Close'].pct_change()
        df['daily_return'] = df['daily_return'].fillna(0)

        # 6. 只保留需要的列
        df = df[['Date', 'Close', 'daily_return']]

        # 7. 保存
        output_path = self.processed_dir / f"{symbol}.csv"
        df.to_csv(output_path, index=False)
        print(f" 已保存: {output_path.name} ({len(df)} 行)")

        return df

    def run(self):
        """清洗所有股票"""
        print(f" \n 开始清洗...............")
        print(f'=' * 50)

        for file_path in self.raw_files:
            self.clean_stock(file_path)

        print(f"\n" + '=' * 50)
        print(f" 清洗数据完成")
        print(f" 数据保存在: {self.processed_dir}")
        print(f" =" * 60)

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()







