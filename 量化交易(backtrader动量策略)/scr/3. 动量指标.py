"""
第3天：动量指标

1. 文件名：3. 动量指标.py
2. 输入：data/processed/*.csv
3. 输出：data/signals/*.csv

4. 步骤：
   - 创建 data/signals/ 目录
   - 读取清洗后数据
   - 计算动量10天、20天、60天
   - 生成3种信号：
     a. 信号20 vs 60（加阈值5%）
     b. 信号10 vs 60（加阈值5%）
     c. 信号10_20_60（三线多头）
   - 保存信号数据

5. 测试：
   - 检查动量值范围
   - 检查信号值（1, 0, -1）
   - 统计买入/卖出/观望天数
"""

import pandas as pd
import numpy as np
from pathlib import Path

class MomentumIndicator:
    """动量指标计算"""
    def __init__(self):
        print(f'\n 计算动量指标')
        print('=' * 50)

        # 设置目录
        current_dir = Path(__file__).parent
        self.project_dir = current_dir.parent
        self.processed_dir = self.project_dir / "data" / "processed"
        self.signals_dir = self.project_dir / "data" / 'signals'
        self.signals_dir.mkdir(parents=True, exist_ok=True)

        print(f" 输入目录: {self.processed_dir}")
        print(f" 输出目录: {self.signals_dir}")

        # 查找文件
        self.files = list(self.processed_dir.glob("*.csv"))
        print(f" 找到 {len(self.files)}个数据文件")

    def load_data(self, file_path):
        """加载数据"""
        symbol = file_path.stem
        print(f" \n 处理; {symbol}")

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f" 数据行数: {len(df)}")
        return df

    def cal_momentum(self, df):
        """计算动量10, 20, 60"""
        print(f" \n 计算动量指标............")

        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1

        print(f" Momentum_10: {df['momentum_10'].min():.2%} ~ {df['momentum_10'].max():.2%}")
        print(f" Momentum_20: {df['momentum_20'].min():.2%} ~ {df['momentum_20'].max():.2%}")
        print(f" Momentum_60: {df['momentum_60'].min():.2%} ~ {df['momentum_60'].max():.2%}")

        return df

    def signal_10_60(self, df):
        """信号: 动量10 vs 动量60 (加阈值5%) """
        print(f" \n 信号10 vs 60............")

        df['signal_10_60'] = 0
        df.loc[(df['momentum_10'] > df['momentum_60']) & (df['momentum_10'] > 0.05), 'signal_10_60'] = 1
        df.loc[(df['momentum_10'] < df['momentum_60']) & (df['momentum_10'] < -0.05), 'signal_10_60'] = -1

        buy = (df['signal_10_60'] == 1).sum()
        sell = (df['signal_10_60'] == -1).sum()
        hold = (df['signal_10_60'] == 0).sum()

        print(f" 买入: {buy} 天, 卖出: {sell}天, 观望:{hold}天")
        return df

    def signal_20_60(self, df):
        """信号: 动量20 vs 动量60 (加阈值5%)"""
        print(f" \n 信号20 vs 60.............")

        df['signal_20_60'] = 0
        df.loc[(df['momentum_20'] > df['momentum_60']) & (df['momentum_20'] > 0.05), 'signal_20_60'] = 1
        df.loc[(df['momentum_20'] < df['momentum_60']) & (df['momentum_20'] < -0.05), 'signal_20_60'] = -1

        buy = (df['signal_20_60'] == 1).sum()
        sell = (df['signal_20_60'] == -1).sum()
        hold = (df['signal_20_60'] == 0).sum()

        print(f' 买入: {buy}天, 卖出: {sell}天, 观望; {hold}天')

        return df

    def signal_10_20_60(self, df):
        """三线多头: 10 > 20 > 60买入, 10 < 20 < 60卖出 """
        print(f" \n 三线多头 (10>20>60)............")

        df['signal_10_20_60'] = 0
        df.loc[(df['momentum_10']>df['momentum_20'])&(df['momentum_20']>df['momentum_60']), 'signal_10_20_60'] = 1
        df.loc[(df['momentum_10']<df['momentum_20'])&(df['momentum_20']<df['momentum_60']), 'signal_10_20_60'] = -1

        buy = (df['signal_10_20_60'] == 1).sum()
        sell = (df['signal_10_20_60'] == -1).sum()
        hold = (df['signal_10_20_60'] == 0).sum()
        print(f" 买入: {buy}天, 卖出: {sell}天, 观望: {hold}天")
        return df

    def process_stock(self, file_path):
        """处理单个股票"""
        symbol = file_path.stem
        print(f" 处理: {symbol}")
        print(f' \n' + '=' * 50)

        df = self.load_data(file_path)
        df = self.cal_momentum(df)
        df = self.signal_10_60(df)
        df = self.signal_20_60(df)
        df = self.signal_10_20_60(df)

        # 保存
        output_path = self.signals_dir / f"{symbol}.csv"
        df.to_csv(output_path, index=False)
        print(f" \n 已保存: {output_path.name}")
        return df

    def run(self):
        """运行所有股票"""
        print(f" \n 开始计算动量指标............")
        for file_path in self.files:
            self.process_stock(file_path)

        print(f"\n" + '=' * 80)
        print(f" 计算动量指标完成")
        print(f" 数据保存在: {self.signals_dir}")
        print(f'=' * 80)

if __name__ == "__main__":
    indicator = MomentumIndicator()
    indicator.run()








