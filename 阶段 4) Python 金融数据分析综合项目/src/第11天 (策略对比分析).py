'''
第11天
					策略对比分析
-对比均线与动量策略表现
-分析不同市场阶段的表现差异

练习：
-输出策略对比表格
'''


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

class StrategyComparator:
    """策略对比分析器"""
    def __init__(self):
        """初始化策略对比分析器"""
        print("=" * 80)
        print(f" 方法1: 初始化策略对比分析器")
        print("=" * 80)

        # 1. 获取当前文件目录
        print(f" \n1. 获取当前文件目录")
        current_dir = Path(__file__).parent
        print(f' 当前文件目录: {current_dir}')

        # 2. 找到项目根目录
        print(f" \n2. 找到项目根目录")
        self.project_root = current_dir.parent
        print(f" 项目根目录: {self.project_root}")

        # 3. 设置数据目录 (获取均线和动量数据: 下面就是均线和动量)
        print(f" \n3. 设置数据目录")
        self.ma_result_dir = self.project_root / "data" / "策略结果"
        self.momentum_result_dir = self.project_root / "data" / "动量策略结果"
        print(f" 均线策略结果目录: {self.ma_result_dir}")
        print(f" 动量策略结果目录: {self.momentum_result_dir}")

        # 4. 设置输出目录
        print(f' \n4. 设置输出目录')
        self.output_dir = self.project_root / "data" / "策略对比"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f" 输出目录: {self.output_dir}")

        # 5. 设置图表目录
        print(f" \n5. 设置图表目录")
        self.charts_dir = self.project_root / "charts" / "策略对比"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        print(f" 图表目录: {self.charts_dir}")

        # 6. 设置中文字体
        print(f" \n6. 设置中文字体")
        self._setup_chinese_font()

        # 7. 初始化结果存储
        print(f" \n7. 初始化结果存储")
        self.comparison_results = []

        print(f'\n' + "=" * 70)
        print(f" 方法1完成: 初始化成功")
        print(f'=' * 70)
        print(f" 均线策略目录: {self.ma_result_dir}")
        print(f" 动量策略目录: {self.momentum_result_dir}")
        print(f" 输出目录: {self.output_dir}")
        print(f" 图表目录: {self.charts_dir}")

    # 下面的代码是用绘制图表的时候使用.  很经常绘制图表没办法显示中文. 所以需要下面这些代码.
    def _setup_chinese_font(self):
        """配置中文字体"""
        import matplotlib.font_manager as fm
        import os

        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf'
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                    font_name = fm.FontProperties(fname=path).get_name()
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f" 中文字体配置成功: {font_name}")
                    return
                except:
                    continue
        print(f" 未找到中文字体, 使用默认字体")
        plt.rcParams['axes.unicode_minus'] = False




# 调用测试
if __name__ == "__main__":
    comparator = StrategyComparator()








