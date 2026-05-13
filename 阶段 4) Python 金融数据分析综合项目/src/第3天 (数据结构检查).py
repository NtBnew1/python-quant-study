'''
第3天
					数据结构检查
-查看数据字段（Open / High / Low / Close / Volume）
-设置时间索引
-统一数据格式

练习：
-输出数据基本信息（head、info、describe）
'''


import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class DataInspector:
    """数据检查器 - 只检查不修复"""
    def __init__(self):
        # 获取项目根目录
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent    # 上两级到项目根目录
        self.data_dir = self.project_root / "data" / "raw"

        self.start_time = datetime.now()

        print("=" * 60)
        print("Day 3: 数据结构检查")
        print("=" * 60)
        print("任务: 查看数据字段、检查时间索引、输出基本信息")
        print("=" * 60)

        # 检查目录是否存在
        if not self.data_dir.exists():
            print(f"❌ 数据目录不存在: {self.data_dir}")
            print("  请先运行任务2获取数据")


    def load_excel_files(self):
        """加载所有Excel文件"""
        excel_files = list(self.data_dir.glob("*_data_stock.xlsx"))
        if not excel_files:
            print("❌ 未找到数据文件")
            print(f"  请确保 {self.data_dir} 目录下有 *_data_stock.xlsx 文件")
            print(f"  请先运行任务2获取数据")
            return []
        return excel_files

    def show_basic_info(self, df, symbol):
        """显示任务要求的三项基本信息"""
        print(f"\n {symbol} - 基本信息")
        print("-" * 50)

        # 1. head() - 前5行数据
        print("1. head() - 数据预览:")
        print(df.head())

        # 2. info() - 数据信息
        print(f"\n2. info() - 数据信息:")
        print(f" 行数: {len(df)}, 列数: {len(df.columns)}")
        print(f" 内存使用: {df.memory_usage().sum() / 1024:.1f} KB")

        # 3. describe() - 统计描述
        print(f"\n3. describe() - 统计描述:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe().round(2))
        else:
            print("   没有数值列")

    def check_fields(self, df, symbol):
        """检查数据字段"""
        print(f"\n📋 {symbol} - 字段检查")
        print("-" * 50)
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        all_present = True

        for field in required_fields:
            if field in df.columns:
                dtype = df[field].dtype
                non_null = df[field].count()
                print(f"  ✅ {field}: {dtype}, {non_null}个非空值")
            else:
                print(f"  ❌ {field}: 缺失")
                all_present = False

        if all_present:
            print("  ✅ 所有必需字段都存在")
        else:
            print("  ⚠️  缺少必需字段")

        # 显示所有列
        print(f"\n 所有列({len(df.columns)}个):")
        print(f" {','.join(df.columns.tolist())}")

    def check_time_index(self, df, symbol):
        """检查时间索引"""
        print(f"\n⏰ {symbol} - 时间索引检查")
        print("-" * 50)

        if isinstance(df.index, pd.DatetimeIndex):
            print(f"  ✅ 索引类型: DatetimeIndex")

            # 检查排序
            if df.index.is_monotonic_increasing:
                print(f"  ✅ 时间已排序 (升序)")
            else:
                print(f"  ⚠️  时间未排序")

            # 检查重复
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count == 0:
                print(f"  ✅ 无重复日期")
            else:
                print(f"  ⚠️  有 {duplicate_count} 个重复日期")

            # 显示时间范围
            if len(df) > 0:
                print(f"  📅 时间范围:{df.index[0].date()}至{df.index[-1].date()}")
                print(f"  📈 总交易日: {len(df)}")

                # 检查连续性
                if len(df) > 1:
                    date_diff = df.index.to_series().diff().dropna()
                    avg_gap = date_diff.mean().days
                    max_gap = date_diff.max().days
                    print(f"  🔗 平均间隔: {avg_gap:.1f} 天")
                    if max_gap > 5:
                        print(f"  ⚠️  最大间隔: {max_gap} 天 (可能有数据缺失)")

        else:
            print(f" ❌ 索引类型: {type(df.index).__name__}")
            print(f"  💡 建议: 使用 pd.to_datetime() 转换为时间索引")

    def check_data_quality(self, df, symbol):
        """检查数据质量"""
        print(f"\n🧪 {symbol} - 数据质量检查")
        print("-" * 50)

        # 缺失值检查
        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            print(f"  ✅ 无缺失值")
        else:
            print(f"  ⚠️  总缺失值:{missing_total}个")

            # 按列显示缺失
            missing_by_col = df.isnull().sum()
            missing_cols = missing_by_col[missing_by_col > 0]
            if len(missing_cols) > 0:
                print(f"    各列缺失:")
                for col, count in missing_cols.items():
                    percent = count / len(df) * 100
                    print(f"{col}: {count}个({percent:.1f}%)")

        # 价格合理性检查
        if all(col in df.columns for col in ['high', 'low']):
            invalid_rows = df[df['high'] < df['low']]
            if len(invalid_rows) == 0:
                print(f"  ✅ 价格关系合理 (最高价 ≥ 最低价)")
            else:
                print(f"  ❌ 有 {len(invalid_rows)} 行最高价 < 最低价")

        # 收盘价检查
        if 'close'in df.columns:
            zero_or_negative = df[df['close'] <= 0]
            if len(zero_or_negative) == 0:
                print(f"  ✅ 收盘价全部为正")
            else:
                print(f"  ⚠️  有 {len(zero_or_negative)}行收盘价 ≤ 0")

    def inspect_single_file(self, filepath):
        """检查单个文件"""
        try:
            # 读取数据
            df = pd.read_excel(filepath, sheet_name='股票数据', index_col=0)
            symbol = filepath.stem.replace('_data_stock', '')

            print(f"\n{'=' * 60}")
            print(f"🔍 检查: {symbol} ({filepath.name})")
            print(f"{'=' * 60}")

            # 执行各项检查
            self.show_basic_info(df, symbol)     # 任务核心要求
            self.check_fields(df, symbol)        # 查看数据字段
            self.check_time_index(df, symbol)    # 检查时间索引
            self.check_data_quality(df, symbol)  # 数据质量
            return True
        except Exception as e:
            print(f"❌ 检查失败 {filepath.name}: {e}")
            return False

    def run_inspection(self):
        """运行检查"""
        print("\n" + "=" * 60)
        print("开始数据结构检查")
        print("=" * 60)

        # 加载文件
        excel_files = self.load_excel_files()
        if not excel_files:
            return

        print(f"找到 {len(excel_files)} 个数据文件，开始检查...")

        # 检查统计
        total_files = len(excel_files)
        successful_checks = 0

        # 逐个检查
        for i, filepath in enumerate(excel_files, 1):
            print(f"\n[{i}/{total_files}]", end="")

            if self.inspect_single_file(filepath):
                successful_checks += 1

        # 显示总结
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'=' * 60}")
        print("🎉 数据结构检查完成!")
        print(f"{'=' * 60}")

        print(f"\n📊 检查统计:")
        print(f"   总文件数: {total_files}")
        print(f"   成功检查: {successful_checks}")
        print(f"   失败检查: {total_files - successful_checks}")
        print(f"   总耗时: {total_time:.1f} 秒")

        print(f"\n💡 检查完成!")
        print(f"   已输出: head(), info(), describe()")
        print(f"   已检查: 数据字段、时间索引、数据质量")
        print(f"   所有结果已显示在控制台")
        print(f"{'=' * 60}")

# ===================== 主程序 =====================
def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("Day 3: 数据结构检查")
    print("=" * 70)
    print("功能:")
    print("  1. 查看数据字段 (Open/High/Low/Close/Volume)")
    print("  2. 检查时间索引")
    print("  3. 输出基本信息 (head/info/describe)")
    print("=" * 70)

    try:
        inspector = DataInspector()
        inspector.run_inspection()

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    # Windows下保持窗口
    import sys
    if sys.platform == "win32":
        input("\n按 Enter 键退出...")


'''
## 第3天：数据结构检查

**任务目标**：
- 查看数据字段（Open / High / Low / Close / Volume）
- 设置时间索引
- 统一数据格式
- 输出数据基本信息（head、info、describe）

**实现方案**：
1. **数据加载与路径管理**：
   - 自动定位项目根目录和data/raw文件夹
   - 批量查找并加载第2天生成的Excel数据文件
   - 支持多股票数据文件的自动化批量处理

2. **结构化检查流程**：
   - **基础信息输出**：按要求实现head()、info()、describe()方法输出
   - **字段完整性检查**：验证OHLCV（开盘、最高、最低、收盘、成交量）必需字段
   - **时间索引评估**：检查是否为DatetimeIndex、排序状态、日期连续性
   - **数据质量分析**：检查缺失值、价格逻辑关系、异常值

**核心代码结构**：
```python
class DataInspector:
    ├── __init__()               # 初始化项目路径和数据目录
    ├── load_excel_files()       # 查找并加载所有Excel数据文件
    ├── show_basic_info()        # 输出head、info、describe（任务核心要求）
    ├── check_fields()           # 检查OHLCV字段完整性
    ├── check_time_index()       # 检查时间索引类型和连续性
    ├── check_data_quality()     # 检查缺失值、价格逻辑等数据质量
    ├── inspect_single_file()    # 对单个文件执行完整检查流程
    └── run_inspection()         # 批量执行所有文件检查并汇总统计
``` 
'''










