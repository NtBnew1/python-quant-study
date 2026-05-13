'''
第4天
					数据清洗
-处理缺失值
-处理异常值
-保存清洗后的数据

练习：
-对比清洗前后数据差异
'''

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, Optional, Tuple, List
import sys


warnings.filterwarnings('ignore')

class SmartDataCleaner:
    '''智能数据清洗器'''
    def __init__(self, config: Dict = None):
        print("=" * 70)
        print("智能数据清洗系统")
        print("=" * 70)

        # 获取当前脚本所在目录的父目录作为项目根目录
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent       # 上一级目录（项目根目录）
        print(f"项目根目录: {self.project_root}")

        # 配置 - 使用您现有的目录结构
        self.config = {
            'project_root': self.project_root,
            'raw_dir': self.project_root / "data" / "raw",      # 原始数据目录
            'clean_dir': self.project_root / "data" / "clean",  # 清洗数据目录
            'report_dir': self.project_root / "report",          # 报告目录（单数）
            'min_data_points': 30,
        }

        if config:
            self.config.update(config)

        # 检查并创建目录
        self.check_and_create_directories()

        # 查找文件
        self.files = self.find_data_files()
        self.results = []

    def check_and_create_directories(self):
        """检查并创建必要的目录"""
        print("\n📁 检查目录结构:")

        # 首先检查现有目录
        directories = [
            ('原始数据', self.config['raw_dir']),
            ('清洗数据', self.config['clean_dir']),
            ('报告目录', self.config['report_dir'])
        ]

        for name, dir_path in directories:
            if dir_path.exists():
                file_count = len(list(dir_path.glob('*')))
                print(f"{name}: {dir_path.relative_to(self.project_root)}/ (已有{file_count}个文件)")

            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"创建: {dir_path.relative_to(self.project_root)}/")
                except Exception as e:
                    print(f"创建失败 {name}: {e}")
                    # 如果报告目录创建失败，尝试在项目根目录创建
                    if name == "报告目录":
                        alt_dir = self.project_root / "report"
                        print(f" ⚠️  尝试使用备选路径: {alt_dir.relative_to(self.project_root)}")
                        alt_dir.mkdir(parents=True, exist_ok=True)
                        self.config['report_dir'] = alt_dir
        # 特别显示原始数据目录内容
        raw_dir = self.config['raw_dir']
        if raw_dir.exists():
            raw_files = list(raw_dir.glob('*.xlsx')) + list(raw_dir.glob('*.xls')) + list(raw_dir.glob('*.csv'))
            print(f"\n🔍 原始数据目录扫描: {raw_dir.relative_to(self.project_root)}/")
            print(f"   找到 {len(raw_files)} 个数据文件")

    def find_data_files(self) -> List[Path]:
        """查找数据文件"""
        raw_dir = self.config['raw_dir']

        if not raw_dir.exists():
            print(f"\n❌ 错误: 原始数据目录不存在")
            print(f" 请检查路径: {raw_dir.relative_to(self.project_root)}")
            return []

        # 查找Excel文件
        files = list(raw_dir.glob('*.xlsx'))+list(raw_dir.glob('*.xls'))+list(raw_dir.glob('*.csv'))
        files = [f for f in files if 'cleaned' not in f.name.lower() and 'verified' not in f.name.lower()]

        if not files:
            print(f"\n❌ 在 {raw_dir.relative_to(self.project_root)} 中没有找到数据文件")
            print(f"   支持的文件格式: Excel (.xlsx, .xls), CSV (.csv)")

            # 显示目录中的其他文件（帮助调试）
            all_files = list(raw_dir.glob('*'))
            if all_files:
                print(f"   目录中的文件:")
                for file in all_files[:10]:
                    print(f"     - {file.name}")
            return []

        print(f"\n✅ 找到 {len(files)} 个股票数据文件:")
        # 显示文件列表（最多显示10个）
        for i, file in enumerate(files[:10], 1):
            try:
                size_kb = file.stat().st_size / 1024
                print(f"{i:2d}. {file.name:<30} ({size_kb:.1f} KB)")
            except:
                print(f"{i:2d}.{file.name}")

        if len(files) > 10:
            print(f"  ... 还有 {len(files) - 10} 个文件")
        return files

    def extract_symbol(self, filename: str) -> str:
        """提取股票代码"""
        name = Path(filename).stem
        # 移除常见后缀
        suffixes = ['_data', '_stock', '_data_stock', '_stock_data', '_daily', '_price']
        for suffix in suffixes:
            name = name.replace(suffix, '')

        # 提取股票代码
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[0]

        else:
            # 如果不是标准格式，返回文件名（去掉扩展名）
            return name

    def analyze_data_quality(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
               分析数据质量 - 只报告问题
               """
        issues = []

        # 1. 检查缺失值
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                issues.append({
                    'type': 'missing',
                    'column': col,
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                })
        # 2. 检查重复行
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'duplicate',
                'count': int(duplicates),
                'percentage': duplicates / len(df) * 100
            })
        # 3. 检查价格数据逻辑（如果有）
        price_cols = ['open', 'high', 'low', 'close']
        price_cols= [col for col in price_cols if col in df.columns]

        if len(price_cols) >= 4:  # 确保有所有价格列
            logic_errors = 0
            for idx, row in df.iterrows():
                if (row['low'] > row['open'] or
                row['low'] > row['close'] or
                row['high'] < row['open'] or
                row['high'] < row['close']):
                    logic_errors += 1

            if logic_errors > 0:
                issues.append({
                    'type': 'price_logic',
                    'count': logic_errors
                })
        return issues

    def display_analysis_results(self, df: pd.DataFrame, symbol: str, issues: List[Dict]):
        """显示分析结果"""
        print(f"\n🔍 {symbol} 数据分析:")
        print(f"  数据形状: {len(df)} 行 × {len(df.columns)} 列")

        # 显示列名（前5列）
        columns = df.columns.tolist()
        columns_str = ', '.join(columns[:5])
        if len(columns) > 5:
            columns_str += f", ...(+{len(columns)-5}列)"
        print(f"  列名: {columns_str}")

        if not issues:
            print(f"  ✅ 数据质量良好")
        else:
            # 显示问题
            for issue in issues:
                if issue['type'] == 'missing':
                    print(f"  ⚠️  {issue['column']}: {issue['count']}个缺失值 ({issue['percentage']:.1f}%)")
                elif issue['type'] == 'duplicate':
                    print(f"  ⚠️  重复行: {issue['count']}行 ({issue['percentage']:.1f}%)")
                elif issue['type'] == 'price_logic':
                    print(f"  ⚠️  价格逻辑错误: {issue['count']}处")
        # 额外分析：显示数据概况
        if 'close' in df.columns:
            print(f"  📊 收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
            print(f"  📊 平均收盘价: {df['close'].mean():.2f}")

    def smart_clean_data(self, df: pd.DataFrame, issues: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
        """
                智能清理数据 - 只处理真正的问题
                """
        df_cleaned = df.copy()
        actions =[]

        for issue in issues:
            if issue['type'] == 'missing':
                col = issue['column']
                # 价格和成交量数据特殊处理
                price_volume_cols = ['close', 'open', 'high', 'low', 'volume', 'Volume', 'VOLUME']
                if any(pv_col.lower() == col.lower() for pv_col in price_volume_cols):
                    # 只填充确实缺失的值
                    missing_mask = df[col].isnull()
                    if missing_mask.any():
                        # 前向填充
                        df_cleaned.loc[missing_mask, col] = df[col].ffill()[missing_mask]
                        # 如果开头还有缺失，后向填充
                        still_missing = df_cleaned[col].isnull()
                        if still_missing.any():
                            df_cleaned.loc[still_missing, col] = df_cleaned[col].bfill()[still_missing]

                        filled_count = missing_mask.sum() - df_cleaned[col].isnull().sum()
                        if filled_count > 0:
                            actions.append(f"填充{col} {filled_count}个缺失值")
                else:
                    # 其他列：中位数填充（数值列）或众数填充（分类列）
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                        missing_count = df[col].isnull().sum()
                        df_cleaned[col].fillna(fill_value, inplace=True)
                        if missing_count > 0:
                            actions.append(f"填充{col} {missing_count}个缺失值（中位数）")
            elif issue['type'] == 'duplicate':
                before = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates()
                after = len(df_cleaned)
                removed = before - after
                if removed > 0:
                    actions.append(f"删除{removed}行重复数据")
            elif issue['type'] == 'price_logic':
                # 修正价格逻辑
                corrections = 0
                price_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in price_cols):
                    for idx, row in df.iterrows():
                        open_val = row['open']
                        high_val = row['high']
                        low_val = row['low']
                        close_val = row['close']

                        # 确保 low <= min(open, close) 且 high >= max(open, close)
                        correct_low = min(open_val, close_val, low_val)
                        correct_high = max(open_val, close_val, low_val)

                        if low_val != correct_low or high_val != correct_high:
                            df_cleaned.at[idx, 'low'] = correct_low
                            df_cleaned.at[idx, 'high'] = correct_high
                            corrections += 1
                    if corrections > 0:
                        actions.append(f"修正{corrections}处价格逻辑")
        return df_cleaned, actions

    def save_cleaned_data(self, df: pd.DataFrame, symbol: str,
                          original_info: Dict, actions: List[str]) -> Optional[Path]:
        """
                保存清洗后的数据
                """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d")

            if actions:  # 有清理操作
                filename = f"{symbol}_cleaned_{timestamp}.xlsx"
                status =  "已清理"
            else:
                filename = f"{symbol}_verified_{timestamp}.xlsx"
                status = "已验证"

            filepath = self.config['clean_dir'] / filename
            print(f"  💾 保存到: data/clean/{filename}")

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 1. 数据sheet
                df.to_excel(writer, sheet_name='股票数据', index=False)

                # 2. 清洗信息sheet
                info_data = [
                    ['股票代码', symbol],
                    ['原始文件', original_info.get('filename', '未知')],
                    ['处理状态', status],
                    ['处理时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['原始行数', original_info.get('original_rows', len(df))],
                    ['处理后行数', len(df)],
                    ['数据列数', len(df.columns)],
                    ['清理操作', '; '.join(actions) if actions else '无'],
                    ['保存路径', str(filepath)],
                    ['保存目录', 'data/clean/']
                ]
                info_df = pd.DataFrame(info_data, columns=['项目', '值'])
                info_df.to_excel(writer, sheet_name='处理信息', index=False)

            # 文件大小
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ 保存成功 ({file_size_mb:.2f} MB)")
            return filepath
        except Exception as e:
            print(f"  ❌ 保存失败: {e}")
            return None

    def process_file(self, filepath: Path) -> Optional[Dict]:
        """处理单个文件"""
        symbol = self.extract_symbol(filepath.name)
        print(f"\n" + "=" * 60)
        print(f"📈 处理: {symbol}")
        print(f"📄 文件: {filepath.name}")
        print("=" * 60)

        try:
            # 1. 加载数据
            print(f"  正在加载数据...")

            # 根据文件扩展名选择读取方式
            if filepath.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            else:
                print(f"  ❌ 不支持的文件格式: {filepath.suffix}")
                return None

            df_original = df.copy()
            # 记录原始信息
            original_info = {
                'filename': filepath.name,
                'original_rows': len(df),
                'columns': list(df.columns),
                'file_size_kb': filepath.stat().st_size / 1024
            }

            # 2. 分析数据质量
            print(f"  分析数据质量...")
            issues = self.analyze_data_quality(df, symbol)
            self.display_analysis_results(df, symbol, issues)

            # 3. 智能清理
            print(f"  执行智能清理...")
            df_cleaned, actions = self.smart_clean_data(df, issues)

            # 4. 检查是否有变化
            data_changed = False
            if actions:
                print(f"  🔧 执行操作: {', '.join(actions)}")
                data_changed = True
            else:
                print(f"  ✅ 数据良好，无需清理")

            # 5. 保存数据
            saved_path = self.save_cleaned_data(df_cleaned, symbol, original_info, actions)

            if not saved_path:
                return None

            # 6. 返回结果
            return {
                'symbol': symbol,
                'status': 'success',
                'original_rows': len(df_original),
                'cleaned_rows': len(df_cleaned),
                'file_path': str(saved_path),
                'data_changed': data_changed,
                'actions': actions,
                'issues_count': len(issues),
                'filename': filepath.name
            }
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'filename': filepath.name
            }

    def run_analysis(self):
        """运行分析流程"""
        if not self.files:
            print(f"\n❌ 没有找到数据文件")
            print(f"   请将数据文件放入: {self.config['raw_dir'].relative_to(self.project.root)}")
            return

        print(f"\n📊 将分析 {len(self.files)} 个文件")
        print("=" * 60)

        # 自动开始，不需要确认
        print("自动开始分析数据...")

        # 处理每个文件
        for i, file in enumerate(self.files, 1):
            print(f"\n[{i}/{len(self.files)}] ", end="")
            result = self.process_file(file)
            if result:
                self.results.append(result)

        # 生成报告
        self.generate_summary_report()

    def generate_summary_report(self):
        """生成总结报告"""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        print("\n" + "=" * 70)
        print("📋 数据质量分析报告")
        print("=" * 70)

        print(f"分析文件总数: {len(self.results)}")
        print(f"成功处理: {len(successful)}")
        print(f"处理失败: {len(failed)}")

        if successful:
            # 统计需要清理的文件
            cleaned_files = [r for r in successful if r.get('data_cleaned', False)]
            verified_files = [r for r in successful if not r.get('data_cleaned', False)]

            print(f"\n📊 处理结果:")
            print(f"  需要清理的文件: {len(cleaned_files)}")
            print(f"  数据良好的文件: {len(verified_files)}")

            if cleaned_files:
                print(f"\n🔧 执行了清理操作的文件（前5个）:")
                for i, result in enumerate(cleaned_files[:5], 1):
                    print(f"  {i:2d}. {result['symbol']} ({result['filename']})")
                    for action in result['actions']:
                        print(f"      → {action}")
            if verified_files:
                print(f"\n✅ 数据质量良好的文件（前5个）:")
                for i, result in enumerate(verified_files[:5], 1):
                    issues = result.get('issues_count', 0)
                    print(f"  {i:2d}. {result['symbol']} ({result['filename']}) - {issues}个问题但无需清理")
            # 保存详细报告到report目录
            self.save_detailed_report(successful)

        if failed:
            print(f"\n❌ 处理失败的文件:")
            for i, result in enumerate(failed, 1):
                print(f"  {i:2d}. {result.get('symbol', '未知')} ({result.get('filename', '未知')})")
                print(f"      错误: {result.get('error', '未知错误')}")
        print(f"\n💡 重要信息:")
        print(f"  原始数据目录: data/raw/")
        print(f"  清洗后数据: data/clean/")
        print(f"  报告目录: report/")
        print("=" * 70)

    def save_detailed_report(self, successful_result: List[Dict]):
        """保存详细报告到现有的report目录"""
        try:
            report_data = []
            for result in successful_result:
                report_data.append({
                    '股票代码': result['symbol'],
                    '原文件名': result.get('filename', '未知'),
                    '处理状态': '成功',
                    '原始行数': result['original_rows'],
                    '处理后行数': result['cleaned_rows'],
                    '数据是否变化': '是' if result.get('data_cleaned', False) else '否',
                    '清理操作': '; '.join(result.get('actions', [])),
                    '发现问题数': result.get('issues_count', 0),
                    '保存文件名': Path(result['file_path']).name,
                    '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            if report_data:
                report_df = pd.DataFrame(report_data)

                # 生成报告文件名
                report_filename = f"data_cleaning_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
                report_path = self.config['report_dir'] / report_filename

                # 保存到Excel
                with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                    report_df.to_excel(writer, sheet_name='清洗报告', index=False)

                # 添加统计信息sheet
                stats_data = [
                    ['统计项', '数值'],
                    ['总文件数', len(successful_result)],
                    ['需要清理的文件', len([r for r in successful_result if r.get('data_changed', False)])],
                    ['数据良好的文件', len([r for r in successful_result if not r.get('data_changed', False)])],
                    ['平均发现问题数', report_df['发现问题数'].mean() if not report_df.empty else 0],
                    ['报告生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['报告保存路径', str(report_path)],
                    ['清洗数据目录', 'data/clean/'],
                    ['报告目录', 'report/']
                ]

                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='统计信息', index=False)
            print(f"\n📄 详细报告已保存到 report/{report_filename}")

            # 显示报告文件大小
            file_size_kb = report_path.stat().st_size / 1024
            print(f"📏 报告文件大小: {file_size_kb:.1f} KB")
        except Exception as e:
            print(f"保存报告时出错: {e}")
            import traceback
            traceback.print_exc()

# 主程序
def main():
    print("=" * 70)
    print("智能数据质量分析系统")
    print("功能: 分析数据质量，智能清理，自动创建目录")
    print("=" * 70)

    try:
        # 创建清洗器（会自动检查和创建目录）
        cleaner = SmartDataCleaner()

        # 运行分析
        cleaner.run_analysis()

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

# 程序入口
if __name__ == "__main__":
    main()

    # 在Windows下保持窗口
    if sys.platform == "win32":
        input("\n按 Enter 键退出...")



'''
## 第4天：数据清洗与质量优化

**任务目标**：
- 处理缺失值（价格、成交量等关键数据）
- 处理异常值（价格逻辑错误、不合理数值）
- 保存清洗后的数据到指定目录
- 对比清洗前后数据差异，生成分析报告

**实现方案**：
1. **智能目录管理**：
   - 自动识别项目根目录和子目录结构
   - 动态创建缺失的数据目录（data/clean/, report/）
   - 扫描data/raw/目录下的所有数据文件

2. **质量分析引擎**：
   - **缺失值检测**：识别各列的缺失数据，统计缺失比例
   - **重复数据检查**：检测并统计完全重复的行
   - **价格逻辑验证**：确保价格关系合理（最低价≤开盘价/收盘价≤最高价）
   - **数据完整性评估**：检查关键字段是否存在

3. **智能清理策略**：
   - **价格数据填充**：使用前向填充和后向填充处理缺失价格
   - **其他数据填充**：数值列使用中位数，分类列使用众数
   - **重复数据处理**：自动删除完全重复的行
   - **逻辑错误修正**：自动修正不合理的价格关系

4. **报告生成系统**：
   - 保存清洗后的数据到data/clean/目录
   - 生成详细的Excel报告，包含处理信息和统计
   - 提供清晰的摘要报告，显示清理操作和效果

**核心代码结构**：
"""python
class SmartDataCleaner:
    ├── __init__()                # 初始化配置和目录结构
    ├── check_and_create_directories()  # 检查并创建必要目录
    ├── find_data_files()         # 查找数据文件
    ├── extract_symbol()          # 从文件名提取股票代码
    ├── analyze_data_quality()    # 分析数据质量问题
    ├── display_analysis_results() # 显示分析结果
    ├── smart_clean_data()        # 执行智能数据清理
    ├── save_cleaned_data()       # 保存清洗后的数据
    ├── process_file()            # 处理单个文件
    ├── run_analysis()            # 运行完整分析流程
    ├── generate_summary_report() # 生成总结报告
    └── save_detailed_report()    # 保存详细报告到Excel
"""
'''