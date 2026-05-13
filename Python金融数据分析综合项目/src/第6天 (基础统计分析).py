'''
第6天
					基础统计分析
-计算年化收益率
-计算年化波动率
-计算最大回撤

练习：
-输出基础统计指标表格
'''

# 导入库
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class StatisticsCalculator:
    '''基础统计分析, 每一步都会打印结果确认结果'''
    def __init__(self):
        """第一步：初始化设置"""
        print("=" * 70)
        print("🔧 第1步：初始化设置")
        print("=" * 70)

        # 1. 获取当前文件所有目录
        current_dir = Path(__file__).parent
        print(f"当前文件目录: {current_dir}")

        # 2. 找到项目的目录 (假设是当前目录的上一级)
        self.project_root = current_dir.parent
        print(f" 📁 项目根目录: {self.project_root}")

        # 3. 现在要找第5天的项目
        self.input_dir = self.project_root / "data" / "returns"
        print(f"   📁 输入目录路径: {self.input_dir}")

        # 检查输入目录是否存在
        if self.input_dir.exists():
            print(f"   ✅ 输入目录存在！")

            # 现在要查看目录下有什么文件
            print(f"查看目录内容...")
            files_in_dir = list(self.input_dir.glob('*'))

            # 现在要循环查看每一个目录
            if files_in_dir:
                print(f" 目录中有{len(files_in_dir)}个文件/ 文件夹")  # len()是查看多少个文件
                for i, item in enumerate(files_in_dir[:5], 1):          # [:5] 这个就会显示5个文件. 总共是25个文件
                    if item.is_file():
                        print(f"     📄 {i}.{item.name} (文件)")
                    else:
                        print(f"     📁 {i}.{item.name} (文件夹)")

                if len(files_in_dir) > 5:        # 这代码是显示还有多少文件
                    print(f" 还有{len(files_in_dir) - 5}个")

            # 如果没有查到就会打印为空
            else:
                print(f"  ⚠️ 目录为空")

        else:
            print(f"   ❌ 输入目录不存在！")         # 这是如果文件不存在就会打印这一行.
            print(f"检查路径是否正确: {self.input_dir}")

        # 4. 设置输出目录
        self.output_dir = self.project_root / "data" / "statistics"     # 这个代码会把我们下面计算的结果保存到statistics文件.
        print(f"    📁 输出目录路径: {self.output_dir}")
        # 检查输出目录是否存在: (statistics文件是不存在的,  所以会运行else下面一行. )
        if self.output_dir.exists():
            print(f"   ✅ 输出目录已存在")          #
        else:
            print(f"   ℹ️ 输出目录不存在, 运行时会自动创建")

        # 5. 初始化结果列表
        print(f" 初始化时间存储......")
        self.all_results = []
        print(f"   📊 创建空的结果列表: {self.all_results}")
        print(f"   💾 准备存储股票统计结果")

        # 6 显示配置完成
        print("-" * 70)
        print("配置总结:")
        print(f" 项目目录: {self.project_root}")
        print(f" 输入目录: {self.input_dir}")
        print(f" 输出目录: {self.output_dir}")
        print(f" 存储列表: 已初始化")
        print("-" * 70)


    def find_files(self):
        '''第二步: 查找第5天的Excel文件'''
        # 检查输入目录是否存在
        print(f" \n第2步:  查找Excel文件目录")
        print(f" \n 1. 检查输入目录是否存在........")
        if not self.input_dir.exists():
            print(f"输入目录不存在")
            print(f" 检查路径: {self.input_dir}")
            return []

        print(f"输入目录存在: {self.input_dir}")

        # 查找所有的excel文件
        print(f" \n 2. 查找所有 .xlsx 文件")
        excel_files = list(self.input_dir.glob('*.xlsx'))
        if len(excel_files) == 0:
            print(f" 没有找到 .xlsx 文件")

            # 查找目录中有什么其它文件     这些代码是查看有没有其它的文件.  我们只有excel文件.
            print(f" 查看目录中所有文件........")
            all_files = list(self.input_dir.glob('*'))
            if all_files:
                print(f" 找到{len(all_files)}个文件/文件夹:")
                for i, item in enumerate(all_files[:10], 1):
                    if item.is_file():
                        print(f"{i}.{item.name}")
                    else:
                        print(f"{i}.{item.name} (文件夹)")
            return []
        print(f"找到{len(excel_files)}个excel文件")

        # 显示文件详细信息
        print(f" \n 3. 文件详细信息:")
        for i, file in enumerate(excel_files[:10], 1):      # 显示前10个
            file_size = file.stat().st_size / 1024      # 转为KB
            print(f" {i:2d}. {file.name}")
            print(f" 大小: {file_size:.1f}KB")
            print(f" 路径: {file}")

        if len(excel_files) > 10:
            print(f" 还有{len(excel_files) - 10}个文件")

        # 保存到属性中
        self.found_files = excel_files
        print(f" \n 已将文件保存到self.found_files: {self.found_files}")

        return excel_files

    def analyze_file(self, file_path):
        """第三步：分析单个Excel文件"""
        print("\n" + "=" * 70)
        print(f"📊 第3步：分析文件 - {file_path.name}")
        print("=" * 70)

        # 1. 从文件名提取股票代码
        print("1. 📝 提取股票代码...")
        print(f"   文件名: {file_path.name}")

        # 去掉扩展名
        filename_without_ext = file_path.stem
        print(f"  # 去掉扩展名: {filename_without_ext}")

        # 按'_'分割文件名
        parts = filename_without_ext.split('_')
        print(f" 分割结果: {parts}")


        if len(parts) > 0:
            symbol = parts[0]    # 取第一部分作为股票代码
            print(f"   ✅ 提取到股票代码: {symbol}")
        else:
            symbol = filename_without_ext
            print(f"   ⚠️ 无法分割，使用完整文件名: {symbol}")

        print(f"   最终股票代码: {symbol}")

        # 2. 读取Excel文件
        print(f"\n 2. 📖 读取Excel文件...")
        try:
            df = pd.read_excel(file_path)
            print(f"\n  ✅  读取成功")
            print(f"   数据形状: {df.shape[0]}行 X {df.shape[1]}列")

            # 显示前几列的名称
            print(f" 列名(前10行): {list(df.columns)}")
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            return None

        # 3. 创建结果字典
        print(f" \n 3.  🗂️ 创建结果存储...")
        result = {
            'symbol': symbol,
            'filename': file_path.name,
            'data_points': len(df)       # 数据行数
        }
        print(f" 初始结果: {result}")

        # 4. 提取年化收益率
        print(f" \n 4.  📈  提取年化收益率")
        if "年化收益率" in df.columns:       #这是提取年化收益率的.  就是从columns获取年化收益率.
            annual_return = df['年化收益率'].iloc[0] # 提取第一个值
            print(f"   ✅ 找到年化收益率: {annual_return}")

            # 现在把年化收益率改成英文annual_return, 之后再把annual_return保存到result字典里
            result['annual_return'] = annual_return

            # 如果是数值，显示百分比格式
            if isinstance(annual_return, (int, float)):
                print(f" 格式化显示: {annual_return:.2%}")

        else:
            print(f"    ❌ 没有找到 '年化收益率' 列")
            result['annual_return'] = None

        # 5. 提取最大回撤
        print(f" \n 5.  📉 提取最大回撤.......")
        if '最大回撤' in df.columns:
            max_drawdown = df['最大回撤'].min()     # 取最小值（因为是负值）
            print(f"   ✅  找到最大回撤: {max_drawdown}")
            result['max_drawdown'] = max_drawdown

            # 现在就是把数据改成%
            if isinstance(max_drawdown, (int, float)):
                print(f" 格式化显示: {max_drawdown:.2%}")

        else:
            print(f"    ❌ 没有找到'最大回撤'列")
            result['max_drawdown'] = None

        # 6. 计算年化波动率
        print(f" \n 6. 📊 计算年化波动率.........")
        # 要想提取日收益率
        if '日收益率' in df.columns:
            print(f"   ✅  找到 '日收益率' 列")

            # 获取日收益率数据，去掉空值
            daily_returns = df['日收益率'].dropna()
            print(f" 有些数据点: {len(daily_returns)} 个")

            if len(daily_returns) >= 2:
                # 计算日波动率(标准差)
                daily_volatility = daily_returns.std()
                print(f" 日波动率: {daily_volatility}")
                print(f" 日波动率: {daily_volatility:.2%}")

                # 计算年化波动率 = 日波动率 *  √252
                annual_volatility = daily_volatility * np.sqrt(252)
                print(f" 年化波动率: {annual_volatility}")
                print(f" 格式化显示: {annual_volatility:.2%}")
                result['annual_volatility'] = annual_volatility
            else:
                print(f"    ⚠️ 数据不足，需要至少2个有效数据点")
                result['annual_volatility'] = None
        else:
            print(f"   ❌ 没有找到'日收益率'列")
            result['annual_volatility'] = None

        # 7. 保存结果
        print(f" \n 7. 💾 保存分析结果...")
        self.all_results.append(result)    # 现在是把result的数据保存到all_results字典里
        print(f"   ✅ 已保存到 self.all_results")
        print(f" 当前结果数据: {len(self.all_results)}")

        # 8. 显示本次分析总结
        print('\n' + '-' * 70)
        print(f" 📋 分析总结: {symbol}")
        print('_' * 70)

        if result.get('annual_return') is not None:
            print(f"   年化收益率: {result['annual_return']:.2%}")
        else:
            print(f"   年化收益率: N/A")

        if result.get('annual_volatility') is not None:
            print(f"   年化波动率: {result['annual_volatility']:.2%}")
        else:
            print(f"   年化波动率: N/A")

        if result.get('max_drawdown') is not None:
            print(f"   最大回撤: {result['max_drawdown']:.2%}")
        else:
            print(f"   最大回撤: N/A")

        print(f"   数据点数: {result['data_points']}")
        print("-" * 70)

        print(f" \n✅ 文件分析完成: {file_path.name}")

        return result

    def process_all_files(self):
        """第四步：批量处理所有文件"""
        print('\n' + '=' * 70)
        print(f" 第4步: 批量处理所有的文件........")
        print('=' * 70)

        # 1. 确保已经找到了文件
        if not hasattr(self, 'found_files') or len(self.found_files) == 0:
            print(f" ❌ 没有找到文件，请先运行 find_files() 方法")
            return []

        print(f" 📋 准备处理 {len(self.found_files)} 个文件")

        # 2. 初始化计数器
        success_count = 0
        failed_count = 0

        print('\n' + '-' * 70)
        print(f" 📊 开始批量处理.........")
        print("-" * 70)

        # 3. 循环处理每一个文件
        for i, file in enumerate(self.found_files, 1):
            print(f"\n 📦 处理第 {i}/{len(self.found_files)} 个文件")
            print(f" 文件: {file.name}")

            try:
                # 调用def analyze_file 方法分析文件
                result = self.analyze_file(file)
                if result:
                    success_count += 1    # 每一次循环都加1
                    print(f"    ✅  处理成功: ({success_count}/{i})")
                else:
                    failed_count += 1
                    print(f"    ❌  处理失败: ({failed_count}/{i})")
            except Exception as e:
                failed_count += 1
                print(f"   ❌  处理异常: {e}")

        # 4. 显示处理结果
        print("\n" + "=" * 70)
        print(f"📈 批量处理完成!")
        print("=" * 70)

        print(f"\n 📊  处理结果统计: ")
        print(f" 总文件数: {len(self.found_files)}")
        print(f" 成功处理: {success_count}")
        print(f" 处理失败: {failed_count}")
        print(f" 成功率: {success_count/len(self.found_files)*100:.1f}%")

        print(f"\n 💾  数据存储: ")
        print(f' 保存到 all_results 的数量: {len(self.all_results)}')

        if self.all_results:
            print(f" \n 📋  所有分析结果的股票代码: ")
            symbols = [result['symbol'] for result in self.all_results]
            # 每行显示5个股票代码
            for i in range(0, len(symbols), 5):
                line_symbols = symbols[i:i+5]
                print(f" {', '.join(line_symbols)}")

        print("\n" + "=" * 70)

        return self.all_results

    def generate_statistics_table(self):
        '''第5步: 生成基础统计指标表格'''
        print("\n" + "=" * 70)
        print(f"📊 第5步：生成统计表格")
        print("=" * 70)

        # 1. 检查是否有分析结果
        if not self.all_results:
            print(f" ❌ 没有分析结果, 先运行process_all_files()")
            return None
        print(f"✅  有{len(self.all_results)}个分析结果")

        # 2. 将结果转换为DataFrame
        print(f" \n1. 将结果转换为DataFrame")
        df = pd.DataFrame(self.all_results)
        print(f" DataFrame 形状: {df.shape[0]}行 x {df.shape[1]}列")

        # 3. 创建显示用的表格
        print(f' \n2. 创建格式化表格.....')
        display_df = pd.DataFrame()

        # 添加股票代码
        display_df['股票代码'] = df['symbol']

        # 格式化年化收益率
        if 'annual_return' in df.columns:
            display_df['年化收益率'] = df['annual_return'].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            print(f"   ✅  已添加: 年化收益率")

        # 格式化年化波动率
        if 'annual_volatility' in df.columns:
            display_df['年化波动率'] = df['annual_volatility'].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            print(f"   ✅  已添加: 年化波动率")

        # 格式化最大回撤
        if 'max_drawdown' in df.columns:
            display_df['最大回撤'] = df['max_drawdown'].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            print(f"   ✅  已添加: 最大回撤")

        # 添加数据点数
        if 'data_points' in df.columns:
            display_df['数据天数'] = df['data_points']
            print(f"   ✅  已添加: 数据天数")

        print(f" \n ✅ 表格创建完成")
        print(f" 表格形状: {display_df.shape[0]}行 x {display_df.shape[1]}列")

        # 4. 显示表格
        print('\n' + '=' * 70)
        print(f"📋 基础统计指标表格预览.......")
        print("=" * 70)

        # 设置显示选项，使表格更美观
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(display_df.to_string(index=False))

        print('\n' + "=" * 70)
        return display_df



        # 再主程序调用方式

    def save_to_excel(self, statistics_table):
        """第6步: 保存统计报告到Excel文件"""
        print('\n' + "=" * 70)
        print(f" 💾 第6步：保存到Excel文件")
        print("=" * 70)

        if statistics_table is None:
            print(f' ❌ 没有表格数据可保存')
            return None

        # 1. 确保输出目录存在
        if not self.output_dir.exists():
            print(f"📁  创建输出目录.......")
            self.output_dir.mkdir(parents=True, exist_ok=True)      # 保存到新的文件里

        # 2. 生成文件名
        '''我把导入库全部放到一起'''
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"基础统计指标表格_{timestamp}.xlsx"
        filepath = self.output_dir / filename

        print(f" 📄 文件名: {filename}")
        print(f" 📂 保存路径: {filepath}")

        # 3. 保存到Excel
        print(f" \n 正在保存Excel文件中......")
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # sheet 1. 格式化表格
                statistics_table.to_excel(writer, sheet_name='统计指标', index=False)

                # sheet 2. 原始数据
                if self.all_results:
                    raw_df = pd.DataFrame(self.all_results)
                    raw_df.to_excel(writer, sheet_name='原始数据', index=False)
                    print(f"   ✅ Sheet 2: 原始数据")

                # sheet 3. 统计摘要
                summary_data = {
                    '统计项目': ['总股票数', '统计时间', '输出文件'],
                    '数值': [
                        len(self.all_results),
                        datetime.now().strftime("%Y%m%d"),
                        filename
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                print(f"    ✅ Sheet 3: 统计摘要")

            # 4. 显示保存成功信息
            file_size = filepath.stat().st_size / 1024
            print(f" \n ✅ Excel文件保存成功！")
            print(f' 文件大小: {file_size:.1f} KB')
            print(f" 包含 { len(self.all_results)} 个股票的统计信息")
            print(f' 保存位置: {filepath}')

            return filepath
        except Exception as e:
            print(f" ❌ 保存失败: {e}")
            return None


if __name__ == "__main__":
    # 调用 创建对象 (只做初始化)
    calculator = StatisticsCalculator()

    # 当需要查找文件时调用
    files = calculator.find_files()


    '''# 已经调用了analyze_file, 再这里就不用了'''
    if files:
        # 3. 批量处理所有文件
        all_results = calculator.process_all_files()

        if all_results:
            # 4. 生成统计报告:  调用generate_statistics_table()
            '''再循环files里, 因为每次批量处理所有文件就会批量生成统计报告'''
            statistics_table = calculator.generate_statistics_table()

            if statistics_table is not None:
                """ 现在需要把每一个处理好的报告保存到Excel"""
                # 5. 保存到Excel
                saved_file = calculator.save_to_excel(statistics_table)

                print("\n" + "=" * 70)
                print(f" 第6天任务完成!")
                print("=" * 70)
                print("完成的任务")
                print("  1. ✅ 初始化设置")
                print("  2. ✅ 查找数据文件")
                print("  3. ✅ 分析每个文件")
                print("  4. ✅ 生成统计指标表格")
                print("  5. ✅ 保存到Excel文件")
                print("=" * 70)

'''
## 第6天：基础统计分析

**任务目标**：
- 计算年化收益率（从第5天结果中提取）
- 计算年化波动率（基于日收益率计算）
- 计算最大回撤（从第5天结果中提取）
- 输出基础统计指标表格
- 保存统计结果到Excel文件

**实现方案**：
1. **数据源管理**：
   - 自动定位第5天生成的收益率数据目录（data/returns/）
   - 查找所有包含收益率数据的Excel文件
   - 自动创建统计分析输出目录（data/statistics/）

2. **逐步执行流程**：
   - **第1步：初始化设置**：配置项目路径、输入输出目录
   - **第2步：查找文件**：扫描并验证第5天的Excel文件
   - **第3步：分析单个文件**：提取股票代码、计算关键指标
   - **第4步：批量处理**：循环处理所有找到的数据文件
   - **第5步：生成表格**：创建格式化的统计指标表格
   - **第6步：保存结果**：将统计结果保存到Excel文件

3. **核心指标计算**：
   - **年化收益率提取**：从第5天结果中直接读取
   - **年化波动率计算**：`日波动率 × √252`（基于日收益率标准差）
   - **最大回撤提取**：从第5天结果中获取最小值
   - **数据完整性检查**：验证每个指标的有效性

4. **结果展示系统**：
   - **详细步骤输出**：每个操作都打印确认信息
   - **格式化表格**：百分比格式显示收益率、波动率、回撤
   - **多维度统计**：包含股票代码、数据天数等基本信息
   - **Excel报告**：包含统计指标、原始数据、统计摘要三个sheet

**核心代码结构**：
```python
class StatisticsCalculator:
    ├── __init__()                    # 第1步：初始化配置和目录
    ├── find_files()                  # 第2步：查找第5天的Excel文件
    ├── analyze_file()                # 第3步：分析单个文件，提取指标
    ├── process_all_files()           # 第4步：批量处理所有文件
    ├── generate_statistics_table()   # 第5步：生成统计指标表格
    └── save_to_excel()               # 第6步：保存结果到Excel文件
```
'''



