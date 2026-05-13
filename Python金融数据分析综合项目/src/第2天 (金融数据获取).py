'''
第2天
					金融数据获取: (使用 pandas-datareader + Stooq)
-使用 Python 获取历史行情数据（日线）
-检查数据完整性（时间、缺失值）
-保存原始数据为 Excel 文件

练习：
-编写脚本，自动下载并保存历史行情数据
'''

# ===================== 基础库导入 =====================
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import time
import re
from pathlib import Path
import warnings

# 导入 pandas-datareader
import pandas_datareader.data as web


# 忽略警告信息
warnings.filterwarnings('ignore')

# ===================== 股票数据获取器类 =====================
class StockDataFetcher:
    """
        股票数据获取器 - 使用 pandas-datareader (Stooq 数据源)
        简化版：只保存Excel文件，无元数据
        """
    def __init__(self):
        """
                初始化数据获取器
                只保存Excel文件，不需要选择格式
                """
        # 设置项目目录结构
        self.setup_project_directories()

        # 数据源信息
        self.data_source = 'Stooq'
        self.data_requests = 0
        self.start_time = datetime.now()

        # Stooq 股票代码映射
        self.stooq_symbol_map = {
            # 常见美股映射
            'AAPL': 'AAPL.US',
            'MSFT': 'MSFT.US',
            'GOOGL': 'GOOGL.US',
            'AMZN': 'AMZN.US',
            'TSLA': 'TSLA.US',
            'NVDA': 'NVDA.US',
            'META': 'META.US',
            'JPM': 'JPM.US',
            'JNJ': 'JNJ.US',
            'V': 'V.US',
            'WMT': 'WMT.US',
            'PG': 'PG.US',
            'UNH': 'UNH.US',
            'HD': 'HD.US',
            'BAC': 'BAC.US',
            'MA': 'MA.US',
            'XOM': 'XOM.US',
            'CVX': 'CVX.US',
            'PFE': 'PFE.US',
            'ABBV': 'ABBV.US',

            # ETF映射
            'SPY': 'SPY.US',
            'QQQ': 'QQQ.US',
            'DIA': 'DIA.US',
            'IWM': 'IWM.US',
            'VTI': 'VTI.US',
            'VOO': 'VOO.US',
        }

        print("=" * 60)
        print("Day 2: 金融数据获取 - 使用 Stooq 数据源")
        print("=" * 60)
        print(f"数据源: {self.data_source}")
        print("文件格式: Excel (.xlsx)")
        print("文件名格式: 股票代码_data_stock.xlsx")
        print("特点: 完全免费、无需API密钥")
        print("=" * 60)

    def setup_project_directories(self):
        """创建项目目录结构"""
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent

        # 只保留一个目录：raw (存放Excel文件)
        self.raw_data_dir = self.project_root / "data" / "raw"

        print("📁 创建项目目录:")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ raw: {self.raw_data_dir}")
        print("   说明: 所有Excel文件将保存在此目录")

    def get_user_input(self):
        """获取用户输入的股票代码列表"""
        print("\n" + "=" * 60)
        print("📈 股票代码输入")
        print("=" * 60)
        print("\n支持的股票代码示例：")
        print("  美股: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA")
        print("  ETF: SPY, QQQ, DIA, VOO")
        print("  其他: 会自动添加 .US 后缀")

        print("\n请输入股票代码（多个用逗号或空格分隔）：")
        print("输入 'demo' 使用示例数据，'quit' 退出")

        while True:
            user_input = input('\n 股票代码: ').strip()
            if user_input.lower() == 'quit':
                print("👋 程序退出")
                return []
            if user_input.lower() == 'demo':
                print("🔄 使用示例股票：AAPL, MSFT, SPY")
                return ["AAPL", "MSFT", "SPY"]
            if not user_input:
                print("❌ 请输入至少一个股票代码")
                continue

            symbols = self.clean_and_validate_symbols(user_input)

            if not symbols:
                print("❌ 未找到有效的股票代码")
                continue

            print(f"\n✅ 识别到 {len(symbols)} 个股票代码:")
            for i, symbol in enumerate(symbols, 1):
                stooq_symbol = self.get_stooq_symbol(symbol)
                print(f"   {i:2d}. {symbol} → {stooq_symbol}")

            confirm = input("\n确认下载？(y/n): ").strip().lower()
            if confirm in ['y', 'yes', '']:
                return symbols

            print("请重新输入")

    def clean_and_validate_symbols(self, user_input):
        """清理和验证股票代码"""
        symbols = re.split(r'[,;\s]+', user_input.strip() )
        cleaned_symbols = []
        for symbol in symbols:
            symbol= symbol.strip().upper()
            if not symbol:
                continue

            symbol = re.sub(f'[^\w]', '', symbol)
            if 1 <= len(symbol) <= 10:
                cleaned_symbols.append(symbol)
            else:
                print(f"⚠️  跳过无效代码: {symbol}")
        return sorted(list(set(cleaned_symbols)))

    def get_stooq_symbol(self, symbol):
        """将标准股票代码转换为Stooq格式"""
        if symbol in self.stooq_symbol_map:
            return self.stooq_symbol_map[symbol]
        elif '.' not in symbol:
            # 添加 .US 后缀
            return f"{symbol}.US"
        else:
            return symbol

    def select_time_range(self):
        """选择时间范围"""
        print("\n📅 选择时间范围:")
        print("  1. 最近1个月")
        print("  2. 最近3个月")
        print("  3. 最近1年（推荐）")
        print("  4. 最近3年")
        print("  5. 最近5年")
        print("  6. 自定义开始日期")
        today = datetime.now()
        end_date = today.strftime('%Y-%m-%d')

        choice = input("请选择 (1-6, 默认3): ").strip() or '3'

        if choice == '1':
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近1个月")
        elif choice == '2':
            start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近3个月")
        elif choice == '3':
            start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近1年")
        elif choice == '4':
            start_date = (today - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近3年")
        elif choice == '5':
            start_date = (today - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近5年")
        elif choice == '6':
            while True:
                start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    if start_dt > today:
                        print("❌ 开始日期不能晚于今天")
                        continue
                    break
                except ValueError:
                    print("❌ 日期格式错误")
            print(f"📅 时间范围: {start_date} 至 {end_date}")
        else:
            start_date = ( today - timedelta(days=365)).strftime('%Y-%m-%d')
            print(f"📅 时间范围: 最近1年")
        return start_date, end_date

    def fetch_daily_data(self, symbol, start_date, end_date):
        """使用 Stooq 数据源获取股票日线数据"""
        print(f"\n📥 开始下载 {symbol} 数据...")

        self.data_requests += 1
        try:
            # 转换为Stooq格式
            stooq_symbol = self.get_stooq_symbol(symbol)
            print(f"   使用Stooq代码: {stooq_symbol}")
            print(f"   日期范围: {start_date} 至 {end_date}")

            # 从Stooq获取数据
            df = web.DataReader(
                stooq_symbol,
                'stooq',
                start=start_date,
                end=end_date
            )

            if df is None or df.empty:
                print(f"❌ {symbol}: 获取的数据为空")
                return None

            # 标准化列名
            df = df.rename(columns=str.lower)

            # 确保有必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            column_mapping = {}

            for col in required_cols:
                if col not in df.columns:
                    # 查找可能的列名
                    possible_matches = {
                        'open': ['open', 'opening', 'Open'],
                        'high': ['high', 'High'],
                        'low': ['low', 'Low'],
                        'close': ['close', 'closing', 'Close'],
                        'volume': ['volume', 'vol', 'Volume']
                    }

                    for df_col in df.columns:
                        for key, possible_list in possible_matches.items():
                            if df_col.lower in possible_list and key not in column_mapping:
                                column_mapping[df_col] = key

            if column_mapping:
                df = df.rename(columns=column_mapping)

            # 添加基本信息
            df['symbol'] = symbol
            df['data_source'] = self.data_source

            # 确保按日期排序
            df = df.sort_index()

            # 添加计算列
            df = self.add_calculated_columns(df)

            print(f"✅ {symbol}: 成功获取 {len(df)} 条记录")
            print(f"   时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")

            return df
        except Exception as e:
            # 修改这里，显示完整的错误信息
            print(f"❌ {symbol}: 获取数据失败 - {type(e).__name__}: {str(e)}")

            # 提供更详细的错误信息
            error_msg = str(e)
            if "No data fetched" in error_msg or "404" in error_msg:
                print(f"   提示: 代码 '{symbol}' 在Stooq中不存在或没有数据")
                print(f"   建议: 尝试常见美股如 AAPL, MSFT, GOOGL")
            elif "date" in error_msg.lower():
                print(f"   提示: 日期范围可能有问题")
                print(f"   建议: 尝试更短的时间范围")
            return None

    def add_calculated_columns(self, df):
        """添加计算列"""
        if 'close' not in df.columns:
            return df
        # 基础计算
        df['price_change'] = df['close'].diff()
        df['pct_change'] = df['close'].pct_change() * 100

        # 添加下载时间
        df['download_date'] = datetime.now().strftime('%Y-%m-%d')
        return df

    def save_to_excel(self, df, symbol):
        """
                保存数据到Excel文件
                文件名格式：股票代码_data_stock.xlsx
                示例：AAPL_data_stock.xlsx
                """
        if df is None or df.empty:
            print(f"❌ {symbol}: 无数据可保存")
            return None

        try:
            # 生成文件名
            filename = f"{symbol}_data_stock.xlsx"
            filepath = self.raw_data_dir / filename
            # 保存Excel文件
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 主数据sheet
                df.to_excel(writer, sheet_name="股票数据")
                # 添加一个信息sheet
                info_df = pd.DataFrame({
                    '信息项': ['股票代码', '数据行数', '时间范围', '数据源', '下载日期', '文件大小'],
                    '内容': [
                        symbol,
                        len(df),
                        f"{df.index[0].date()} 至 {df.index[-1].date()}",
                        self.data_source,
                        datetime.now().strftime('%Y-%m-%d %H:%M'),
                        '待计算'
                    ]
                })
                info_df.to_excel(writer, sheet_name='文件信息', index=False)

            # 计算文件大小
            file_size_kb = filepath.stat().st_size / 1024
            print(f"💾 {symbol}: Excel文件保存成功")
            print(f"   文件名: {filename}")
            print(f"   文件大小: {file_size_kb:.1f} KB")
            print(f"   保存路径: {filepath}")

            # 更新信息sheet中的文件大小
            from openpyxl import load_workbook
            wb = load_workbook(filepath)
            ws_info = wb['文件信息']
            ws_info['B6'] = f"{file_size_kb:.1f} KB"
            wb.save(filepath)
            return filepath

        except PermissionError:
            print(f"❌ {symbol}: 文件被占用，请关闭Excel后重试")
            return None
        except Exception as e:
            print(f"❌ {symbol}: 保存失败 - {type(e).__name__}: {e}")
            return None

    def run_download_task(self):
        """执行下载任务"""
        print("\n" + "=" * 60)
        print("开始数据下载任务")
        print("=" * 60)

        # 1. 获取股票代码
        symbols = self.get_user_input()
        if not symbols:
            return
        # 2. 选择时间范围
        start_date, end_date = self.select_time_range()

        print(f"\n开始下载 {len(symbols)} 个股票的数据...")
        print(f"时间范围: {start_date} 至 {end_date}")
        print(f"文件格式: Excel (.xlsx)")
        print(f"文件名: 股票代码_data_stock.xlsx")
        print("=" * 60)

        results = {}

        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] 处理股票: {symbol}")
            print("-" * 40)

            # 下载数据
            df = self.fetch_daily_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                # 保存为Excel文件
                filepath = self.save_to_excel(df, symbol)
                if filepath:
                    # 记录成功结果
                    results[symbol] = {
                        'status': 'success',
                        'rows': len(df),
                        'start_date': df.index[0].strftime('%Y-%m-%d'),
                        'end_date': df.index[-1].strftime('%Y-%m-%d'),
                        'file_path': str(filepath)
                    }
                    print(f"✅ 保存成功: {filepath.name}")
                else:
                    results[symbol] = {'status': 'failed', 'error': '文件保存失败'}
                    print(f"❌ 文件保存失败")
            else:
                results[symbol] =  {'status': 'failed', 'error': '数据下载失败或无数据'}
                print(f"❌ 数据下载失败或无数据")
            # 避免请求过快
            if i < len(symbols):
                time.sleep(2)
        # 显示最终结果
        total_time = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for r in results.values() if r.get('status') == 'success')

        print("\n" + "=" * 60)
        print("🎉 数据获取任务完成!")
        print("=" * 60)

        print(f"\n📊 任务统计:")
        print(f"   总股票数: {len(symbols)}")
        print(f"   成功下载: {successful}")
        print(f"   失败下载: {len(symbols) - successful}")
        print(f"   总耗时: {total_time:.1f} 秒")

        print(f"\n📁 Excel文件位置:")
        print(f"   {self.raw_data_dir}")

        if successful > 0:
            print(f"\n📄 生成的文件:")
            for symbol, info in results.items():
                if info.get('status') == 'success':
                    filename = Path(info.get('file_path', '')).name
                    print(f"   ✓ {filename}")
        print("\n" + "=" * 60)
        print("💡 使用说明:")
        print("   1. 直接双击Excel文件即可打开查看")
        print("   2. 每个文件包含两个sheet: '股票数据'和'文件信息'")
        print("   3. 如果重新下载同名股票，会覆盖原文件")
        print("   4. 建议定期备份重要数据")
        print("=" * 60)

        return results

# ===================== 主程序入口 =====================
def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("Day 2: 金融数据获取 - Excel版")
    print("=" * 70)
    print("功能: 从 Stooq 获取美股历史数据，保存为Excel文件")
    print("特点: 完全免费、无需API密钥、直接打开Excel查看")
    print("文件名: 股票代码_data_stock.xlsx")
    print("示例: AAPL_data_stock.xlsx, MSFT_data_stock.xlsx")
    print("=" * 70)

    try:
        # 创建数据获取器
        fetcher = StockDataFetcher()
        # 创建数据获取器
        results = fetcher.run_download_task()
        if results:
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            print(f"\n✅ 任务完成! 成功下载 {successful} 个股票的数据")
            print(f"📁 文件位置: {fetcher.raw_data_dir}")
        else:
            print(f"\n⚠️  任务未完成或中断")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 运行主程序
    main()

    # 在Windows下保持窗口
    if sys.platform == 'win32':
        input("\n按 Enter 键退出程序...")


'''
# 第2天：金融数据获取

**任务目标**：
- 使用 Python 获取历史行情数据（日线）
- 检查数据完整性（时间范围、缺失值）
- 保存原始数据为 Excel 文件
- 编写脚本，自动下载并保存历史行情数据

**实现方案**：
1. **数据源选择**：使用 pandas-datareader 的 Stooq 数据源
2. **主要功能**：
   - 支持多股票代码批量下载
   - 用户自定义时间范围选择
   - 自动添加 .US 后缀适配 Stooq 格式
   - 完整的数据完整性检查
   - 保存为 Excel 文件，便于直接查看

3. **技术特点**：
   - 完全免费，无需 API 密钥
   - 欧洲数据源，无 IP 限制
   - 包含基础计算列（价格变化、收益率）
   - 自动生成文件信息 sheet

**核心代码结构**：
```python
class StockDataFetcher:
    ├── __init__()          # 初始化，设置目录和股票代码映射
    ├── get_user_input()    # 获取用户输入的股票代码
    ├── fetch_daily_data()  # 从 Stooq 获取日线数据
    ├── add_calculated_columns() # 添加基础计算列
    ├── save_to_excel()     # 保存为 Excel 文件
    └── run_download_task() # 执行完整的下载任务
``` 
'''



