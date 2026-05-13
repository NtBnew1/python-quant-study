'''
第5天
					收益率计算
-计算日收益率
-计算累计收益
-保存计算结果

练习：
-绘制累计收益曲线
'''


# 导入必要的Python库
import pandas as pd  # 数据处理库，用于读取、处理和分析数据
import numpy as np  # 数值计算库，用于数学运算
from pathlib import Path  # 路径处理库，用于文件和目录操作
from datetime import datetime  # 日期时间库，用于处理时间相关操作
import matplotlib.pyplot as plt  # 绘图库，用于创建图表
import matplotlib  # 图表样式库，用于设置图表样式
from typing import Dict, List, Optional, Tuple  # 类型提示库，用于标注函数参数和返回值的类型
import warnings  # 警告处理库，用于控制警告信息的显示
import sys  # 系统库，用于系统相关操作


# 设置中文字体支持
def setup_chinese_font():
    """
       设置matplotlib中文字体支持
       """
    # 获取操作系统类型
    platform= sys.platform.lower()

    # 根据操作系统设置字体路径
    if platform.startswith('win'):   # Windows系统
        # Windows系统常见中文字体
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        matplotlib.rcParams['font.sans-serif'] = font_names

        # 关键修复：在Windows系统中，只设置sans-serif，让matplotlib自动选择
        matplotlib.rcParams['font.family'] = 'sans-serif'
    else:
        # 其他系统使用默认字体
        font_names = ['DejaVu Sans']
        matplotlib.rcParams['font.sans-serif'] = font_names
        matplotlib.rcParams['font.family'] = 'sans-serif'

    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置字体大小
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10

    # 检查字体是否可用
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    print(f"📊 可用的中文字体: {[f for f in font_names if f in available_fonts]}")

# 调用字体设置函数
setup_chinese_font()

warnings.filterwarnings('ignore')  # 忽略所有警告信息，使输出更整洁


class ReturnCalculator:
    """
    收益率计算器类 - 主要功能：计算日收益率和累计收益，并生成可视化图表
    这个类封装了第5天需要的所有功能
    """

    def __init__(self):
        """
        类的初始化函数，在创建ReturnCalculator对象时自动执行
        主要功能：设置项目路径、创建目录结构、查找数据文件
        """
        # 打印程序标题，使用等号分隔线使输出更清晰
        print("=" * 70)  # 打印70个等号作为分隔线
        print("第5天 - 收益率计算与可视化系统")  # 打印程序标题
        print("=" * 70)  # 打印70个等号作为分隔线

        # 获取当前脚本文件所在的目录路径
        # __file__ 是当前Python文件的路径
        current_dir = Path(__file__).parent  # 获取当前文件所在目录
        self.project_root = current_dir.parent  # 获取上一级目录作为项目根目录

        # 打印项目根目录路径，方便用户了解程序的工作目录
        print(f"📂 项目根目录: {self.project_root}")  # 使用表情符号和格式化字符串

        # 配置程序使用的目录路径
        # 使用字典存储所有重要路径，便于管理和修改
        self.config = {
            'project_root': self.project_root,  # 项目根目录
            'clean_dir': self.project_root / "data" / "clean",  # 清洗后的数据目录（第4天输出）
            'returns_dir': self.project_root / "data" / "returns",  # 收益率计算结果目录（第5天输出）
            'charts_dir': self.project_root / "charts",  # 图表保存目录
            'reports_dir': self.project_root / "report",  # 报告保存目录
        }

        # 调用创建目录的函数，确保所有需要的目录都存在
        self.create_directories()  # 创建或检查所有必要的目录

        # 查找第4天清洗后的数据文件
        self.files = self.find_cleaned_files()  # 调用查找文件函数
        self.results = []  # 初始化结果列表，用于存储每个文件处理的结果

    def create_directories(self):
        """
        创建或检查必要的目录结构
        功能：确保程序运行所需的所有目录都存在，如果不存在则创建
        """
        print("\n📁 创建/检查目录结构:")  # 打印目录结构检查开始提示

        # 定义需要检查的目录列表，包含显示名称和配置键
        directories = [
            ("收益率数据", "returns_dir"),  # 收益率数据目录
            ("图表目录", "charts_dir"),  # 图表目录
            ("报告目录", "reports_dir"),  # 报告目录
        ]

        # 遍历所有需要检查的目录
        for name, key in directories:  # name: 目录显示名称, key: 配置字典中的键
            # 从配置中获取目录路径
            dir_path = self.config[key]  # 根据键获取目录路径

            # 检查目录是否已存在
            if dir_path.exists():  # 如果目录已存在
                # 统计目录中的文件数量
                file_count = len(list(dir_path.glob('*')))  # 使用glob获取所有文件
                # 获取相对于项目根目录的相对路径（显示更友好）
                rel_path = dir_path.relative_to(self.project_root)  # 相对路径
                # 打印目录信息（已存在）
                print(f"  ✓ {name}: {rel_path}/ (已有{file_count}个文件)")  # ✓表示成功
            else:
                # 如果目录不存在，尝试创建
                try:
                    # 创建目录，parents=True表示创建父目录，exist_ok=True表示如果已存在不报错
                    dir_path.mkdir(parents=True, exist_ok=True)  # 创建目录
                    rel_path = dir_path.relative_to(self.project_root)  # 获取相对路径
                    # 打印创建成功信息
                    print(f"  📂 创建: {rel_path}/")  # 📂表示目录创建
                except Exception as e:  # 捕获创建目录时可能发生的异常
                    # 打印错误信息
                    print(f"  ❌ 创建失败 {name}: {e}")  # ❌表示失败

    def find_cleaned_files(self) -> List[Path]:
        """
        查找第4天清洗后的数据文件
        返回：包含文件路径的列表
        类型提示：-> List[Path] 表示函数返回Path对象的列表
        """
        # 从配置中获取清洗数据目录路径
        clean_dir = self.config['clean_dir']  # 清洗数据目录

        # 检查清洗数据目录是否存在
        if not clean_dir.exists():  # 如果目录不存在
            print(f"\n❌ 错误: 清洗数据目录不存在")  # 打印错误信息
            print(f"   请先运行第4天的数据清洗程序")  # 给出建议
            return []  # 返回空列表

        # 在清洗数据目录中查找文件
        # 查找所有包含"cleaned"或"verified"的Excel文件
        # glob模式匹配：*cleaned*.xlsx 匹配所有包含cleaned的Excel文件
        files = list(clean_dir.glob("*cleaned*.xlsx")) + list(clean_dir.glob("*verified*.xlsx"))

        # 检查是否找到文件
        if not files:  # 如果文件列表为空
            # 获取相对路径用于显示
            rel_path = clean_dir.relative_to(self.project_root)  # 相对路径
            print(f"\n❌ 在 {rel_path} 中没有找到清洗后的数据文件")  # 打印错误信息
            print(f"   请先运行第4天的数据清洗程序")  # 给出建议
            return []  # 返回空列表

        # 打印找到的文件信息
        print(f"\n✅ 找到 {len(files)} 个清洗后的数据文件:")  # ✅表示成功找到

        # 显示前10个文件的信息
        for i, file in enumerate(files[:10], 1):  # enumerate生成索引和文件，从1开始计数
            try:
                # 获取文件大小（转换为KB）
                size_kb = file.stat().st_size / 1024  # stat()获取文件信息，st_size是字节大小
                # 打印文件信息：索引、文件名、大小
                print(f"  {i:2d}. {file.name:<40} ({size_kb:.1f} KB)")  # 格式化输出
            except:  # 如果获取文件大小失败
                print(f"  {i:2d}. {file.name}")  # 只打印文件名

        # 如果文件超过10个，显示省略信息
        if len(files) > 10:  # 检查文件数量
            print(f"  ... 还有 {len(files) - 10} 个文件")  # 显示剩余文件数量

        return files  # 返回找到的文件列表

    def extract_symbol(self, filename: str) -> str:
        """
        从文件名中提取股票代码
        参数：filename - 文件名
        返回：股票代码字符串
        """
        # 使用Path对象的stem属性获取文件名（不包含扩展名）
        name = Path(filename).stem  # stem获取文件名主干（不带扩展名）

        # 定义需要去除的常见后缀列表
        suffixes = ['_cleaned', '_verified', '_cleaned_', '_verified_']  # 后缀列表

        # 遍历后缀列表，从文件名中去除这些后缀
        for suffix in suffixes:  # 遍历每个后缀
            name = name.replace(suffix, '')  # 替换后缀为空字符串

        # 分割文件名，提取股票代码（假设格式为"代码_其他信息"）
        parts = name.split('_')  # 使用下划线分割文件名
        if parts:  # 如果分割后至少有一个部分
            return parts[0]  # 返回第一个部分（股票代码）
        return name  # 如果分割失败，返回原始文件名

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算股票的各种收益率指标
        参数：df - 包含股票数据的DataFrame
        返回：包含收益率计算结果的DataFrame
        """
        # 创建数据副本，避免修改原始数据
        df_returns = df.copy()  # copy()创建深拷贝

        # 查找日期列
        date_col = None  # 初始化日期列为None
        # 尝试可能的日期列名
        possible_date_cols = ['date', 'Date', 'DATE', '交易日期', '日期']  # 可能的列名
        for col in possible_date_cols:  # 遍历可能的列名
            if col in df.columns:  # 如果列存在于DataFrame中
                date_col = col  # 设置日期列
                break  # 找到后跳出循环

        # 如果找到日期列且是日期类型，按日期排序
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df_returns = df_returns.sort_values(date_col)  # 按日期升序排序

        # 查找收盘价列
        close_col = None  # 初始化收盘价列为None
        # 尝试可能的收盘价列名
        possible_close_cols = ['close', 'Close', 'CLOSE', '收盘', '收盘价']  # 可能的列名
        for col in possible_close_cols:  # 遍历可能的列名
            if col in df.columns:  # 如果列存在
                close_col = col  # 设置收盘价列
                break  # 找到后跳出循环

        # 如果没有找到标准收盘价列
        if not close_col:
            # 查找所有数值类型的列
            numeric_cols = df.select_dtypes(include=[np.number]).columns  # 选择数值列
            if len(numeric_cols) > 0:  # 如果有数值列
                close_col = numeric_cols[0]  # 使用第一个数值列
                print(f"  ⚠️  未找到标准收盘价列，使用 {close_col} 列进行计算")  # 警告信息
            else:
                # 如果没有数值列，抛出错误
                raise ValueError("未找到数值列用于收益率计算")  # 抛出值错误

        # 1. 计算日收益率（简单收益率）
        # 公式：日收益率 = (今日收盘价 - 昨日收盘价) / 昨日收盘价
        df_returns['日收益率'] = df_returns[close_col].pct_change()  # pct_change计算百分比变化

        # 2. 计算对数收益率（用于统计分析和模型）
        # 公式：对数收益率 = ln(今日收盘价 / 昨日收盘价)
        df_returns['对数收益率'] = np.log(df_returns[close_col] / df_returns[close_col].shift(1))

        # 3. 计算累计收益率
        # 公式：累计收益率 = ∏(1 + 日收益率) - 1，从1开始累计
        df_returns['累计收益率'] = (1 + df_returns['日收益率']).cumprod()  # cumprod计算累积乘积
        df_returns['累计收益率'] = df_returns['累计收益率'].fillna(1)  # 用1填充NaN值（第一行）

        # 4. 计算净值（假设初始投资为1）
        df_returns['净值'] = df_returns['累计收益率']  # 净值等于累计收益率

        # 5. 计算滚动收益率（20日滚动）
        # 公式：20日滚动收益率 = (今日收盘价 - 20日前收盘价) / 20日前收盘价
        df_returns['20日滚动收益率'] = df_returns[close_col].pct_change(20)  # 计算20日变化

        # 6. 计算年化收益率（如果数据足够）
        total_days = len(df_returns)  # 总交易日数
        if total_days > 250:  # 假设一年有250个交易日
            total_return = df_returns['净值'].iloc[-1] - 1  # 总收益率
            # 年化公式：年化收益率 = (1 + 总收益率)^(250/总天数) - 1
            df_returns['年化收益率'] = (1 + total_return) ** (250 / total_days) - 1

        # 7. 计算最大回撤
        # 计算累计最大值
        df_returns['累计最大值'] = df_returns['净值'].cummax()  # cummax计算累积最大值
        # 计算回撤：当前净值相对于历史最大值的下跌幅度
        df_returns['回撤'] = (df_returns['净值'] - df_returns['累计最大值']) / df_returns['累计最大值']
        # 计算最大回撤：回撤的最小值（负值最大）
        df_returns['最大回撤'] = df_returns['回撤'].cummin()  # cummin计算累积最小值

        return df_returns  # 返回包含所有收益率指标的DataFrame

    def plot_cumulative_returns(self, df: pd.DataFrame, symbol: str) -> Tuple[plt.Figure, List[str]]:
        """
        绘制累计收益曲线和其他分析图表
        参数：
            df - 包含收益率数据的DataFrame
            symbol - 股票代码
        返回：
            图表对象和保存的图表路径列表
        """
        chart_paths = []  # 初始化图表路径列表

        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')  # 使用seaborn的深色网格样式

        # ==================== 创建第一个图表（2x2网格）====================
        # 创建2行2列的子图，设置图表大小
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))  # 16x10英寸的图表

        # 关键修复：创建一个FontProperties对象用于所有中文文本
        # 使用常见的Windows中文字体
        font_props= matplotlib.font_manager.FontProperties(fname=None, family='Microsoft YaHei')

        fig1.suptitle(f'{symbol} - 收益率分析', fontsize=16, fontweight='bold',
                      fontproperties=font_props)  # 设置总标题

        # 1.1 绘制价格走势图（左上角）
        if 'close' in df.columns:  # 检查是否有收盘价列
            ax1 = axes1[0, 0]  # 获取第一个坐标轴（第0行，第0列）
            # 绘制价格曲线，设置线宽、颜色和透明度
            ax1.plot(df.index, df['close'], linewidth=2, color='blue', alpha=0.7)
            ax1.set_title('价格走势', fontsize=12, fontproperties=font_props)  # 设置子图标题
            ax1.set_ylabel('价格', fontsize=10, fontproperties=font_props)  # 设置Y轴标签
            ax1.grid(True, alpha=0.3)  # 显示网格，设置透明度
            ax1.tick_params(axis='x', rotation=45)  # 旋转X轴刻度标签45度

        # 1.2 绘制日收益率分布直方图（右上角）
        if '日收益率' in df.columns:  # 检查是否有日收益率列
            ax2 = axes1[0, 1]  # 获取第二个坐标轴（第0行，第1列）
            # 绘制直方图，设置分箱数、透明度、颜色和边框颜色
            ax2.hist(df['日收益率'].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('日收益率分布', fontsize=12, fontproperties=font_props)  # 设置子图标题
            ax2.set_xlabel('日收益率', fontsize=10, fontproperties=font_props)  # 设置X轴标签
            ax2.set_ylabel('频数', fontsize=10, fontproperties=font_props)  # 设置Y轴标签
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)  # 在x=0处添加红色虚线
            ax2.grid(True, alpha=0.3)  # 显示网格

            # 添加统计信息文本框
            mean_return = df['日收益率'].mean()  # 计算日收益率均值
            std_return = df['日收益率'].std()  # 计算日收益率标准差
            # 在图表中添加文本
            ax2.text(0.05, 0.95, f'均值: {mean_return:.4f}\n标准差: {std_return:.4f}',
                     transform=ax2.transAxes, fontsize=9, fontproperties=font_props,
                     verticalalignment='top',  # 垂直对齐方式：顶部
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))  # 文本框样式

        # 1.3 绘制累计收益率曲线（左下角）
        if '累计收益率' in df.columns:  # 检查是否有累计收益率列
            ax3 = axes1[1, 0]  # 获取第三个坐标轴（第1行，第0列）
            # 绘制累计收益率曲线
            ax3.plot(df.index, df['累计收益率'], linewidth=2, color='red')
            ax3.set_title('累计收益率曲线', fontsize=12, fontproperties=font_props)  # 设置子图标题
            ax3.set_ylabel('累计收益率', fontsize=10, fontproperties=font_props)  # 设置Y轴标签
            ax3.set_xlabel('日期', fontsize=10, fontproperties=font_props)
            ax3.grid(True, alpha=0.3)  # 显示网格
            ax3.tick_params(axis='x', rotation=45)  # 旋转X轴刻度

            # 标记起始点和结束点
            if len(df) > 0:  # 确保有数据
                # 在起点添加绿色散点
                ax3.scatter(df.index[0], df['累计收益率'].iloc[0],
                            color='green', s=100, label='开始', zorder=5)
                # 在终点添加红色散点
                ax3.scatter(df.index[-1], df['累计收益率'].iloc[-1],
                            color='red', s=100, label='结束', zorder=5)
                ax3.legend(prop=font_props)  # 显示图例

        # 1.4 绘制最大回撤图（右下角）
        if '最大回撤' in df.columns:  # 检查是否有最大回撤列
            ax4 = axes1[1, 1]  # 获取第四个坐标轴（第1行，第1列）
            # 填充最大回撤区域
            ax4.fill_between(df.index, df['最大回撤'], 0,
                             alpha=0.3, color='red')  # 填充区域
            # 绘制最大回撤曲线
            ax4.plot(df.index, df['最大回撤'], linewidth=1, color='darkred')
            ax4.set_title('最大回撤', fontsize=12, fontproperties=font_props)  # 设置子图标题
            ax4.set_ylabel('回撤幅度', fontsize=10, fontproperties=font_props)  # 设置Y轴标签
            ax4.set_xlabel('日期', fontsize=10, fontproperties=font_props)
            ax4.grid(True, alpha=0.3)  # 显示网格
            ax4.tick_params(axis='x', rotation=45)  # 旋转X轴刻度

        # 调整子图布局，避免重叠
        plt.tight_layout()

        # 保存第一个图表到文件
        chart1_path = self.config['charts_dir'] / f'{symbol}_收益率分析.png'  # 文件路径
        # 保存图表，设置DPI为150，bbox_inches='tight'确保图表内容完整保存
        fig1.savefig(chart1_path, dpi=150, bbox_inches='tight')
        chart_paths.append(str(chart1_path))  # 添加文件路径到列表

        # ==================== 创建第二个图表（详细分析）====================
        # 创建第二个图表（2x2网格）
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        fig2.suptitle(f'{symbol} - 详细收益分析', fontsize=16, fontweight='bold',
                      fontproperties=font_props)  # 总标题

        # 2.1 绘制日收益率时间序列（左上角）
        if '日收益率' in df.columns:
            ax1 = axes2[0, 0]
            ax1.plot(df.index, df['日收益率'], linewidth=1, color='blue', alpha=0.7)
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)  # 添加零线
            ax1.set_title('日收益率时间序列', fontsize=12, fontproperties=font_props)
            ax1.set_ylabel('日收益率', fontsize=10, fontproperties=font_props)
            ax1.set_xlabel('日期', fontsize=10, fontproperties=font_props)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

        # 2.2 绘制20日滚动收益率（右上角）
        if '20日滚动收益率' in df.columns:
            ax2 = axes2[0, 1]
            ax2.plot(df.index, df['20日滚动收益率'], linewidth=2, color='purple')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)  # 添加零线
            ax2.set_title('20日滚动收益率', fontsize=12, fontproperties=font_props)
            ax2.set_ylabel('滚动收益率', fontsize=10, fontproperties=font_props)
            ax2.set_xlabel('日期', fontsize=10, fontproperties=font_props)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

        # 2.3 绘制净值曲线（左下角）
        if '净值' in df.columns:
            ax3 = axes2[1, 0]
            ax3.plot(df.index, df['净值'], linewidth=2, color='darkorange')
            ax3.set_title('净值曲线', fontsize=12, fontproperties=font_props)
            ax3.set_ylabel('净值', fontsize=10, fontproperties=font_props)
            ax3.set_xlabel('日期', fontsize=10, fontproperties=font_props)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)

            # 添加总收益统计文本框
            if len(df) > 0:
                # 计算总收益率（从净值计算）
                total_return = (df['净值'].iloc[-1] - 1) * 100  # 转换为百分比
                ax3.text(0.05, 0.95, f'总收益: {total_return:.2f}%',
                         transform=ax3.transAxes, fontsize=10, fontproperties=font_props,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 2.4 绘制月收益率箱型图或日收益率散点图（右下角）
        if '日收益率' in df.columns and 'date' in df.columns:
            try:
                ax4 = axes2[1, 1]
                # 从日期列提取月份信息
                df['month'] = pd.to_datetime(df['date']).dt.to_period('M')  # 转换为月度周期
                # 按月份分组，获取每月的日收益率列表
                monthly_returns = df.groupby('month')['日收益率'].apply(list)

                # 创建箱型图
                months = [str(m) for m in monthly_returns.index]  # 转换为字符串列表
                ax4.boxplot(monthly_returns.values, labels=months)  # 绘制箱型图
                ax4.set_title('月收益率分布箱型图', fontsize=12, fontproperties=font_props)
                ax4.set_ylabel('收益率', fontsize=10, fontproperties=font_props)
                ax4.set_xlabel('月份', fontsize=10, fontproperties=font_props)
                ax4.grid(True, alpha=0.3)
                ax4.tick_params(axis='x', rotation=45)
            except:
                # 如果无法分组，绘制简单的日收益率散点图
                ax4.scatter(range(len(df)), df['日收益率'], alpha=0.5, s=10)
                ax4.set_title('日收益率散点图', fontsize=12, fontproperties=font_props)
                ax4.set_ylabel('日收益率', fontsize=10, fontproperties=font_props)
                ax4.set_xlabel('样本序号', fontsize=10, fontproperties=font_props)
                ax4.grid(True, alpha=0.3)

        # 调整第二个图表的布局
        plt.tight_layout()

        # 保存第二个图表
        chart2_path = self.config['charts_dir'] / f'{symbol}_详细收益分析.png'
        fig2.savefig(chart2_path, dpi=150, bbox_inches='tight')
        chart_paths.append(str(chart2_path))

        # 关闭图表以释放内存
        plt.close(fig1)  # 关闭第一个图表
        plt.close(fig2)  # 关闭第二个图表

        return fig1, chart_paths  # 返回图表对象和路径列表

    def save_return_data(self, df_returns: pd.DataFrame, symbol: str) -> Path:
        """
        保存收益率计算结果到Excel文件
        参数：
            df_returns - 包含收益率数据的DataFrame
            symbol - 股票代码
        返回：保存的文件路径
        """
        # 生成时间戳，用于文件名，避免重复
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒
        filename = f"{symbol}_收益率_{timestamp}.xlsx"  # 生成文件名
        filepath = self.config['returns_dir'] / filename  # 完整文件路径

        # 使用ExcelWriter保存数据到Excel文件
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. 保存收益率数据到第一个sheet
            df_returns.to_excel(writer, sheet_name='收益率数据', index=False)

            # 2. 计算并保存统计摘要
            stats_data = self.calculate_statistics(df_returns, symbol)  # 调用统计函数
            # 将字典转换为DataFrame
            stats_df = pd.DataFrame(list(stats_data.items()), columns=['指标', '值'])
            stats_df.to_excel(writer, sheet_name='统计摘要', index=False)  # 保存到第二个sheet

            # 3. 保存收益率分布描述
            if '日收益率' in df_returns.columns:  # 如果有日收益率数据
                # 使用describe()获取基本统计信息
                returns_desc = df_returns['日收益率'].describe()
                # 创建DataFrame
                returns_df = pd.DataFrame({
                    '统计量': returns_desc.index,  # 统计量名称
                    '数值': returns_desc.values  # 统计值
                })
                returns_df.to_excel(writer, sheet_name='收益率分布', index=False)  # 第三个sheet

        return filepath  # 返回保存的文件路径

    def calculate_statistics(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        计算收益率的统计指标
        参数：
            df - 包含收益率数据的DataFrame
            symbol - 股票代码
        返回：统计指标字典
        """
        # 初始化统计字典，包含基本信息
        stats = {
            '股票代码': symbol,  # 股票代码
            '数据期间': f"{len(df)} 个交易日",  # 数据长度
            '计算时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间
        }

        # 计算日收益率相关统计
        if '日收益率' in df.columns:  # 检查是否有日收益率数据
            returns = df['日收益率'].dropna()  # 去除缺失值
            if len(returns) > 0:  # 如果有有效数据
                stats['平均日收益率'] = f"{returns.mean():.6f}"  # 均值
                stats['日收益率标准差'] = f"{returns.std():.6f}"  # 标准差
                stats['日收益率偏度'] = f"{returns.skew():.6f}"  # 偏度（分布对称性）
                stats['日收益率峰度'] = f"{returns.kurtosis():.6f}"  # 峰度（分布尖峭度）
                stats['最大单日涨幅'] = f"{returns.max():.6f}"  # 最大值
                stats['最大单日跌幅'] = f"{returns.min():.6f}"  # 最小值

                # 计算正收益比例
                positive_ratio = (returns > 0).sum() / len(returns)  # 正收益天数比例
                stats['正收益比例'] = f"{positive_ratio:.2%}"  # 格式化为百分比

        # 计算累计收益率统计
        if '累计收益率' in df.columns and len(df) > 0:  # 检查累计收益率数据
            # 计算总收益率：最后一天的累计收益率减1
            total_return = df['累计收益率'].iloc[-1] - 1
            stats['累计总收益'] = f"{total_return:.2%}"  # 格式化为百分比

            # 计算年化收益率（假设一年250个交易日）
            if len(df) >= 250:  # 数据足够长
                # 年化公式：(1 + 总收益率)^(250/总天数) - 1
                annual_return = (1 + total_return) ** (250 / len(df)) - 1
                stats['年化收益率'] = f"{annual_return:.2%}"  # 年化收益率

        # 计算最大回撤
        if '最大回撤' in df.columns:
            max_drawdown = df['最大回撤'].min()  # 最大回撤是最小值（负值）
            stats['最大回撤'] = f"{max_drawdown:.2%}"  # 格式化为百分比

        # 记录最终净值
        if '净值' in df.columns and len(df) > 0:
            stats['最终净值'] = f"{df['净值'].iloc[-1]:.4f}"  # 最后一天的净值

        return stats  # 返回统计字典

    def process_file(self, filepath: Path) -> Optional[Dict]:
        """
        处理单个数据文件
        参数：filepath - 文件路径
        返回：处理结果字典或None
        """
        # 从文件名提取股票代码
        symbol = self.extract_symbol(filepath.name)

        # 打印处理开始信息
        print(f"\n{'=' * 60}")  # 分隔线
        print(f"📈 计算收益率: {symbol} ({filepath.name})")  # 股票代码和文件名
        print(f"{'=' * 60}")  # 分隔线

        try:
            # 1. 加载清洗后的数据
            print("  1. 加载清洗后的数据...")
            df = pd.read_excel(filepath)  # 读取Excel文件
            print(f"     数据形状: {df.shape[0]}行 × {df.shape[1]}列")  # 显示数据维度

            # 2. 计算收益率
            print("  2. 计算各种收益率指标...")
            df_returns = self.calculate_returns(df)  # 调用收益率计算函数

            # 显示关键指标
            if '日收益率' in df_returns.columns:  # 如果计算了日收益率
                print(f"     日收益率: 均值={df_returns['日收益率'].mean():.4f}, "
                      f"标准差={df_returns['日收益率'].std():.4f}")

            # 显示累计总收益
            if '累计收益率' in df_returns.columns and len(df_returns) > 0:
                total_return = (df_returns['累计收益率'].iloc[-1] - 1) * 100  # 转换为百分比
                print(f"     累计总收益: {total_return:.2f}%")

            # 显示最大回撤
            if '最大回撤' in df_returns.columns:
                max_dd = df_returns['最大回撤'].min() * 100  # 转换为百分比
                print(f"     最大回撤: {max_dd:.2f}%")

            # 3. 绘制图表
            print("  3. 绘制累计收益曲线...")
            fig, chart_paths = self.plot_cumulative_returns(df_returns, symbol)  # 绘制图表
            print(f"     已生成 {len(chart_paths)} 张图表")  # 显示生成的图表数量

            # 4. 保存计算结果
            print("  4. 保存收益率数据...")
            saved_path = self.save_return_data(df_returns, symbol)  # 保存数据
            # 显示保存信息（相对路径）
            rel_path = saved_path.relative_to(self.project_root)  # 相对路径
            print(f"     💾 已保存: {rel_path}")

            # 5. 返回处理结果
            return {
                'symbol': symbol,  # 股票代码
                'status': 'success',  # 处理状态
                'original_file': filepath.name,  # 原始文件名
                'returns_file': saved_path.name,  # 收益率文件名
                'chart_files': [Path(p).name for p in chart_paths],  # 图表文件名列表
                'total_return': total_return if 'total_return' in locals() else None,  # 总收益
                'max_drawdown': max_dd if 'max_dd' in locals() else None,  # 最大回撤
                'data_points': len(df_returns)  # 数据点数
            }

        except Exception as e:  # 捕获所有异常
            print(f"  ❌ 处理失败: {e}")  # 打印错误信息
            import traceback  # 导入traceback模块
            traceback.print_exc()  # 打印完整的错误堆栈信息
            return {
                'symbol': symbol,  # 股票代码
                'status': 'failed',  # 失败状态
                'error': str(e)  # 错误信息
            }

    def generate_summary_report(self):
        """
        生成汇总报告，显示所有股票的处理结果和排名
        """
        # 分离成功和失败的结果
        successful = [r for r in self.results if r['status'] == 'success']  # 成功列表
        failed = [r for r in self.results if r['status'] == 'failed']  # 失败列表

        # 打印报告标题
        print(f"\n{'=' * 70}")
        print("📋 收益率计算完成报告")
        print(f"{'=' * 70}")

        # 显示处理结果统计
        print(f"📊 处理结果:")
        print(f"  成功处理: {len(successful)} 个文件")
        print(f"  处理失败: {len(failed)} 个文件")

        # 如果有成功的处理结果
        if successful:
            # 按总收益排序（从高到低）
            # 过滤出有总收益数据的结果
            successful_with_return = [r for r in successful if r.get('total_return') is not None]
            # 按总收益降序排序
            successful_sorted = sorted(
                successful_with_return,  # 待排序列表
                key=lambda x: x['total_return'],  # 排序键：总收益
                reverse=True  # 降序排列
            )

            # 打印收益率排名（前10）
            print(f"\n🏆 收益率排名（前10）:")
            # 表头
            print(f"{'排名':<4} {'股票代码':<8} {'累计收益':<12} {'最大回撤':<12} {'数据点数':<10}")
            print(f"{'-' * 50}")  # 分隔线

            # 打印前10名
            for i, result in enumerate(successful_sorted[:10], 1):  # 从1开始编号
                total_return = result.get('total_return', 0)  # 获取总收益，默认为0
                max_dd = result.get('max_drawdown', 0)  # 获取最大回撤，默认为0
                data_points = result.get('data_points', 0)  # 获取数据点数，默认为0

                # 格式化输出
                print(f"{i:<4} {result['symbol']:<8} {total_return:>10.2f}% {max_dd:>10.2f}% {data_points:>10}")

            # 保存详细报告
            self.save_detailed_report(successful)

        # 如果有失败的处理结果
        if failed:
            print(f"\n❌ 处理失败的文件:")
            for i, result in enumerate(failed[:5], 1):  # 最多显示5个失败文件
                print(f"  {i}. {result.get('symbol', '未知')}: {result.get('error', '未知错误')}")

        # 显示输出位置信息
        print(f"\n💡 输出位置:")
        print(f"  收益率数据: data/returns/")
        print(f"  分析图表: charts/")
        print(f"  报告文件: report/")
        print(f"{'=' * 70}")

    def save_detailed_report(self, successful_results: List[Dict]):
        """
        保存详细报告到Excel文件
        参数：successful_results - 成功处理的结果列表
        """
        try:
            report_data = []  # 初始化报告数据列表

            # 遍历成功的结果，提取需要报告的信息
            for result in successful_results:
                report_data.append({
                    '股票代码': result['symbol'],  # 股票代码
                    '原数据文件': result.get('original_file', '未知'),  # 原始文件名
                    '收益率文件': result.get('returns_file', '未知'),  # 收益率文件名
                    '累计收益(%)': result.get('total_return', 0),  # 累计收益
                    '最大回撤(%)': result.get('max_drawdown', 0),  # 最大回撤
                    '数据点数': result.get('data_points', 0),  # 数据点数
                    '图表数量': len(result.get('chart_files', [])),  # 图表数量
                    '处理状态': '成功',  # 处理状态
                    '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 处理时间
                })

            # 如果有报告数据
            if report_data:
                # 创建DataFrame
                report_df = pd.DataFrame(report_data)

                # 生成报告文件名（带时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"收益率计算报告_{timestamp}.xlsx"
                report_path = self.config['reports_dir'] / report_filename

                # 保存到Excel文件
                with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                    # 1. 保存收益率汇总表
                    report_df.to_excel(writer, sheet_name='收益率汇总', index=False)

                    # 2. 保存统计信息表
                    stats_data = [
                        ['统计项', '数值'],  # 表头
                        ['总股票数', len(successful_results)],  # 股票总数
                        ['平均累计收益(%)', report_df['累计收益(%)'].mean()],  # 平均收益
                        ['平均最大回撤(%)', report_df['最大回撤(%)'].mean()],  # 平均最大回撤
                        ['最高收益(%)', report_df['累计收益(%)'].max()],  # 最高收益
                        ['最低收益(%)', report_df['累计收益(%)'].min()],  # 最低收益
                        ['报告生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],  # 生成时间
                        ['收益率数据目录', 'data/returns/'],  # 数据目录
                        ['图表目录', 'charts/'],  # 图表目录
                        ['报告目录', 'report/']  # 报告目录
                    ]

                    # 创建统计信息DataFrame
                    stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                    stats_df.to_excel(writer, sheet_name='统计信息', index=False)

                    # 3. 保存收益分布分析
                    if '累计收益(%)' in report_df.columns:  # 如果有累计收益数据
                        returns_desc = report_df['累计收益(%)'].describe()  # 描述性统计
                        dist_df = pd.DataFrame({
                            '统计量': returns_desc.index,  # 统计量名称
                            '数值': returns_desc.values  # 统计值
                        })
                        dist_df.to_excel(writer, sheet_name='收益分布', index=False)

                # 打印保存成功信息
                rel_path = report_path.relative_to(self.project_root)  # 相对路径
                print(f"📄 详细报告已保存: {rel_path}")

        except Exception as e:  # 捕获异常
            print(f"保存报告时出错: {e}")  # 打印错误信息

    def run(self):
        """
        运行主流程：处理所有文件并生成报告
        """
        # 检查是否有文件需要处理
        if not self.files:  # 如果没有文件
            print(f"\n❌ 没有找到清洗后的数据文件")  # 错误信息
            print(f"   请先运行第4天的数据清洗程序")  # 建议
            return  # 结束函数

        # 打印开始处理信息
        print(f"\n🚀 开始计算 {len(self.files)} 个股票的收益率")
        print(f"{'=' * 60}")

        # 遍历所有文件，逐个处理
        for i, file in enumerate(self.files, 1):  # i从1开始
            print(f"\n[{i}/{len(self.files)}] ", end="")  # 显示进度：[当前/总数]
            result = self.process_file(file)  # 处理单个文件
            if result:  # 如果有结果
                self.results.append(result)  # 添加到结果列表

        # 生成总结报告
        self.generate_summary_report()

        # 显示完成信息
        print(f"\n{'=' * 70}")
        print("🎉 第5天任务完成!")
        print(f"{'=' * 70}")
        print("✅ 已完成的任务:")
        print("  1. 计算日收益率")
        print("  2. 计算累计收益")
        print("  3. 保存计算结果")
        print("  4. 绘制累计收益曲线")
        print(f"{'=' * 70}")

        # 显示最佳和最差表现的股票
        successful = [r for r in self.results if r['status'] == 'success' and r.get('total_return') is not None]
        if successful:  # 如果有成功的结果
            # 找到最佳表现（总收益最高）
            best = max(successful, key=lambda x: x['total_return'])
            # 找到最差表现（总收益最低）
            worst = min(successful, key=lambda x: x['total_return'])

            # 打印结果
            print(f"\n🏆 最佳表现: {best['symbol']} ({best['total_return']:.2f}%)")
            print(f"📉 最差表现: {worst['symbol']} ({worst['total_return']:.2f}%)")
            print(f"{'=' * 70}")


def main():
    """
    主函数：程序入口点
    负责创建ReturnCalculator对象并运行主流程
    """
    try:
        # 创建收益率计算器对象
        calculator = ReturnCalculator()
        # 运行主流程
        calculator.run()

    except KeyboardInterrupt:  # 捕获用户中断（Ctrl+C）
        print("\n\n⚠️ 用户中断操作")  # 打印中断信息
    except Exception as e:  # 捕获其他所有异常
        print(f"\n❌ 程序执行出错: {e}")  # 打印错误信息
        import traceback  # 导入traceback模块
        traceback.print_exc()  # 打印错误堆栈


# Python程序入口点
if __name__ == "__main__":
    main()  # 调用主函数

    # 在Windows系统下，保持命令行窗口打开
    if sys.platform == "win32":  # 检查是否是Windows系统
        input("\n按 Enter 键退出...")  # 等待用户按Enter键



"""
## 第5天：收益率计算与分析

**任务目标**：
- 计算日收益率（简单收益率）
- 计算对数收益率（用于统计分析和模型）
- 计算累计收益率和净值曲线
- 计算滚动收益率（20日）和最大回撤
- 保存计算结果到Excel文件
- 绘制累计收益曲线和多种分析图表

**实现方案**：
1. **数据加载与预处理**：
   - 自动识别并加载第4天清洗后的数据文件
   - 智能识别日期列和收盘价列（支持多种命名格式）
   - 自动创建必要的目录结构（data/returns/, charts/, report/）

2. **收益率计算引擎**：
   - **日收益率计算**：`(今日收盘价 - 昨日收盘价) / 昨日收盘价`
   - **对数收益率计算**：`ln(今日收盘价 / 昨日收盘价)`（符合统计假设）
   - **累计收益率计算**：累积乘积 `∏(1 + 日收益率)`
   - **滚动收益率**：20日滚动收益计算
   - **最大回撤计算**：识别投资期间的最大损失幅度
   - **年化收益率**：基于250个交易日假设进行年化

3. **可视化分析系统**：
   - **中文字体支持**：自动配置Windows中文字体显示
   - **多图表布局**：2×2网格展示价格、收益率分布、累计收益、回撤分析
   - **详细分析图表**：日收益率时间序列、滚动收益率、净值曲线、月度分布
   - **统计信息标注**：在图表中标注均值、标准差、总收益等关键指标

4. **报告生成系统**：
   - **Excel数据保存**：包含收益率数据、统计摘要、分布分析三个sheet
   - **股票排名报告**：按累计收益率自动排序，显示最佳和最差表现股票
   - **详细统计报告**：包含平均收益、最大回撤、正收益比例等关键指标

**核心代码结构**：
```python
class ReturnCalculator:
    ├── __init__()                   # 初始化配置和目录结构
    ├── create_directories()         # 创建收益率、图表、报告目录
    ├── find_cleaned_files()         # 查找第4天的清洗后数据文件
    ├── extract_symbol()             # 从文件名提取股票代码
    ├── calculate_returns()          # 计算所有收益率指标
    ├── plot_cumulative_returns()    # 绘制累计收益曲线和分析图表
    ├── save_return_data()           # 保存计算结果到Excel文件
    ├── calculate_statistics()       # 计算收益率的统计指标
    ├── process_file()               # 处理单个数据文件
    ├── generate_summary_report()    # 生成汇总报告和股票排名
    ├── save_detailed_report()       # 保存详细报告到Excel
    └── run()                        # 运行完整的收益率计算流程
```

"""