'''
第5天：
加入投资组合约束条件，如资产最大权重限制和行业权重限制。
练习：编写包含多个约束的投资组合优化模型。
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 显示中文
plt.rcParams['font.sans-serif'] = ['Simhei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ConstrainedPortfolioOptimizer:
    """
        带约束条件的投资组合优化器
        第5天任务：学习添加各种投资组合约束条件
        """
    def __init__(self, risk_free_rate=0.02):
        """
               初始化投资组合优化器

               Parameters:
               risk_free_rate: 无风险利率，默认2%，用于计算夏普比率
               """
        self.risk_free_rate = risk_free_rate
        self.data = None                    # 存储原始股价数据
        self.returns = None                 # 存储日收益率数据
        self.mu = None                      # 年化预期收益率向量
        self.Sigma = None                   # 年化协方差矩阵
        self.assets = None                  # 有效资产列表
        self.industry_mapping = {}          # 资产行业分类映射
        self.data_days = {}                 # 每个资产的数据天数

    def load_stock_data_from_current_dir(self, min_days=500):
        """
        从当前目录加载股票数据，并进行质量过滤

        Parameters:
        min_days: 最小数据天数要求，默认500天（约2年）

        Returns:
        bool: 数据加载是否成功"""
        print(f"正在从当前目录加载股票数据...")
        all_data = {}
        valid_ticker = []

        # 使用glob扫描所有符合命名规则的股票数据文件
        stock_files = glob.glob('./*_stock_data.xlsx')

        if not stock_files:
            print(f"错误: 当前目录下未找到股票数据文件...")
            print(f"请确认文件命格式为: ./AAPL_stock_data.xlsx")
            return False

        print(f"找到{len(stock_files)}个股票数据文件")

        # 逐个文件处理，包含异常处理
        for file_path in stock_files:
            filename = os.path.basename(file_path)
            ticker = filename.replace('_stock_data.xlsx', '')
            try:
                # 读取Excel文件，第一列为日期索引
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)

                # 数据质量检查：必须有Close列且数据量足够
                if 'Close' in df.columns and len(df) >= min_days:
                    df = df.sort_index()    # 确保时间顺序正确
                    all_data[ticker] = df['Close']
                    valid_ticker.append(ticker)
                    self.data_days[ticker] = len(df)
                    print(f"加载{ticker}数据成功({len(df)}天)")
                else:
                    days = len(df) if 'Close' in df.columns else 0
                    print(f"{ticker}: 数据不足({days}天, 需要{min_days}天)")

            except Exception as e:
                print(f"加载{ticker}失败:{e}")

        # 检查是否有足够股票进行优化
        if len(valid_ticker) < 5:
            print(f"错误: 需要至少5只股票进行组合优化, 当前只有{len(valid_ticker)}只")
            return False

        # 创建完整的数据DataFrame
        self.data = pd.DataFrame(all_data)
        self.data = self.data.sort_index()    # 整体排序
        self.data = self.data.ffill().dropna()      # 处理缺失值：前向填充后删除仍有缺失的行

        # 计算收益率和统计量
        self.returns = self.data.pct_change().dropna()
        self.assets = self.data.columns.tolist()

        # 计算年化预期收益率和协方差矩阵
        self.mu = self.returns.mean() * 252      # 年化：日收益率均值 × 252个交易日
        self.Sigma = self.returns.cov() * 252   # 年化：日收益率协方差 × 252

        # 创建行业映射
        self.industry_mapping = self.smart_industry_detection(self.assets)

        # 输出数据汇总信息
        total_days = len(self.data)
        date_range = self.data.index[-1] - self.data.index[0]
        years = date_range.days / 365.25

        print(f"数据加载完毕!")
        print(f'有效股票数量; {len(self.assets)}')
        print(f"   交易日数: {len(self.data)}")
        print(f"   时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 到 "
              f"{self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   数据覆盖: {years:.1f} 年")

        # 显示行业分布
        self.display_industry_distribution()
        return True

    def smart_industry_detection(self, assets):
        """
               基于股票代码和常见知识的智能行业推测

               Parameters:
               assets: 资产代码列表

               Returns:
               dict: 资产到行业的映射字典
               """
        industry_mapping = {}

        # 已知的知名公司行业映射 - 基于公开信息
        known_companies = {
            # 科技公司 - 硬件、软件、互联网
            'AAPL': '科技', "INTC": "科技", 'LSCC': '科技', 'TTD': "科技",
            'PLTR': '科技', 'CFLT': '科技', "MSTR": '科技', 'SOUN': "科技",

            # 互联网/电商公司
            'BABA': '科技', 'BIDU': '科技',

            # 半导体/硬件相关
            'LAC': '原材料',  # 锂业公司，属于新能源原材料

            # 汽车/新能源
            'LCID': '汽车',  # Lucid Motors 电动汽车
            'PLUG': '能源',  # Plug Power 氢能源

            # 金融/券商
            'HOOD': '金融',  # Robinhood 券商平台
            'SCHD': '金融',  # Schwab US Dividend Equity ETF

            # 消费/零售
            'SBUX': '消费', 'LULU': '消费', 'DIS': '消费',

            # 工业/制造业
            'BA': '工业',  # Boeing 航空航天
            'GE': '工业',  # General Electric 综合工业

            # 医疗健康
            'UNH': '医疗',  # UnitedHealth 医疗保险
            'HIMS': '医疗',  # Hims & Hers Health 远程医疗
            'ARCT': '医疗',  # Arcturus Therapeutics 生物制药

            # ETF和特殊产品
            'VOO': 'ETF',  # Vanguard S&P 500 ETF
            'KWEB': 'ETF',  # KraneShares CSI China Internet ETF
            'VXX': '衍生品',  # iPath Series B S&P 500 VIX Short-Term Futures ETN

            # 媒体/娱乐
            'DJT': '媒体',  # Trump Media & Technology Group
        }

        # 首先匹配已知公司
        for asset in assets:
            if asset in known_companies:
                industry_mapping[asset] = known_companies[asset]
            else:
                # 基于名称关键词推测未知公司
                asset_upper = asset.upper()
                if any(keyword in asset_upper for keyword in ['BANK', 'FIN', 'CREDIT', 'CAPITAL']):
                    industry_mapping[asset] = '金融'
                elif any(keyword in asset_upper for keyword in ['TECH', 'SOFT', 'DATA', 'CLOUD', 'AI',
                                                                'DIGITAL']):
                    industry_mapping[asset] = '科技'
                elif any(keyword in asset_upper for keyword in ['MED', 'BIO', 'HEALTH', 'CARE', 'PHARMA',
                                                                'LIFE']):
                    industry_mapping[asset] = '医疗'
                elif any(keyword in asset_upper for keyword in ['OIL', 'GAS', 'ENERGY', 'POWER', 'FULE']):
                    industry_mapping[asset] = '能源'
                elif any(keyword in asset_upper for keyword in ['AUTO', 'CAR', 'VEHICLE', 'MOTOR']):
                    industry_mapping[asset] = '汽车'
                elif any(keyword in asset_upper for keyword in ['METAL', 'MINING', 'MATERIAL', 'RESOURCE']):
                    industry_mapping[asset] = '原材料'
                else:
                    industry_mapping[asset] = '其他'  # 默认分类
        return industry_mapping

    def display_industry_distribution(self):
        """
                显示资产的行业分布情况
                """
        industry_counts = pd.Series(self.industry_mapping).value_counts()
        print(f"\n📊 行业分布:")

        for industry, count in industry_counts.items():
            # 获取该行业的所有股票
            stocks = [k for k, v in self.industry_mapping.items() if v == industry]
            # 显示前5只股票，超过5只用...表示
            stocks_display = ', '.join(stocks[:5]) + ('...' if len(stocks) > 5 else '')
            print(f"{industry}: {count}只股票 - {stocks_display}")

    def basic_constrained_optimization(self):
        """
                基础约束优化 - 只有最基本的权重约束
                体现投资组合管理的最基本要求

                Returns:
                dict: 优化结果包含权重、收益、风险等指标
                """
        print("\n" + "=" * 60)
        print("基础约束优化")
        print("=" * 60)
        print("约束条件: 权重和=1, 不允许卖空, 单股≤15%, 收益≥8%")

        n = len(self.assets)
        w = cp.Variable(n)       # 定义优化变量：n维权重向量

        # 目标函数：最小化投资组合风险
        portfolio_risk = cp.quad_form(w, self.Sigma.values) # w^T Σ w
        portfolio_return = w @ self.mu.values       # w^T μ

        # 基础约束条件 - 投资组合管理的基本要求
        constraints = [
            cp.sum(w) == 1,              # 权重和为1 - 完全投资约束
            w >= 0,                      # 不允许卖空 - 非负约束
            w <= 0.15,                   # 单个资产最大权重15% - 分散化约束
            portfolio_return >= 0.08    # 最低收益要求8% - 收益目标约束
        ]

        # 构建优化问题：最小化风险
        objective = cp.Minimize(portfolio_risk)
        problem = cp.Problem(objective, constraints)

        # 求解优化问题
        problem.solve()

        # 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败: 状态: {problem.status}")
            return None

        # 处理优化结果
        weights_value = w.value
        weights = pd.Series(weights_value, index=self.assets)

        # 计算绩效指标 - 使用numpy数组确保维度一致
        actual_return = np.dot(weights_value, self.mu.values)
        actual_risk = np.sqrt(weights_value @ self.Sigma.values @ weights_value)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk > 0 else 0

        # 只用于显示的过滤权重（计算使用完整权重）
        display_weights = weights[weights > 0.001]

        print(f"✅ 基础约束优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   有效股票数量: {len(display_weights)}")

        # 显示前5大权重股票
        if len(display_weights) > 0:
            top_5 = display_weights.nlargest(5)
            print(f"   前5大权重: {', '.join([f'{asset}({weight:.1%})' for asset, weight in top_5.items()])}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': '基础约束优化'
        }

    def industry_constrained_optimization(self):
        """
                行业权重约束优化 - 在基础约束上添加行业层面的约束
                体现行业风险控制和资产配置策略

                Returns:
                dict: 优化结果包含权重、收益、风险、行业分布等指标
                """
        print("\n" + "=" * 60)
        print("行业权重约束优化")
        print("=" * 60)
        print("约束条件: 行业权重限制 + 基础约束")

        n = len(self.assets)
        w = cp.Variable(n)      # 定义优化变量

        # 目标函数：最小化投资组合风险
        portfolio_risk = cp.quad_form(w, self.Sigma.values)
        portfolio_return = w @ self.mu.values

        # 基础约束条件（比基础优化更严格）
        constraints = [
            cp.sum(w) == 1,
            w >= 0.01,          # 最低权重1% - 避免过于分散
            w <= 0.12,          # 最高权重12% - 比基础更严格
            portfolio_return >= 0.10        # 收益要求10% - 比基础更高
        ]

        # 行业权重约束 - 核心新增内容
        industries = set(self.industry_mapping.values())
        industry_constraints_info = []      # 记录约束信息用于显示

        for industry  in industries:
            # 获取该行业的所有资产索引
            industry_indices = [i for i, asset in enumerate(self.assets)
                                if self.industry_mapping[asset] == industry]

            if industry_indices:
                # 计算该行业的总权重
                industry_weight = cp.sum([w[i] for i in industry_indices])
                # 根据不同行业特点设置不同的权重限制
                if industry == '科技':
                    # 科技行业：成长性强但波动大，给予较大但有限的范围
                    constraints.append(industry_weight <= 0.45)  # 上限45%
                    constraints.append(industry_weight >= 0.25) # 下限25%
                    industry_constraints_info.append(f"科技: 25% - 45%")
                elif industry == '金融':
                    # 金融行业：稳定性较好，作为基础配置
                    constraints.append(industry_weight <= 0.25)  # 上限25%
                    constraints.append(industry_weight >= 0.10)  # 下限10%
                    industry_constraints_info.append(f"金融: 10% - 25%")
                elif industry == '医疗':
                    # 医疗行业：防御性强但专业性高，控制上限
                    constraints.append(industry_weight <= 0.20)  # 上限20%
                    industry_constraints_info.append(f"医疗: <= 20%")
                elif industry == 'ETF':
                    # ETF产品：工具性产品，严格限制
                    constraints.append(industry_weight <= 0.15)  # 上限15%
                    industry_constraints_info.append(f"ETF: <= 15%")
                elif industry == '衍生品':
                    # 衍生品：高风险产品，严格限制
                    constraints.append(industry_weight <= 0.05)      # 上限5%
                    industry_constraints_info.append(f"衍生品: <= 5%")
                else:
                    # 其他行业：统一上限控制
                    constraints.append(industry_weight <= 0.25) # 上限25%
                    industry_constraints_info.append(f"{industry}: <= 25%")
        # 显示设置的行业约束
        print(f" 行业约束: {', '.join(industry_constraints_info)}")

        # 构建并求解优化问题
        objective = cp.Minimize(portfolio_risk)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败: {problem.status}")
            return None

        # 处理优化结果
        weights_values = w.value
        weights = pd.Series(weights_values, index=self.assets)

        # 计算绩效指标
        actual_return = np.dot(weights_values, self.mu.values)
        actual_risk = np.sqrt(weights_values @ self.Sigma.values @ weights_values)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk > 0 else 0

        # 计算行业权重分布 - 行业约束的核心输出
        industry_weights = {}
        for asset, weight in weights.items():
            industry = self.industry_mapping[asset]
            industry_weights[industry] = industry_weights.get(industry, 0) + weight

        # 只用于显示的过滤权重
        display_weights = weights[weights > 0.001]

        print(f"✅ 行业约束优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   有效股票数量: {len(display_weights)}")

        # 显示实际行业权重分布
        print(f"\n📊 实际行业权重:")
        for industry, weight in sorted(industry_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.001: # 只显示有权重的行业
                print(f"{industry}: {weight:.1%}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'industry_weights': industry_weights, # 新增行业权重信息
            'method': '行业约束优化'
        }

    def advanced_constrained_optimization(self):
        """
                高级多重约束优化 - 综合各种约束条件
                体现专业投资组合管理的完整风控框架

                Returns:
                dict: 优化结果包含权重、收益、风险、集中度等综合指标
                """
        print("\n" + "=" * 60)
        print("高级多重约束优化")
        print("=" * 60)
        print("约束条件: 行业限制 + 集中度控制 + 换手率限制")

        n = len(self.assets)
        w = cp.Variable(n)

        # 目标函数：风险厌恶型 - 权衡风险和收益
        portfolio_risk = cp.quad_form(w, self.Sigma.values)
        portfolio_return = w @ self.mu.values

        # 1. 基础约束（最严格版本）
        constraints = [
            cp.sum(w) == 1,
            w >= 0.02,              # 最低权重2% - 避免过度分散
            w <= 0.10,              # 最高权重10% - 严格分散化
            portfolio_return >= 0.12        # 收益目标12% - 较高要求
        ]

        # 2. 行业约束（精细化版本）
        industries = set(self.industry_mapping.values())
        for industry in industries:
            industry_indices = [i for i, asset in enumerate(self.assets)
                                if self.industry_mapping[asset] == industry]
            if industry_indices:
                industry_weight = cp.sum([w[i] for i in industry_indices])
                # 更精细的行业控制
                if industry == '科技':
                    constraints.append(industry_weight <= 0.40)     # 科技上限40%
                    constraints.append(industry_weight >= 0.20)     # 科技下限20%
                elif industry == '金融':
                    constraints.append(industry_weight <= 0.20)  # 金融上限20%
                    constraints.append(industry_weight >= 0.08)  # 金融下限8%
                elif industry in ['ETF', '衍生品']:
                    constraints.append(industry_weight <= 0.10) # 严格限制特殊产品
                elif industry == '医疗':
                    constraints.append(industry_weight <= 0.18)      # 医疗上限18%
                    constraints.append(industry_weight >= 0.05)     # 医疗下限5%

        # 3. 风险集中度约束（赫芬达尔指数）
        herfindahl_index = cp.sum_squares(w)     # 赫芬达尔指数计算
        constraints.append(herfindahl_index <= 0.08)    # 集中度上限8%

        # 4. 组合目标函数（风险收益权衡）
        risk_aversion = 0.3     # 风险厌恶系数
        objective = cp.Minimize(portfolio_risk - risk_aversion * portfolio_return)

        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败: {problem.status}")
            return None

        # 处理优化结果
        weights_values = w.value
        weights = pd.Series(weights_values, index= self.assets)

        # 计算绩效指标
        actual_return = np.dot(weights_values, self.mu.values)
        actual_risk = np.sqrt(weights_values @ self.Sigma.values @ weights_values)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk >0 else 0

        # 计算各种风险指标
        industry_weights = {}
        for asset, weight in weights.items():
            industry = self.industry_mapping[asset]
            industry_weights[industry] = industry_weights.get(industry, 0) + weight

        concentration = herfindahl_index.value  # 集中度指数

        print(f"✅ 高级约束优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   集中度指数: {concentration:.3f}")
        print(f"   有效股票数量: {len(weights[weights > 0.001])}")

        # 显示行业权重分布
        print(f"\n📊 实际行业权重:")
        for industry, weight in sorted(industry_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01: # 只显示权重大于1%的行业
                print(f"{industry}: {weight:.1%}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'industry_weights': industry_weights,
            'concentration': concentration, # 新增集中度指标
            'method': '高级约束优化'
        }

    def compare_constrained_methods(self):
        """
                比较不同约束优化方法的结果
                提供全面的性能对比分析

                Returns:
                dict: 所有优化方法的结果字典
                """
        print("\n" + "=" * 80)
        print("🎯 不同约束优化方法对比分析")
        print("=" * 80)

        results = {}

        # 1. 基础约束优化 - 建立基准
        print("\n1. 执行基础约束优化...")
        results['basic'] = self.basic_constrained_optimization()

        # 2. 行业约束优化 - 添加行业风控
        print("\n2. 执行行业约束优化...")
        results['industry'] = self.industry_constrained_optimization()

        # 3. 高级多重约束优化 - 综合风控框架
        print("\n3. 执行高级多重约束优化...")
        results['advanced'] = self.advanced_constrained_optimization()

        # 创建对比表格 - 核心分析输出
        comparsion_data = []
        for key, result in results.items():
            if result is not None:
                # 确保所有必要的键都存在
                required_keys = ['method', 'expected_return', 'risk', 'sharpe_ratio', 'weights']
                if all(k in result for k in required_keys):
                    comparsion_data.append({
                        '优化方法': result['method'],
                        '年化收益率': f"{result['expected_return']:.2%}",
                        '年化波动率': f"{result['risk']:.2%}",
                        '夏普比率': f"{result['sharpe_ratio']:.2f}",
                        '股票数量': len(result['weights'][result['weights']>0.001]),
                        '前3大资产': self._get_top_assets_str(result['weights'])
                    })
                else:
                    print(f"警告: {key} 优化结果缺少必要字段")

        # 显示对比表格
        if comparsion_data:
            comparsion_df = pd.DataFrame(comparsion_data)
            print(f"\n 约束优化方法对比: ")
            print(comparsion_df.to_string(index=False))

        return results

    def _get_top_assets_str(self, weights, n=3):
        """
                辅助方法：获取前n大权重资产的格式化字符串

                Parameters:
                weights: 权重Series
                n: 显示前几名

                Returns:
                str: 格式化字符串
                """
        top_assets = weights.nlargest(n)
        return ', '.join([f"{asset}({weight:.1%})" for asset, weight in top_assets.items()])

    def plot_constraint_comparsion(self, results):
        """
                绘制约束优化方法的对比图表
                提供直观的可视化分析

                Parameters:
                results: 优化结果字典
                """
        if not results:
            print(f"没有可能的优化结果进行绘图!")
            return

        # 图表1：风险收益散点图 - 核心绩效对比
        plt.figure(figsize=(10,6))
        methods = []
        returns = []
        risks = []
        sharpe_ratio = []

        # 提取数据用于绘图
        for key, result in results.items():
            if result is not None:
                methods.append(result['method'])
                returns.append(result['expected_return'])
                risks.append(result['risk'])
                sharpe_ratio.append(result['sharpe_ratio'])

        # 绘制散点图，颜色表示夏普比率
        scatter = plt.scatter(risks, returns, c=sharpe_ratio, cmap='viridis', s=100, alpha=0.07)
        for i, method in enumerate(methods):
            plt.annotate(method, (risks[i], returns[i]), xytext=(5,5), textcoords='offset points',
                            fontsize=9)
        plt.xlabel('年化波动率(风险)')
        plt.ylabel('年化收益率')
        plt.title('不同约束优化方法的风险收益率', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='夏普比率')
        plt.tight_layout()
        plt.show()

        # 图表2：行业权重分布对比 - 行业约束效果可视化
        industry_results = {k: v for k, v in results.items()
                            if v is not None and 'industry_weights' in v}

        if len(industry_results) >= 2:
            fig, axes = plt.subplots(1, len(industry_results), figsize=(15,6))
            if len(industry_results) == 1:
                axes = [axes]

            for idx, (key, result) in enumerate(industry_results.items()):
                industry_weights = result['industry_weights']
                # 只显示权重大于2%的行业，避免图表过于复杂
                filtered_weights = {k: v for k, v in industry_weights.items() if v > 0.02}

                # 确保有数据才绘制
                if filtered_weights:
                    axes[idx].pie(filtered_weights.values(), labels=filtered_weights.keys(),
                              autopct=lambda p: f'{p:.1f}' if p >= 1 else '', startangle=90)
                    axes[idx].set_title(f"{result['method']}\n行业权重分布", fontsize=10)
                else:
                    axes[idx].text(0.5, 0.5, '无行业权重数据',
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=axes[idx].transAxes, fontsize=12)
                    axes[idx].set_title(f"{result['method']}\n行业权重分布", fontsize=10)

            plt.tight_layout()
            plt.show()

    def explain_constraint_types(self):
        """
                解释不同类型的约束条件
                提供理论背景知识教育
                """
        print("\n" + "=" * 80)
        print("📚 投资组合约束条件类型解释")
        print("=" * 80)

        constraints_info = {
            "权重和约束": "所有资产权重之和必须等于1 (∑w_i = 1)，确保完全投资",
            "非负约束": "不允许卖空操作 (w_i ≥ 0)，所有权重必须非负",
            "单个资产权重限制": "限制单只股票的最大权重 (w_i ≤ max_weight)，避免过度集中",
            "行业权重限制": "限制特定行业的总体权重，控制行业风险暴露",
            "最低收益要求": "确保投资组合达到最低预期收益水平",
            "集中度约束": "限制投资组合的集中程度（如赫芬达尔指数）",
            "换手率约束": "限制权重变化幅度，控制交易成本",
            "流动性约束": "考虑资产的流动性限制（基于交易量等指标）",
            "ESG约束": "基于环境、社会和治理评分的投资限制",
            "因子暴露约束": "控制投资组合对特定风险因子的暴露程度"
        }

        for constraint, explanation in constraints_info.items():
            print(f"{constraint}: {explanation}")

def main():
    """
        主函数 - 第5天任务执行
        协调完整的优化分析流程
        """
    print('=' * 70)
    print("第5天：投资组合约束条件优化")
    print("加入资产最大权重限制和行业权重限制等约束条件")
    print('=' * 70)

    # 创建约束优化器实例
    optimizer = ConstrainedPortfolioOptimizer(risk_free_rate=0.02)
    #  加载数据（过滤掉数据量不足的股票）
    if optimizer.load_stock_data_from_current_dir(min_days=500):

        # 解释约束条件类型 - 理论教育
        optimizer.explain_constraint_types()

        # 执行不同约束优化方法的对比 - 核心分析
        print(f"\n🚀 开始执行多种约束优化方法对比...")
        results = optimizer.compare_constrained_methods()

        # 生成可视化图表 - 结果展示
        print(f"\n📊 正在生成约束优化方法对比图表...")
        optimizer.plot_constraint_comparsion(results)

        print('\n' + '=' * 70)
        print("✅ 第5天任务完成！")
        print("   成功实现多种投资组合约束条件优化")
        print("   理解了行业权重限制和多重约束的设计")
        print('=' * 70)


if __name__ == "__main__":
    main()


"""
🎯 任务目标：
1. 学习投资组合优化中的各种约束条件设计
2. 实现从基础到高级的多层次约束优化
3. 分析约束条件对投资组合绩效的影响

📚 核心概念：
- 权重约束：控制单个资产的最大最小权重
- 行业约束：控制行业层面的风险暴露
- 集中度约束：使用赫芬达尔指数衡量组合分散程度
- 风险收益权衡：在控制风险的同时追求收益

🔧 技术栈：
- CVXPY: 凸优化求解
- Pandas: 数据处理
- NumPy: 数值计算
- Matplotlib: 结果可视化
"""
















