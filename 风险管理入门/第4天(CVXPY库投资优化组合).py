'''
第4天：
学习cvxpy，理解凸优化中的约束条件和目标函数设计。
练习：用cvxpy实现带有风险惩罚项的自定义投资组合优化。
'''

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 是学习cvxpy库.  是专门用于优化投资组合的.
class CVXPortfolioOptimizer:
    """
      使用CVXPY实现自定义投资组合优化
      CVXPY是专门用于凸优化的Python库，可以轻松解决各种约束优化问题
      """
    def __init__(self, risk_free_rate=0.02):
        """
               初始化优化器

               参数说明：
               risk_free_rate: 无风险利率，用于计算夏普比率，默认2%
               """
        self.risk_free_rate = risk_free_rate
        self.data = None            # 存储原始股价数据
        self.returns = None         # 存储收益率数据
        self.mu = None              # 预期收益率向量
        self.Sigma = None           # 协方差矩阵
        self.assets = None          # 资产列表

    def load_stock_data_from_current_dir(self):
        """
                从当前目录加载股票数据
                文件命名格式：AAPL_stock_data.xlsx
                """
        print("正在从当前目录加载股票数据...")
        all_data = {}
        valid_tickers = []

        # 使用glob查找所有股票数据文件
        stock_files = glob.glob('./*_stock_data.xlsx')

        if not stock_files:
            print("错误: 当前目录下未找到股票数据文件")
            print("请确保文件命名格式为: ./AAPL_stock_data.xlsx")
            return False

        print(f"找到{len(stock_files)}个股票数据文件")

        # 遍历每个文件并加载数据
        for file_path in stock_files:
            filename = os.path.basename(file_path)
            ticker = filename.replace('_stock_data.xlsx', '')

            try:
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                # 检查数据有效性
                if 'Close' in df.columns and len(df) >500:
                    df = df.sort_index()
                    date_range = df.index[-1] - df.index[0]
                    years = date_range.days / 365.25

                    # 要求数据至少覆盖2年
                    if years >=2:
                        all_data[ticker] = df['Close']
                        valid_tickers.append(ticker)
                        print(f"✓ 加载 {ticker} 数据成功 ({len(df)} 天, {years:.1f} 年)")
                    else:
                        print(f"✗ {ticker}: 数据时间范围不足 ({years:.1f} 年)")
                else:
                    print(f"✗ {ticker}: 数据无效或数据点不足 ({len(df)} 天)")
            except Exception as e:
                print(f"✗ 加载 {ticker} 失败: {e}")

        # 检查是否有足够股票进行优化
        if len(valid_tickers) < 2:
            print(f"错误: 需要至少2只股票进行组合优化，当前只有 {len(valid_tickers)} 只")
            return False
        # 将数据转换为DataFrame
        self.data = pd.DataFrame(all_data)
        self.data = self.data.sort_index()
        self.data = self.data.ffill().dropna()  ## 处理缺失值

        if len(self.data) <500:
            print(f"错误: 合并后数据量不足，至少需要500个交易日，当前只有 {len(self.data)} 天")
            return False

        # 计算收益率和统计量
        self.returns = self.data.pct_change().dropna()
        self.assets = self.data.columns.tolist()

        # 计算年化预期收益率和协方差矩阵
        self.mu = self.returns.mean() * 252         # 年化预期收益率
        self.Sigma = self.returns.cov() * 252       # 年化协方差矩阵

        # 输出数据汇总信息
        total_days = len(self.data)
        date_range = self.data.index[-1] - self.data.index[0]
        years = date_range.days / 365.25

        print(f"\n✅ 数据加载完成!")
        print(f"   有效股票数量: {len(self.assets)}")
        print(f"   交易日数: {len(self.data)}")
        print(f"   时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 到 "
              f"{self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   数据覆盖: {years:.1f} 年")
        return True

    def basic_mean_variance_optimization(self, target_return=None):
        """
                基础均值-方差优化（马科维茨模型）
                这是投资组合优化的经典方法

                数学形式：
                最小化: w^T Σ w (投资组合方差/风险)
                约束条件:
                    w^T μ ≥ 目标收益 (如果指定)
                    ∑w_i = 1 (权重和为1)
                    w_i ≥ 0 (不允许卖空)
                    w_i ≤ 0.4 (单个资产最大权重40%)
                """
        print("\n" + "=" * 60)
        print("基础均值-方差优化 (马科维茨模型)")
        print("=" * 60)

        n = len(self.assets)

        # 1. 定义优化变量 - 投资组合权重
        # cp.Variable(n) 创建n维优化变量，代表各资产的配置比例
        w = cp.Variable(n)

        # 2. 定义投资组合的预期收益和风险
        # w @ self.mu.values 计算投资组合预期收益 (向量点积)
        # A @ B: 这是矩阵乘法
        portfolio_return = w @ self.mu.values

        # cp.quad_form(w, Sigma) 计算 w^T Σ w，即投资组合方差
        portfolio_risk = cp.quad_form(w, self.Sigma.values)

        # 3. 定义约束条件
        constraints = [
            cp.sum(w) == 1,      # 权重和为1 (100%)
            w >= 0,              # 不允许卖空 (权重非负)
            w <= 0.4            # 单个资产最大权重40%
        ]

        # 如果指定了目标收益，添加收益约束
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)

        # 4. 定义目标函数：最小化风险
        objective = cp.Minimize(portfolio_risk)

        # 5. 创建优化问题并求解
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # 6. 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败! 状态: {problem.status}")
            return None

        # 7. 提取优化结果
        weights = pd.Series(w.value, index=self.assets)

        # 8. 计算绩效指标
        actual_return = weights @ self.mu.values
        actual_risk = np.sqrt(weights.values @ self.Sigma.values @ weights.values)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk >0 else 0

        print(f"✅ 优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': '基础均值-方差'
        }

    def max_sharpe_optimization(self):
        """
                最大夏普比率优化
                目标是最大化 (收益 - 无风险利率) / 风险

                由于夏普比率不是凸函数，我们通过变量替换将其转化为凸问题：
                令 k > 0, 且 w = x / k
                则原问题转化为：
                最小化 x^T Σ x
                约束: (μ - r_f)^T x = 1
                     ∑x_i = k
                     x_i ≥ 0
                """
        print("\n" + "=" * 60)
        print("最大夏普比率优化")
        print("=" * 60)

        n = len(self.assets)
        mu_vec = self.mu.values
        excess_return = mu_vec - self.risk_free_rate    # 超额收益

        # 1. 定义优化变量
        x = cp.Variable(n)      # 辅助变量
        k = cp.Variable()          # 缩放变量

        # 2. 定义投资组合风险
        portfolio_risk = cp.quad_form(x, self.Sigma.values)

        # 3. 定义约束条件
        constraints = [
            excess_return @ x == 1,     # 超额收益归一化
            cp.sum(x) == k,             # 权重和为k
            x >= 0,                     # 非负权重
            x <= 0.4 * k,               # 单个资产最大权重40%
            k >= 1e-6                   # k必须为正
        ]

        # 4. 目标函数：最小化风险
        objective = cp.Minimize(portfolio_risk)

        # 5. 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败! 状态: {problem.status}")
            return None

        # 6. 计算实际权重: w = x / k
        if abs(k.value) > 1e-6:
            weights_values = x.value / k.value
        else:
            weights_values = x.value

        # 创建完整的权重Series
        weights = pd.Series(0.0, index=self.assets)
        for i, asset in enumerate(self.assets):
            if i < len(weights_values):
                weights[asset] = weights_values[i]

        # 7. 计算绩效指标
        actual_return = weights @ mu_vec
        actual_risk = np.sqrt(weights.values @ self.Sigma.values @ weights.values)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk >0 else 0

        print(f"✅ 最大夏普优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': '最大夏普比率'
        }

    def custom_risk_penalty_optimization(self, risk_aversion=1.0, turnover_penalty=0.1):
        """
               自定义风险惩罚项优化
               在基础均值-方差模型上添加各种惩罚项

               目标函数: 最小化 [基础风险 + 风险厌恶×下行风险 + 集中度惩罚 + 换手率惩罚]

               参数说明：
               risk_aversion: 风险厌恶系数，越大表示越厌恶风险
               turnover_penalty: 换手率惩罚系数，控制权重变化幅度
               """
        print("\n" + "=" * 60)
        print("自定义风险惩罚项优化")
        print("=" * 60)
        print(f"风险厌恶系数: {risk_aversion}")
        print(f"换手率惩罚系数: {turnover_penalty}")

        n = len(self.assets)

        # 1. 定义优化变量
        w = cp.Variable(n)

        # 2. 基础项: 投资组合风险 (方差)
        portfolio_risk = cp.quad_form(w, self.Sigma.values)

        # 3. 自定义惩罚项1: 下行风险惩罚
        # 使用历史收益率的负部分计算下行风险
        negative_returns = np.minimum(self.returns.values, 0)   # 只取负收益
        downside_risk = cp.quad_form(w, negative_returns.T @ negative_returns / len(self.returns)*252)

        # 4. 自定义惩罚项2: 权重集中度惩罚 (赫芬达尔指数)
        # 惩罚权重过于集中，促进分散化投资
        concentration_penalty = cp.sum_squares(w)       # 改为 sum_squares

        # 5. 自定义惩罚项3: 换手率惩罚
        # 假设初始权重为等权重，惩罚权重变化幅度
        w0 = np.ones(n) / n     # 初始等权重
        turnover = cp.norm(w - w0, 1)       # L1范数衡量权重变化

        # 6. 组合预期收益
        portfolio_return = w @ self.mu.values

        # 7. 约束条件
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.4,
            portfolio_return >= self.risk_free_rate # 至少获得无风险收益
        ]

        # 8. 复合目标函数
        objective = cp.Minimize(
            portfolio_risk +
            risk_aversion * downside_risk +
            0.5 * concentration_penalty +
            turnover_penalty * turnover
        )

        # 9. 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"优化失败! 状态: {problem.status}")
            return None

        # 10. 提取结果
        weights_values = w.value
        weights = pd.Series(weights_values, index=self.assets)

        # 11. 计算各项指标
        actual_return = weights @ self.mu.values
        actual_risk = np.sqrt(weights.values @ self.Sigma.values @ weights.values)
        sharpe_ratio = (actual_return - self.risk_free_rate) / actual_risk if actual_risk >0 else 0

        # 计算惩罚项的具体数值
        downside_risk_value = downside_risk.value
        concentration_value = concentration_penalty.value
        turnover_value = turnover.value

        print(f"✅ 自定义优化成功!")
        print(f"   投资组合预期收益: {actual_return:.2%}")
        print(f"   投资组合风险: {actual_risk:.2%}")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   下行风险惩罚项: {downside_risk_value:.6f}")
        print(f"   集中度惩罚项: {concentration_value:.6f}")
        print(f"   换手率惩罚项: {turnover_value:.6f}")

        return {
            'weights': weights,
            'expected_return': actual_return,
            'risk': actual_risk,
            'sharpe_ratio': sharpe_ratio,
            'downside_risk': downside_risk_value,
            'concentration': concentration_value,
            'turnover': turnover_value,
            'method': '自定义风险惩罚'  # 这个会在后面被重命名
        }

    def compare_optimization_methods(self):
        """
                比较不同优化方法的结果
                """
        print("\n" + "=" * 80)
        print("🎯 不同优化方法对比分析")
        print("=" * 80)

        results = {}

        # 1. 基础均值-方差优化
        print("\n1. 执行基础均值-方差优化...")
        results['basic_mv'] = self.basic_mean_variance_optimization()

        # 2. 最大夏普比率优化
        print("\n2. 执行最大夏普比率优化...")
        results['max_sharpe'] = self.max_sharpe_optimization()

        # 3. 自定义风险惩罚优化 (低风险厌恶)
        print("\n3. 执行自定义风险惩罚优化 (低风险厌恶)...")
        custom_low = self.custom_risk_penalty_optimization(
            risk_aversion=1.0, turnover_penalty=0.1
        )
        if custom_low is not None:
            custom_low['method'] = '自定义低风险'  # 清晰命名
            results['custom_low'] = custom_low

        # 4. 自定义风险惩罚优化 (高风险厌恶)
        print("\n4. 执行自定义风险惩罚优化 (高风险厌恶)...")
        custom_high = self.custom_risk_penalty_optimization(
            risk_aversion=2.0, turnover_penalty = 0.2
        )
        if custom_high is not None:
            custom_high['method'] = '自定义高风险'  # 清晰命名
            results['custom_high'] = custom_high

        # 创建对比表格
        comparison_data = []
        for key, result in results.items():
            if result is not None:
                comparison_data.append({
                    '优化方法': result['method'],
                    '年化收益率': f"{result['expected_return']:.2%}",
                    '年化波动率': f"{result['risk']:.2%}",
                    '夏普比率': f"{result['sharpe_ratio']:.2%}",
                    '前3大资产': self._get_top_assets_str(result['weights'])
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\n📊 优化方法对比:")
            print(comparison_df.to_string(index=False))

        return results

    def _get_top_assets_str(self, weights, n=3):
        """获取前n大权重资产的字符串表示"""
        top_assets = weights.nlargest(n)
        return ", ".join([f"{asset}({weight:.1%})" for asset, weight in top_assets.items()])

    def plot_optimization_comparison(self, results):
        """
                绘制不同优化方法的对比图表 - 分开显示，避免拥挤
                """
        if not results:
            print("没有可用的优化结果进行绘图")
            return
        # 图表1：风险收益散点图
        plt.figure(figsize=(10, 6))
        methods = []
        returns = []
        risks = []
        sharpe_ratio = []

        for key, result in results.items():
            if result is not None:
                methods.append(result['method'])
                returns.append(result['expected_return'])
                risks.append(result['risk'])
                sharpe_ratio.append(result['sharpe_ratio'])

        # 绘制散点图，颜色表示夏普比率
        scatter = plt.scatter(risks, returns, c=sharpe_ratio, cmap='viridis', s=100, alpha=0.7)
        for i, method in enumerate(methods):
            plt.annotate(method, (risks[i], returns[i]), xytext=(5,5), textcoords='offset points',
                         fontsize=9)

        plt.xlabel('年化波动率 (风险)')
        plt.ylabel('年化收益率')
        plt.title('不同优化方法的风险收益分布', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='夏普比率')
        plt.tight_layout()
        plt.show()

        # 图表2：夏普比率对比柱状图
        plt.figure(figsize=(10,6))
        plt.bar(methods, sharpe_ratio, color='lightblue', alpha=0.7)
        plt.ylabel('夏普比率')
        plt.title('不同优化方法的夏普比率对比', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 图表3：权重分布对比
        plt.figure(figsize=(12,6))
        weight_data = []
        labels = []
        for key, result in results.items():
            if result is not None:
                weight_data.append(result['weights'])
                labels.append(result['method'])

        # 取前8个主要资产进行显示
        common_assets = set.intersection(*[set(weights.index) for weights in weight_data])
        common_assets = sorted(common_assets)[:8]

        if common_assets:
            weight_matrix = np.zeros((len(weight_data), len(common_assets)))
            for i, weights in enumerate(weight_data):
                for j, asset in enumerate(common_assets):
                    weight_matrix[i, j] = weights.get(asset, 0)

            x = np.arange(len(common_assets))
            width = 0.8 / len(weight_data)

            for i in range(len(weight_data)):
                offset = width * i - width * (len(weight_data) - 1) / 2
                plt.bar(x+offset, weight_matrix[i], width, label=labels[i], alpha=0.7)
            plt.xlabel('资产')
            plt.ylabel('权重')
            plt.title('主要资产权重对比(前8个资产)', fontsize=12)
            plt.xticks(x, common_assets, rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 图表4：自定义优化的惩罚项对比
        custom_results = {k: v for k, v in results.items() if 'custom' in k and v is not None}
        if custom_results:
            plt.figure(figsize=(10,6))
            downside_risks = [v.get('downside_risk', 0) for v in custom_results.values()]
            concentrations = [v.get('concentration', 0) for v in custom_results.values()]
            turnovers = [v.get('turnover', 0) for v in custom_results.values()]

            x_custom = np.arange(len(custom_results))
            width = 0.25

            plt.bar(x_custom - width, downside_risks, width, label='下行风险惩罚', alpha=0.7)
            plt.bar(x_custom, concentrations, width, label='集中度惩罚', alpha=0.7)
            plt.bar(x_custom +width, turnovers, width, label='换手率惩罚', alpha=0.7)

            plt.xlabel('自定义优化方法')
            plt.ylabel('惩罚项数值')
            plt.title('自定义优化的惩罚项对比', fontsize=12)
            plt.xticks(x_custom, [v['method'] for v in custom_results.values()], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def explain_cvxpy_concepts(self):
        """
               解释CVXPY的核心概念和优化原理
               """
        print("\n" + "=" * 80)
        print("📚 CVXPY凸优化概念解释")
        print("=" * 80)

        concepts = {
            "凸优化问题": "目标函数是凸函数，约束条件是凸集的优化问题。具有全局最优解。",
            "CVXPY变量": "cp.Variable(n) 定义优化变量，n是变量维度。",
            "目标函数": "cp.Minimize() 或 cp.Maximize() 定义优化目标。",
            "约束条件": "使用 ==, <=, >= 等运算符定义线性或非线性约束。",
            "二次型": "cp.quad_form(w, Sigma) 计算 w^T Σ w，用于方差计算。",
            "L1/L2范数": "cp.norm(x,1) 或 cp.norm(x,2) 用于惩罚项设计。",
            "问题求解": "problem.solve() 调用求解器，返回优化状态和结果。"
        }

        for concept, explanation in concepts.items():
            print(f"• {concept}: {explanation}")

def main():
    """
       主函数 - 第4天任务执行
       """
    print('=' * 70)
    print("第4天：CVXPY凸优化投资组合")
    print("学习凸优化中的约束条件和目标函数设计")
    print('=' * 70)

    # 创建CVXPY优化器实例
    optimizer = CVXPortfolioOptimizer(risk_free_rate=0.02)

    # 加载数据
    if optimizer.load_stock_data_from_current_dir():
        # 解释CVXPY概念
        optimizer.explain_cvxpy_concepts()

        # 执行不同优化方法的对比
        print(f"\n🚀 开始执行多种优化方法对比...")
        results = optimizer.compare_optimization_methods()

        # 生成可视化图表
        print(f"\n📊 正在生成优化方法对比图表...")
        optimizer.plot_optimization_comparison(results)

        print('\n' + '=' * 70)
        print("✅ 第4天任务完成！")
        print("   成功使用CVXPY实现多种投资组合优化方法")
        print("   理解了凸优化的约束条件和目标函数设计")
        print('=' * 70)

if __name__ == "__main__":
    main()


'''
📊 第4天任务总结：

🎯 核心学习成果：
1. ✅ 掌握CVXPY凸优化库的基本用法
2. ✅ 理解投资组合优化中的凸优化问题形式化
3. ✅ 实现基础均值-方差优化 (马科维茨模型)
4. ✅ 实现最大夏普比率优化 (变量替换技巧)
5. ✅ 实现自定义风险惩罚项优化

📈 凸优化关键技术：
• 变量定义: cp.Variable()
• 目标函数: cp.Minimize() / cp.Maximize()  
• 约束条件: 等式约束、不等式约束
• 二次规划: cp.quad_form() 处理方差
• 范数惩罚: L1/L2范数用于正则化

💡 自定义优化特色：
1. 下行风险惩罚 - 只惩罚负收益
2. 集中度惩罚 - 促进分散化投资  
3. 换手率惩罚 - 控制交易成本
4. 风险厌恶系数 - 灵活调整风险偏好

🔧 实践应用价值：
• 机构投资者: 根据特定需求设计自定义目标函数
• 风险管理部门: 添加各种风险约束条件
• 量化研究员: 快速原型开发各种优化策略
'''
























