'''
第12天：
绘制投资组合风险归因图，分解风险来源。
练习：实现Brinson风险贡献分析，展示各因子对总风险的贡献。
'''

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class RiskAttributionAnalyzer:
    def __init__(self, portfolio_returns, stock_returns, portfolio_weights, market_returns=None):
        """
        初始化风险归因分析器
        参数:
                portfolio_returns -- 投资组合收益率序列
                stock_returns -- 各股票收益率DataFrame (股票为列，日期为索引)
                portfolio_weights -- 投资组合权重字典 {股票代码: 权重}
                market_returns -- 市场基准收益率序列 (可选)\
        功能说明:
                - 存储投资组合、股票收益率和权重数据
                - 验证输入数据的有效性
                - 准备后续分析所需的基础数据结构
        为什么需要这些参数:
                1. portfolio_returns: 投资组合整体表现，用于计算总风险
                2. stock_returns: 各成分股表现，用于分析风险来源
                3. portfolio_weights: 各股票的投资比例，决定对总风险的影响程度
                4. market_returns: 可选的市场基准，用于计算超额收益和Beta
                """
        self.portfolio_returns = portfolio_returns
        self.stock_returns = stock_returns
        self.portfolio_weights = portfolio_weights
        self.market_returns = market_returns

        # 验证数据
        self._validate_data()
        print("🔄 风险归因分析器初始化完成")
        print(f"投资组合数据: {len(portfolio_returns)}个交易日")
        print(f"包含股票数量: {len(stock_returns.columns)}")

    def _validate_data(self):
        """
                验证输入数据的有效性
        功能说明:
                - 检查收益率数据是否为空
                - 验证权重总和是否为1（投资比例完整）
                - 如果权重总和不是1，自动进行归一化处理
        为什么要验证:
                1. 避免空数据导致计算错误
                2. 确保权重总和为1，这是投资组合分析的基本要求
                3. 自动处理用户输入的小误差，提高代码鲁棒性
        数学原理:
                权重归一化: w_i' = w_i / Σw_i
                确保所有权重加起来等于1，这样计算才有意义
                """
        # 检查收益率数据
        if len(self.portfolio_returns) == 0:
            raise ValueError("投资组合收益率数据为空")
        if len(self.stock_returns.columns) == 0:
            raise ValueError("股票收益率数据为空")
        # 检查权重总和是否为1
        total_weight = sum(self.portfolio_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ 权重总和为{total_weight:.2f}，自动归一化到1")
            for stock in self.portfolio_weights:
                self.portfolio_weights[stock] /= total_weight

    def calculate_risk_metrics(self):
        """
        计算基础风险指标
        功能说明:
                - 计算投资组合的年化收益率和波动率
                - 计算各股票的年化波动率和收益率
                - 计算股票间的相关性矩阵
        为什么要计算这些指标:
                1. 年化收益率和波动率: 评估投资组合的整体表现和风险水平
                2. 股票波动率: 了解各股票的固有风险程度
                3. 相关性矩阵: 分析股票间的联动关系，理解分散化效果
        计算公式:
                1. 年化收益率 = 日均收益率 × 252（一年交易天数）
                2. 年化波动率 = 日波动率 × √252（平方根法则）
                3. 相关性 = 协方差 / (标准差1 × 标准差2)
                """
        print("\n📊 计算基础风险指标...")
        # 投资组合年化收益率和波动率
        portfolio_annual_return = self.portfolio_returns.mean() * 252
        portfolio_annual_vol = self.portfolio_returns.std() * np.sqrt(252)

        # 各股票年化波动率
        stock_volatilities = {}
        stock_returns_annual = {}
        for stock in self.stock_returns.columns:
            stock_ret = self.stock_returns[stock]
            stock_annual_ret = stock_ret.mean() * 252
            stock_annual_vol = stock_ret.std() * np.sqrt(252)
            stock_volatilities[stock] = stock_annual_vol
            stock_returns_annual[stock] = stock_annual_ret

        # 相关性矩阵
        correlation_matrix = self.stock_returns.corr()
        risk_metrics = {
            'portfolio_annual_return': portfolio_annual_return,
            'portfolio_annual_vol': portfolio_annual_vol,
            'stock_volatilities': stock_volatilities,
            'stock_returns_annual': stock_returns_annual,
            'correlation_matrix': correlation_matrix
        }

        print(f"投资组合年化波动率: {portfolio_annual_vol:.2%}")
        print(f"投资组合年化收益率: {portfolio_annual_return:.2%}")
        return risk_metrics

    def calculate_risk_contribution(self):
        """
        计算各资产对总投资组合风险的贡献（Brinson模型）
        功能说明:
                - 计算各股票对投资组合总风险的贡献度
                - 计算边际风险贡献（权重微小变化对总风险的影响）
                - 计算相对风险贡献（各股票风险贡献的百分比）
        数学原理（Brinson模型）:
                1. 投资组合方差: σ_p² = w'Σw
                   w: 权重向量, Σ: 协方差矩阵
                2. 边际风险贡献: ∂σ_p/∂w_i = (Σw)_i / σ_p
                   表示权重微小变化时总风险的变化率
                3. 风险贡献: RC_i = w_i × (Σw)_i / σ_p
                   各股票对总风险的绝对贡献
                4. 相对风险贡献: RC_i / ΣRC_i
                   各股票风险贡献的百分比
        为什么重要:
                1. 识别哪些股票是主要风险来源
                2. 了解风险是否与投资权重成比例
                3. 为风险调整和再平衡提供依据
                """
        print("\n📈 计算风险贡献（Brinson模型）...")

        # 准备权重向量
        stocks = list(self.portfolio_weights.keys())
        weights = np.array([self.portfolio_weights[stock] for stock in stocks])
        # 获取收益率数据（对齐日期）
        aligned_returns = {}
        common_dates = self.portfolio_returns.index

        for stock in stocks:
            if stock in self.stock_returns.columns:
                stock_ret = self.stock_returns[stock].reindex(common_dates).dropna()
                aligned_returns[stock] = stock_ret
        # 确保所有数据长度一致
        min_length = min(len(ret) for ret in aligned_returns.values())
        for stock in stocks:
            aligned_returns[stock] = aligned_returns[stock].iloc[:min_length]
        # 构建收益率矩阵
        returns_matrix = pd.DataFrame(aligned_returns)
        # 计算协方差矩阵（年化）
        covariance_matrix = returns_matrix.cov() * 252
        # 计算总风险（投资组合波动率）
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        # 计算边际风险贡献
        marginal_contributions = covariance_matrix @ weights / portfolio_volatility

        # 计算绝对风险贡献
        absolute_contributions = weights * marginal_contributions
        # 计算相对风险贡献（百分比）
        total_risk_contribution = np.sum(absolute_contributions)
        relative_contribution = absolute_contributions / total_risk_contribution

        # 创建风险贡献DataFrame
        risk_contributions = pd.DataFrame({
            '股票': stocks,
            '权重': weights,
            '年化波动率': [self.stock_returns[stock].std() * np.sqrt(252) for stock in stocks],
            '绝对风险贡献': absolute_contributions,
            '边际风险贡献': marginal_contributions,
            '相对风险贡献': relative_contribution
        })
        # 按绝对风险贡献排序
        risk_contributions = risk_contributions.sort_values('绝对风险贡献', ascending=False)
        print(f"总投资组合风险: {portfolio_volatility:.2%}")
        print(f"总风险贡献: {total_risk_contribution:.2%}")
        return risk_contributions, portfolio_volatility, covariance_matrix

    def calculate_factor_risk_attribution(self, factor_returns= None):
        """
        计算因子风险归因（基于CAPM或多因子模型）
        参数:
                factor_returns -- 因子收益率DataFrame (可选，如不提供则使用简化模型)
        功能说明:
                - 分解投资组合风险为系统性风险和特异性风险
                - 系统性风险: 市场整体风险，无法通过分散化消除
                - 特异性风险: 个股特有风险，可以通过分散化降低
        为什么需要因子风险归因:
                1. 了解投资组合的风险来源是市场因素还是个股因素
                2. 评估分散化效果：特异性风险占比越高，分散化空间越大
                3. 指导投资策略：如果系统性风险过高，可能需要调整beta暴露
        简化模型原理:
                1. 系统性风险 = 平均相关性 × 平均方差 × 股票数量
                2. 特异性风险 = 总方差 - 系统性风险
                这是一种简化的估计方法，实际应用中可以使用多因子模型
                """
        print("\n🎯 计算因子风险归因...")
        if factor_returns is None:
            # 使用简化模型：市场、规模、价值、动量因子
            print("使用简化因子模型进行风险归因")
            return self._simplified_factor_attribution()
        else:
            # 使用提供的因子数据进行归因
            return self._full_factor_attribution(factor_returns)

    def _simplified_factor_attribution(self):
        """
        简化因子风险归因（基于CAPM和基本统计）
        功能说明:
                - 使用协方差矩阵的平均相关性估计系统性风险
                - 计算投资组合总方差
                - 分解为系统性风险和特异性风险
        数学原理:
                1. 平均相关性: 协方差矩阵上三角元素的平均值
                2. 平均方差: 协方差矩阵对角线元素的平均值
                3. 系统性风险 = 平均相关性 × 平均方差 × N
                   N为股票数量，这是基于等权重组合的简化估计
                4. 特异性风险 = 总方差 - 系统性风险
        局限性:
                - 假设所有股票对系统性风险的贡献相同
                - 基于平均相关性，可能不够精确
                - 适用于快速分析和理解基本风险结构
                """
        stocks = list(self.portfolio_weights.keys())
        weights = np.array([self.portfolio_weights[stock] for stock in stocks])
        # 准备数据
        aligned_returns = {}
        common_dates = self.portfolio_returns.index
        for stock in stocks:
            if stock in self.stock_returns.columns:
                stock_ret = self.stock_returns[stock].reindex(common_dates).dropna()
                aligned_returns[stock] = stock_ret

        # 确保数据对齐
        min_length = min(len(ret) for ret in aligned_returns.values())
        returns_matrix = pd.DataFrame({stock: ret.iloc[:min_length]
                                       for stock, ret in aligned_returns.items()})

        # 计算协方差矩阵
        cov_matrix = returns_matrix.cov() * 252
        # 计算总方差和分解
        portfolio_variance = weights.T @ cov_matrix @ weights

        # 使用更合理的方法：系统性风险 = 平均相关系数 * 投资组合方差
        correlation_matrix = returns_matrix.corr()
        n_stocks = len(stocks)

        # 计算平均相关系数（排除对角线）
        if n_stocks > 1:
            corr_values = correlation_matrix.values
            # 获取上三角矩阵（排除对角线）
            upper_tri_indices = np.triu_indices_from(corr_values, k=1)
            avg_correlation = corr_values[upper_tri_indices].mean()
        else:
            avg_correlation =0

        #系统性风险计算
        systematic_variance = avg_correlation * portfolio_variance
        idiosyncratic_variance = portfolio_variance - systematic_variance

        # 确保非负
        idiosyncratic_variance = max(idiosyncratic_variance, 0)
        systematic_variance = max(systematic_variance, 0)

        factor_attribution = {
            '系统性风险': systematic_variance,
            '特异性风险': idiosyncratic_variance,
            '总风险': portfolio_variance
        }

        print(f"系统性风险贡献: {systematic_variance / portfolio_variance:.1%}")
        print(f"特异性风险贡献: {idiosyncratic_variance / portfolio_variance:.1%}")
        return factor_attribution

    def calculate_diversification_benefit(self):
        """
        计算分散化效益指标
        功能说明:
                - 计算加权平均波动率（假设无分散化的风险）
                - 计算实际投资组合波动率
                - 计算分散化比率和效益
        计算公式:
                1. 加权平均波动率 = Σ(权重_i × 波动率_i)
                   假设各股票完全正相关时的风险
                2. 实际组合波动率 = √(w'Σw)
                   考虑相关性后的实际风险
                3. 分散化比率 = 加权平均波动率 / 实际组合波动率
                   比率越大，分散化效果越好
                4. 分散化效益 = 1 - (实际波动率 / 加权平均波动率)
                   风险降低的百分比
        为什么重要:
                1. 量化分散化带来的风险降低效果
                2. 评估投资组合构造的有效性
                3. 指导是否需要进行进一步分散化
                """
        print("\n🔄 计算分散化效益...")
        stocks = list(self.portfolio_weights.keys())
        # 计算加权平均波动率
        weighted_avg_vol = 0
        for stock in stocks:
            if stock in self.stock_returns.columns:
                stock_vol = self.stock_returns[stock].std() * np.sqrt(252)
                weight = self.portfolio_weights[stock]
                weighted_avg_vol += weight * stock_vol
        # 投资组合实际波动率
        portfolio_vol = self.portfolio_returns.std() * np.sqrt(252)
        # 计算分散化指标
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        diversification_benefit = 1 - (portfolio_vol/weighted_avg_vol) if weighted_avg_vol>0 else 0

        diversification_metrics = {
            '加权平均波动率': weighted_avg_vol,
            '投资组合波动率': portfolio_vol,
            '分散化比率': diversification_ratio,
            '分散化效益': diversification_benefit
        }
        print(f"分散化比率: {diversification_ratio:.2f}")
        print(f"分散化效益: {diversification_benefit:.1%}")
        return diversification_metrics

    def plot_risk_contribution_chart(self, risk_contribution, portfolio_volatility):
        """
        绘制风险贡献图 - 分成两个图表，每个图表2个子图
        参数:
               risk_contributions -- 风险贡献DataFrame
               portfolio_volatility -- 投资组合总波动率
        功能说明:
               - 图表1: 风险贡献瀑布图和分布饼图
               - 图表2: 风险收益关系散点图
        为什么这样设计图表:
               1. 瀑布图: 直观展示每只股票的风险贡献和累计效果
               2. 饼图: 显示风险来源的主要分布
               3. 散点图(权重vs风险): 识别风险与权重不成比例的股票
               4. 散点图(风险vs收益): 分析风险调整后收益
        图表解读:
               - 瀑布图中的绿色虚线: 总投资组合风险水平
               - 散点图中的红色对角线: 理想状态（权重=风险贡献）
               - 气泡大小: 表示波动率或权重大小
               """
        print("\n🎨 绘制风险贡献图...")
        stocks = risk_contribution['股票']
        risk_contrib = risk_contribution['绝对风险贡献']
        weights = risk_contribution['权重']
        rel_contrib = risk_contribution['相对风险贡献']
        # ==================== 第一个图表：风险贡献分析 ====================
        fig1, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
        fig1.suptitle('投资组合风险贡献分析', fontsize=16, fontweight='bold')
        # 1. 风险贡献瀑布图 - 左子图
        cumulative = np.cumsum(risk_contrib)
        ax1.bar(stocks, risk_contrib, alpha=0.7, label='个股风险贡献', color='skyblue')
        ax1.plot(stocks, cumulative, 'ro-', linewidth=2, markersize=6, label='累计风险贡献')
        # 添加总投资组合风险线
        ax1.axhline(y=portfolio_volatility, color='green', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'总风险 ({portfolio_volatility:.2%})')
        ax1.set_title('风险贡献瀑布图', fontweight='bold', fontsize=14)
        ax1.set_ylabel('风险贡献 (%)', fontsize=12)
        ax1.set_xticklabels(stocks, rotation=45, ha='right')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        # 添加数值标签
        for i, (stock, contrib) in enumerate(zip(stocks, risk_contrib)):
            ax1.text(i, contrib + 0.001, f'{contrib:.2%}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. 风险贡献饼图 - 右子图
        top_n = min(8, len(risk_contribution))  #  # 最多显示8只股票
        top_stocks = risk_contribution.head(top_n).copy()
        if len(risk_contribution) > top_n:
            other_contrib = risk_contribution.iloc[top_n:]['相对风险贡献'].sum()
            other_row = pd.DataFrame({
                '股票': ['其他'],
                '相对风险贡献': [other_contrib]
            })
            pie_data = pd.concat([top_stocks[['股票', '相对风险贡献']], other_row])
        else:
            pie_data = top_stocks[['股票', '相对风险贡献']]

        # 创建爆炸效果（突出最大贡献者）
        explode = [0.1 if i==0 else 0 for i in range(len(pie_data))]
        wedges, texts, autotexts = ax2.pie(
            pie_data['相对风险贡献'],
            labels=pie_data['股票'],
            autopct = '%1.1f%%',
            startangle = 90,
            explode = explode,
            shadow=True,
            colors = plt.cm.Set3(np.linspace(0,1,len(pie_data)))
        )
        # 美化百分比文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        ax2.set_title('风险贡献分布', fontweight='bold', fontsize=14)
        ax2.axis('equal')   # 确保饼图是圆形
        plt.tight_layout()
        plt.show()

        # ==================== 第二个图表：风险收益关系分析 ====================
        fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(16,6))
        fig2.suptitle('风险收益关系分析', fontsize=16, fontweight='bold')
        # 3. 权重 vs 风险贡献散点图 - 左子图
        scatter1 = ax3.scatter(weights, rel_contrib,
                               s=risk_contribution['年化波动率']*500,
                               alpha=0.6, cmap='coolwarm', edgecolors='black')

        # 添加对角线（权重=风险贡献的理想线）
        max_val = max(max(weights), max(rel_contrib)) * 1.1
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='权重=风险贡献')
        # 添加股票标签
        for i, stock in enumerate(stocks):
            ax3.annotate(stock, (weights.iloc[i], rel_contrib.iloc[i]),
                         xytext=(5,5), textcoords='offset points',
                         fontsize=8, fontweight='bold')
        ax3.set_title('权重 vs 风险贡献', fontweight='bold', fontsize=14)
        ax3.set_xlabel('投资权重', fontsize=12)
        ax3.set_ylabel('相对风险贡献', fontsize=12)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        # 添加颜色条表示波动率
        cbar1 = plt.colorbar(scatter1, ax=ax3)
        cbar1.set_label('年化波动率', fontsize=12)

        # 4. 风险贡献与收益率关系图 - 右子图
        returns = []
        for stock in stocks:
            if stock in self.stock_returns.columns:
                stock_ret = self.stock_returns[stock].mean() * 252
                returns.append(stock_ret)
            else:
                returns.append(0)
        # 创建气泡图
        scatter2 = ax4.scatter(risk_contrib, returns, s=weights*1000,
                               alpha=0.6, cmap='viridis', edgecolors='black')
        # 添加每个象限的解释
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(x=portfolio_volatility/len(stocks), color='black',
                    linestyle='-', alpha=0.3)
        # 添加股票标签
        for i, stock in enumerate(stocks):
            ax4.annotate(stock, (risk_contrib.iloc[i], returns[i]),
                         xytext=(5,5), textcoords='offset points',
                         fontsize=8, fontweight='bold')
        ax4.set_title('风险贡献 vs 收益率', fontweight='bold', fontsize=14)
        ax4.set_xlabel('风险贡献 (%)', fontsize=12)
        ax4.set_ylabel('年化收益率 (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        # 添加颜色条表示权重
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label('投资权重', fontsize=12)

        plt.tight_layout()
        plt.show()

        # 单独绘制相关性热图
        self._plot_correlation_heatmap(risk_contribution)

    def _plot_correlation_heatmap(self, risk_contribution):
        """
        绘制相关性热图（单独图表）
        功能说明:
               - 显示主要风险贡献者之间的相关性矩阵
               - 使用热图颜色表示相关性强弱
               - 在单元格中显示具体的相关系数值
        为什么重要:
               1. 相关性是决定分散化效果的关键因素
               2. 高度相关的股票会同时涨跌，降低分散化效果
               3. 负相关的股票可以提供对冲效果
               4. 帮助识别风险集中区域
        图表解读:
               - 红色: 正相关（越红相关性越强）
               - 蓝色: 负相关（越蓝负相关性越强）
               - 白色: 接近零相关
               - 数值: 具体的相关系数（-1到1之间）
               """
        fig, ax = plt.subplots(figsize=(16,6))
        # 获取相关性矩阵（只包含风险贡献高的股票）
        top_stocks = risk_contribution.head(10)['股票'].tolist()
        if len(top_stocks) > 1:
            # 获取这些股票的相关性数据
            returns_top = self.stock_returns[top_stocks]
            corr_matrix = returns_top.corr()
            # 创建热图
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            # 添加数值
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    ax.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr_matrix.iloc[i,j]) > 0.5 else 'black',
                            fontweight='bold', fontsize=9)
            # 设置坐标轴
            ax.set_xticks(range(len(corr_matrix)))
            ax.set_yticks(range(len(corr_matrix)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(corr_matrix.columns, fontsize=10)
            ax.set_title('主要风险贡献者相关性矩阵', fontweight='bold', fontsize=14)
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('相关系数', fontsize=12)
            plt.tight_layout()
            plt.show()

    def plot_diversification_analysis(self, diversification_metrics, risk_contributions):
        """
        绘制分散化分析图 - 分成两个图表，每个图表2个子图
        参数:
                diversification_metrics -- 分散化指标字典
                risk_contributions -- 风险贡献DataFrame
        功能说明:
                - 图表1: 分散化前后对比和指标展示
                - 图表2: 权重与风险贡献的详细对比
        为什么需要这些图表:
                1. 对比图: 直观展示分散化带来的风险降低效果
                2. 指标图: 量化分散化的具体数值
                3. 权重对比图: 识别哪些股票的风险贡献偏离其权重
        图表解读:
                - 蓝色柱: 投资权重
                - 红色柱: 风险贡献
                - 理想情况: 蓝柱和红柱高度相近
                - 红柱>蓝柱: 过度承担风险（考虑减仓）
                - 红柱<蓝柱: 风险利用不足（考虑加仓）
                """
        print("\n📉 绘制分散化分析图...")
        # ==================== 第一个图表：分散化效益分析 ====================
        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig1.suptitle('分散化效益分析', fontsize=16, fontweight='bold')
        # 1. 分散化效益展示 - 左子图
        categories = ['加权平均波动率', '投资组合波动率']
        values = [diversification_metrics['加权平均波动率'],
                 diversification_metrics['投资组合波动率']]

        bars = ax1.bar(categories, values, color=['lightblue', 'lightgreen'], alpha=0.8)
        ax1.set_title('分散化前后风险对比', fontweight='bold', fontsize=14)
        ax1.set_ylabel('波动率 (%)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{value:.2%}', ha='center', va='bottom', fontweight='bold')

        # 添加效益标注
        benefit = diversification_metrics['分散化效益']
        ax1.text(0.5, max(values)*0.9, f'分散化效益: {benefit:.1%}',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        # 2. 分散化指标展示 - 右子图
        indicators = ['分散化比率', '分散化效益']
        indicator_values = [diversification_metrics['分散化比率'],
                          diversification_metrics['分散化效益']]
        bars2 = ax2.bar(indicators, indicator_values, color=['orange', 'purple'], alpha=0.8)
        ax2.set_title('分散化指标', fontweight='bold', fontsize=14)
        ax2.set_ylabel('指标值', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for i, (bar, value) in enumerate(zip(bars2, indicator_values)):
            height = bar.get_height()
            format_str = f'{value:.2f}' if indicators[i] == '分散化比率' else f'{value:.1%}'
            ax2.text(bar.get_x() + bar.get_width()/2., height+(0.01 if value >= 0 else -0.02),
                     format_str, ha='center', va='bottom' if value >= 0 else 'top',
                     fontweight='bold')
        plt.tight_layout()
        plt.show()

        # ==================== 第二个图表：权重与风险贡献对比 ====================
        fig2, ax3 = plt.subplots(1,1, figsize=(14,6))
        fig2.suptitle('权重与风险贡献对比分析', fontsize=16, fontweight='bold')
        stocks = risk_contributions['股票']
        weights = risk_contributions['权重']
        risk_share = risk_contributions['相对风险贡献']
        x = np.arange(len(stocks))
        width = 0.35
        bars1 = ax3.bar(x-width/2, weights, width, label='投资权重', alpha=0.7, color='lightblue')
        bars2 = ax3.bar(x+width/2, risk_share, width, label='风险贡献', alpha=0.7, color='lightcoral')
        ax3.set_title('权重 vs 风险贡献对比', fontweight='bold', fontsize=14)
        ax3.set_xlabel('股票', fontsize=12)
        ax3.set_ylabel('百分比', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(stocks, rotation=45, ha='right', fontsize=10)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:   # 只显示较大的值
                    ax3.text(bar.get_x()+bar.get_width()/2., height+0.01,
                             f'{height:.1%}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()

    def generate_risk_report(self, risk_metrics, risk_contributions, portfolio_volatility,
                             factor_attribution, diversification_metrics):
        """
        生成详细的风险归因分析报告
        功能说明:
                - 汇总所有风险指标和计算结果
                - 提供专业的投资建议
                - 识别高风险和低风险股票
                - 给出具体的后续监控建议
        报告结构:
                1. 基础风险指标: 整体表现评估
                2. 风险贡献分析: 各股票的风险影响
                3. 因子风险归因: 系统性vs特异性风险
                4. 分散化效益: 风险降低效果
                5. 投资建议: 具体的调整建议
                6. 后续监控: 持续的监控计划
        为什么重要:
                1. 将复杂数据转化为可操作的见解
                2. 提供量化的投资决策依据
                3. 帮助持续优化投资组合
                4. 建立风险管理的系统方法
                """
        print("\n" + "=" * 80)
        print("📊 投资组合风险归因分析报告")
        print("=" * 80)

        # 基础风险信息
        print(f"\n📈 基础风险指标:")
        print(f"   投资组合年化波动率: {portfolio_volatility:.2%}")
        print(f"   投资组合年化收益率: {risk_metrics['portfolio_annual_return']:.2%}")
        print(f"   风险调整收益（夏普比）: {risk_metrics['portfolio_annual_return'] / portfolio_volatility:.2f}")
        # 风险贡献分析
        print(f"\n🎯 风险贡献分析（Brinson模型）:")
        print(f"   总风险贡献: {risk_contributions['绝对风险贡献'].sum():.2%}")
        print(f"   前3大风险贡献者:")

        top_3 = risk_contributions.head(3)
        for _, row in top_3.iterrows():
            print(f"{row['股票']}: 权重={row['权重']:.1%}, "
                  f"风险贡献={row['相对风险贡献']:.1%}, "
                  f"边际风险={row['边际风险贡献']:.3f}")

        # 风险集中度
        herfindahl_index = (risk_contributions['相对风险贡献']**2).sum()
        print(f"   风险集中度指数: {herfindahl_index:.3f} "
              f"{'(较高)' if herfindahl_index > 0.25 else '(适中)' if herfindahl_index > 0.15 else '(较低)'}")

        # 因子风险归因
        print(f"\n🔍 因子风险归因:")
        if '系统性风险' in factor_attribution:
            sys_risk_share = factor_attribution['系统性风险'] / factor_attribution['总风险']
            idio_risk_share = factor_attribution['特异性风险'] / factor_attribution['总风险']
            print(f"   系统性风险: {sys_risk_share:.1%}")
            print(f"   特异性风险: {idio_risk_share:.1%}")

        # 分散化分析
        print(f"\n🔄 分散化效益分析:")
        print(f"   加权平均波动率: {diversification_metrics['加权平均波动率']:.2%}")
        print(f"   实际组合波动率: {diversification_metrics['投资组合波动率']:.2%}")
        print(f"   分散化比率: {diversification_metrics['分散化比率']:.2f}")
        print(f"   分散化效益: {diversification_metrics['分散化效益']:.1%}")

        # 风险归因总结
        print(f"\n💡 风险归因总结:")

        # 识别高风险股票
        high_risk_stocks = risk_contributions[risk_contributions['相对风险贡献'] >
                                             risk_contributions['权重'] * 1.5]

        if len(high_risk_stocks) > 0:
            print(f"   高风险股票（风险贡献显著高于权重）:")
            for _, row in high_risk_stocks.iterrows():
                risk_multiple = row['相对风险贡献'] / row['权重']
                print(f"     {row['股票']}: 权重={row['权重']:.1%}, "
                      f"风险贡献={row['相对风险贡献']:.1%}, 风险倍数={risk_multiple:.1f}x")

        # 识别低风险股票
        low_risk_stocks = risk_contributions[risk_contributions['相对风险贡献'] <
                                                 risk_contributions['权重'] * 0.7]

        if len(low_risk_stocks) > 0:
            print(f"   低风险股票（风险贡献显著低于权重）:")
            for _, row in low_risk_stocks.iterrows():
                risk_multiple = row['相对风险贡献'] / row['权重']
                print(f"     {row['股票']}: 权重={row['权重']:.1%}, "
                      f"风险贡献={row['相对风险贡献']:.1%}, 风险倍数={risk_multiple:.1f}x")

        # 投资建议
        print(f"\n🎯 投资建议:")
        # 基于风险贡献的建议
        max_contrib_stock = risk_contributions.iloc[0]
        if max_contrib_stock['相对风险贡献'] > 0.3:
            print(f"   • {max_contrib_stock['股票']}贡献了{max_contrib_stock['相对风险贡献']:.1%}的风险，"
                  f"考虑降低其权重以分散风险")

        # 基于分散化的建议
        if diversification_metrics['分散化比率'] < 1.1:
            print(f"   • 分散化效益有限（比率{diversification_metrics['分散化比率']:.2f}），"
                  f"考虑增加低相关性资产")
        elif diversification_metrics['分散化比率'] > 1.5:
            print(f"   • 分散化效果显著（比率{diversification_metrics['分散化比率']:.2f}），"
                  f"当前配置合理")

        # 基于风险集中度的建议
        if herfindahl_index > 0.25:
            print(f"   • 风险集中度较高（指数{herfindahl_index:.3f}），"
                      f"建议进一步分散投资")
        print(f"\n📋 后续监控建议:")
        print(f"   1. 定期（每月）重新计算风险贡献")
        print(f"   2. 监控高风险股票的表现")
        print(f"   3. 关注相关性矩阵的变化")
        print(f"   4. 根据市场环境调整风险预算")

        print("=" * 80)

def load_real_stock_data(stock_list, start_date='2019-01-01', end_date='2025-12-02'):
    """
    从Excel文件加载真实的股票数据
    参数:
        stock_list -- 股票代码列表
        start_date -- 开始日期（默认为2019年，使用最近数据）
        end_date -- 结束日期（默认为2025年12月2日）
    功能说明:
        - 逐个加载每个股票的Excel数据文件
        - 自动识别日期列和价格列
        - 计算日收益率
        - 对齐所有股票的数据日期
        - 返回清理后的收益率DataFrame
    数据处理流程:
        1. 读取Excel文件
        2. 识别日期列（支持多种格式）
        3. 设置日期索引
        4. 识别价格列（支持多种格式）
        5. 计算收益率
        6. 数据质量检查
    为什么选择2019-2025年数据:
        1. 最近数据更能反映当前市场特征
        2. 避免过时的市场结构影响分析
        3. 足够的数据量进行可靠分析（6-7年）
        """
    print("📊 从Excel文件加载真实股票数据...")
    print(f"数据时间范围: {start_date} 到 {end_date}")

    all_returns = {}
    all_prices = {}
    loaded_stocks = []
    for stock in stock_list:
        try:
            file_path = f"./{stock}_stock_data.xlsx"
            if not os.path.exists(file_path):
                print(f"⚠️  文件不存在: {file_path}")
                continue
            df = pd.read_excel(file_path)
            print(f"\n📈 处理 {stock} 数据:")
            print(f"  文件列名: {list(df.columns)}")
            print(f"  数据行数: {len(df)}")

            # 检查数据时间范围
            # 尝试不同的日期列名
            date_columns = ['date', 'Date', 'DATE', 'datetime', 'Datetime', '日期', '时间']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    print(f"  找到日期列: {date_col}")
                    break

            if date_col is None:
                # 尝试第一列是否是日期类型
                first_col = df.columns[0]
                print(f"  尝试第一列作为日期: {first_col}")

                # 尝试多种日期格式
                try:
                    df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
                    if df[first_col].isnull().all():
                        raise ValueError("无法转换为日期")
                    date_col = first_col
                    print(f"  成功转换第一列为日期")
                except:
                    # 尝试其他可能的日期列
                    for col in df.columns:
                        if 'date' in str(col).lower() or 'time' in str(col).lower():
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                date_col = col
                                print(f"  找到日期列: {col}")
                                break
                            except:
                                continue
            if date_col is None:
                raise ValueError(f"未找到日期列，可用列: {list(df.columns)}")
            # 设置日期索引
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)

            df = df.sort_index()
            # 显示数据时间范围
            if len(df) > 0:
                print(f"  原始数据时间范围: {df.index[0].date()} 到 {df.index[-1].date()}")
            # 尝试不同的价格列名
            price_columns = ['close', 'Close', 'Adj Close', 'Price', 'price',
                             'Close_Price', 'close_price', 'Adj Close_Price',
                             '收盘价', '收盘', 'ClosePrice']
            price_col = None
            for col in price_columns:
                if col in df.columns:
                    price_col = col
                    print(f"  找到价格列: {price_col}")
                    break

            if price_col is None:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'price' in col_lower or 'close' in col_lower or 'adj' in col_lower:
                        price_col = col
                        print(f"  使用可能的价格列: {price_col}")
                        break
            if price_col is None and len(df.columns) >=2:
                # 如果还是没找到，尝试数值列
                for col in df.columns:
                    if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                        price_col = col
                        print(f"  使用数值列作为价格: {price_col}")
                        break
            if price_col is None:
                raise ValueError(f"未找到价格列，可用列: {list(df.columns)}")

            # 获取价格序列
            prices = df[price_col]

            # 检查价格数据
            print(f"  价格数据统计:")
            print(f"    非空值数量: {prices.count()}")
            print(f"    缺失值数量: {prices.isnull().sum()}")
            print(f"    价格范围: {prices.min():.2f} - {prices.max():.2f}")

            prices = prices.dropna()
            if len(prices) == 0:
                print(f"  ⚠️  {stock}: 价格数据为空，跳过")
                continue

            # 显示数据截止日期
            latest_date = prices.index[-1]
            print(f"  最新数据日期: {latest_date.date()}")

            # 确保有足够的历史数据
            min_required_date = pd.Timestamp('2019-01-01')
            if prices.index[0] > min_required_date:
                print(f"  ⚠️  {stock}: 历史数据不足，最早数据从 {prices.index[0].date()} 开始")

            # 使用完整数据范围（不进行额外过滤）
            print(f"  使用数据范围: {prices.index[0].date()} 到 {prices.index[-1].date()}")

            # 计算日收益率
            returns = prices.pct_change().dropna()

            # 检查收益率数据有效性
            print(f"  收益率数据统计:")
            print(f"    有效收益率数量: {len(returns)}")
            print(f"    平均日收益率: {returns.mean():.4%}")
            print(f"    日收益率波动率: {returns.std():.4%}")

            if len(returns) < 100:
                print(f"  ⚠️  {stock}: 有效收益率数据不足 ({len(returns)}天)，跳过")
                continue

            all_returns[stock] = returns
            all_prices[stock] = prices
            loaded_stocks.append(stock)

            print(f"  ✅ {stock}: 成功加载{len(prices)}天价格数据，{len(returns)}天收益率数据")
            print(f"     时间范围: {prices.index[0].date()} 到 {prices.index[-1].date()}")
        except Exception as e:
            print(f"  ❌ {stock}: 加载失败 - {str(e)[:100]}...")
            continue

    if not all_returns:
        raise ValueError("没有成功加载任何股票数据")

    # 创建收益率DataFrame
    returns_df = pd.DataFrame(all_returns)
    print(f"\n✅ 成功加载 {len(all_returns)} 只股票数据: {', '.join(loaded_stocks)}")
    print(f"   最终数据时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    print(f"   交易日数: {len(returns_df)}")

    # 显示各股票数据量
    print(f"\n📊 各股票数据量统计:")
    for stock in loaded_stocks:
        if stock in returns_df.columns:
            data_count = returns_df[stock].count()
            if data_count > 0:
                date_range = f"{returns_df[stock].dropna().index[0].date()} 到{returns_df[stock].dropna().index[-1].date()}"
                print(f"  {stock}: {data_count}个交易日，{date_range}")
    return returns_df, all_prices, loaded_stocks

def create_portfolio_from_real_data(stock_list, investment_amounts):
    """
    从真实数据创建投资组合

    参数:
    stock_list -- 股票代码列表
    investment_amounts -- 投资金额字典 {股票代码: 投资金额}
    """
    print("\n💰 创建投资组合...")

    # 设置日期范围 - 使用2019-2025年数据
    start_date = '2019-01-01'
    end_date = '2025-12-02'
    print(f"📅 数据范围: {start_date} 到 {end_date}")
    print("说明: 使用2019年以来的数据，以获得更准确的市场特征")

    # 加载股票数据
    stock_returns_df, stock_prices, loaded_stocks = load_real_stock_data(
        stock_list, start_date=start_date, end_date=end_date
    )

    # 只保留成功加载的股票
    available_stocks = [s for s in stock_list if s in loaded_stocks]

    if not available_stocks:
        raise ValueError("没有可用的股票数据")

    # 重新计算投资金额（只包括成功加载的股票）
    available_investments = {s: investment_amounts[s] for s in available_stocks if s in investment_amounts}
    total_investment = sum(available_investments.values())

    # 计算投资组合权重
    portfolio_weights = {}

    print("\n📋 投资组合权重计算:")
    for stock, amount in available_investments.items():
        if stock in stock_returns_df.columns:
            weight = amount / total_investment
            portfolio_weights[stock] = weight
            print(f"  {stock}: 投资${amount:,} → 权重{weight:.2%}")

    print(f"\n📊 投资组合概况:")
    print(f"   总投资: ${total_investment:,}")
    print(f"   包含股票: {len(portfolio_weights)}只")

    # 对齐所有股票的数据日期
    print("\n🔄 对齐股票数据日期...")

    # 找到所有股票共同的交易日期
    common_dates = None
    for stock in portfolio_weights.keys():
        if stock in stock_returns_df.columns:
            stock_dates = stock_returns_df[stock].dropna().index
            if common_dates is None:
                common_dates = stock_dates
            else:
                common_dates = common_dates.intersection(stock_dates)

    if common_dates is None or len(common_dates) < 100:
        print(f"⚠️  共同交易日数量: {len(common_dates) if common_dates else 0}")
        if common_dates and len(common_dates) < 100:
            print("警告: 共同交易日较少，可能影响分析准确性")
        print("尝试使用非对齐数据...")
        common_dates = stock_returns_df.index

    print(f"   共同交易日: {len(common_dates)}天")
    print(f"   时间范围: {common_dates[0].date()} 到 {common_dates[-1].date()}")

    # 计算投资组合收益率
    portfolio_returns = pd.Series(0, index=common_dates)

    for stock, weight in portfolio_weights.items():
        if stock in stock_returns_df.columns:
            # 获取对齐的收益率数据
            stock_returns_aligned = stock_returns_df[stock].reindex(common_dates).fillna(0)
            portfolio_returns += stock_returns_aligned * weight

    # 计算基本统计
    portfolio_daily_return = portfolio_returns.mean()
    portfolio_daily_vol = portfolio_returns.std()

    print(f"\n📈 投资组合绩效统计:")
    print(f"   数据期间: {common_dates[0].date()} 到 {common_dates[-1].date()}")
    print(f"   交易日数: {len(portfolio_returns)}")
    print(f"   日收益率均值: {portfolio_daily_return * 100:.4f}%")
    print(f"   日收益率波动率: {portfolio_daily_vol * 100:.4f}%")
    print(f"   年化收益率: {portfolio_daily_return * 252 * 100:.2f}%")
    print(f"   年化波动率: {portfolio_daily_vol * np.sqrt(252) * 100:.2f}%")

    if portfolio_daily_vol > 0:
        sharpe_ratio = portfolio_daily_return / portfolio_daily_vol * np.sqrt(252)
        print(f"   夏普比率: {sharpe_ratio:.2f}")

    return portfolio_returns, stock_returns_df, portfolio_weights

# ==================== 主函数 ====================
def main():
    """
        主函数：使用真实股票数据执行完整的风险归因分析

        功能说明:
        - 定义投资组合配置
        - 检查数据文件存在性
        - 执行完整的风险归因分析流程
        - 生成可视化图表和报告
        - 保存分析结果到Excel文件

        分析流程:
        1. 准备投资组合数据
        2. 创建投资组合并计算收益率
        3. 进行风险贡献分析
        4. 进行因子风险归因
        5. 计算分散化效益
        6. 生成图表和报告
        7. 保存结果

        为什么需要主函数:
        1. 组织整个分析流程
        2. 处理异常和错误
        3. 提供用户友好的输出
        4. 确保分析的完整性和一致性
        """
    print("🏦 投资组合风险归因分析系统（真实数据版）")
    print("版本: 1.0")
    print("功能: Brinson风险贡献分析 + 风险来源分解")
    print("=" * 60)

    try:
        # ==================== 1. 使用你的真实投资组合 ====================
        # 你的真实投资组合数据
        investment_amounts = {
            'KO': 157,
            'VOO': 155,
            'SCHD': 154,
            'LLY': 137,
            'GLD': 105,
            'AAPL': 65,
            'NBIS': 47,
            'AA': 46,
            'UNH': 40,
            'SBUX': 39,
            'GOOG': 32,
            'LCID': 31,
            'META': 23,
            'UPST': 22
        }
        # 股票列表
        stock_list = list(investment_amounts.keys())
        print("\n🔍 检查股票数据文件...")
        print(f"你的投资组合包含 {len(stock_list)} 只股票")
        print(f"总投资金额: ${sum(investment_amounts.values()):,}")

        # 检查文件是否存在
        missing_files = []
        for stock in stock_list:
            file_path = f"./{stock}_stock_data.xlsx"
            if os.path.exists(file_path):
                print(f"  ✅ {stock}: 文件存在 (${investment_amounts[stock]:,})")
            else:
                print(f"  ❌ {stock}: 文件不存在 - {file_path}")
                missing_files.append(stock)

        if missing_files:
            print(f"\n⚠️  缺失 {len(missing_files)} 个股票数据文件:")
            for stock in missing_files:
                print(f"    - {stock}_stock_data.xlsx")
            print("\n请确保所有股票数据文件在当前目录下")
            print("文件格式要求: Excel文件，包含日期和价格列")

        # ==================== 2. 从真实数据创建投资组合 ====================
        print("\n📅 分析设置: 使用最近数据（2019-2025年）")
        print("理由: 市场特征随时间变化，最近数据更能反映当前市场状况")
        portfolio_returns, stock_returns_df, portfolio_weights = create_portfolio_from_real_data(
            stock_list, investment_amounts
        )

        # ==================== 3. 创建风险归因分析器 ====================
        print("\n🔄 创建风险归因分析器...")
        analyzer = RiskAttributionAnalyzer(
            portfolio_returns = portfolio_returns,
            stock_returns= stock_returns_df,
            portfolio_weights= portfolio_weights
        )

        # ==================== 4. 计算风险指标 ====================
        risk_metrics = analyzer.calculate_risk_metrics()

        # ==================== 5. 计算风险贡献 ====================
        risk_contributions, portfolio_volatility, cov_matrix = analyzer.calculate_risk_contribution()

        # ==================== 6. 计算因子风险归因 ====================
        factor_attribution = analyzer.calculate_factor_risk_attribution()

        # ==================== 7. 计算分散化效益 ====================
        diversification_metrics = analyzer.calculate_diversification_benefit()

        # ==================== 8. 绘制风险贡献图 ====================
        print("\n📈 生成可视化图表...")
        analyzer.plot_risk_contribution_chart(risk_contributions, portfolio_volatility)

        # ==================== 9. 绘制分散化分析图 ====================
        analyzer.plot_diversification_analysis(diversification_metrics, risk_contributions)

        # ==================== 10. 生成详细报告 ====================
        analyzer.generate_risk_report(risk_metrics, risk_contributions, portfolio_volatility,
                                      factor_attribution, diversification_metrics)

        # ==================== 11. 输出风险贡献表格 ====================
        """
        风险贡献表格说明:
            - 股票: 股票代码
            - 权重: 投资比例
            - 年化波动率: 股票自身的风险水平
            - 风险贡献: 对组合总风险的贡献百分比
            - 边际风险: 权重变化对总风险的敏感度
            - 风险倍数: 风险贡献 / 权重，衡量风险效率
        风险倍数解读:
            - >1.5: 过度承担风险，可能需要减仓
            - 1.0-1.5: 风险与权重基本匹配
            - <0.7: 风险利用不足，可能可以加仓
        """
        print("\n📋 详细风险贡献表格:")
        print("-" * 90)
        print(f"{'股票':<8} {'权重':<8} {'年化波动率':<12} {'风险贡献':<10} {'边际风险':<10} {'风险倍数':<10}")
        print("-" * 90)

        for _, row in risk_contributions.iterrows():
            risk_multiple = row['相对风险贡献'] / row['权重']
            print(f"{row['股票']:<8} {row['权重']:<8.1%} {row['年化波动率']:<12.2%} "
                  f"{row['相对风险贡献']:<10.1%} {row['边际风险贡献']:<10.3f} {risk_multiple:<10.2f}")
        print("-" * 90)

        # ========================投资建议详细分析==========================
        print("\n🎯 详细投资建议分析:")
        print("=" * 60)

        # 分析每只股票的风险收益特征
        """
        风险收益特征表说明:
        - 股票: 股票代码
        - 权重: 投资比例
        - 收益率: 年化收益率
        - 波动率: 年化波动率
        - 风险贡献: 对组合总风险的贡献
        - 建议: 基于风险倍数的具体建议
    
        建议生成逻辑:
        1. 风险倍数 > 2.0: 风险过高，强烈建议减仓
        2. 风险倍数 > 1.5: 风险偏高，建议减仓
        3. 风险倍数 > 1.2: 风险稍高，监控
        4. 风险倍数 < 0.5: 风险利用严重不足，建议加仓
        5. 风险倍数 < 0.7: 风险利用不足，可考虑加仓
        6. 风险收益比 > 1.0: 风险调整后收益优秀，保持
        """
        print("\n📊 股票风险收益特征:")
        print(f"{'股票':<8} {'权重':<8} {'收益率':<10} {'波动率':<10} {'风险贡献':<10} {'建议':<20}")
        print("-" * 60)

        for _, row in risk_contributions.iterrows():
            stock = row['股票']
            weight = row['权重']
            ret = risk_metrics['stock_returns_annual'].get(stock, 0)
            vol = risk_metrics['stock_volatilities'].get(stock, 0)
            risk_contrib = row['相对风险贡献']
            risk_multiple = risk_contrib / weight

            # 生成建议
            if risk_multiple > 2.0:
                suggestion = "⚠️ 风险过高，强烈建议减仓"
            elif risk_multiple > 1.5:
                suggestion = "⚠️ 风险偏高，建议减仓"
            elif risk_multiple > 1.2:
                suggestion = "风险稍高，监控"
            elif risk_multiple < 0.7:
                suggestion = "✅ 风险利用不足，可考虑加仓"
            elif risk_multiple < 0.5:
                suggestion = "✅ 风险利用严重不足，建议加仓"
            elif ret / vol > 1.0:
                suggestion = "✅ 风险收益比优秀，保持"
            elif ret > 0:
                suggestion = "收益为正，观察"
            else:
                suggestion = "观察"

            print(f"{stock:<8} {weight:<8.1%} {ret:<10.2%} {vol:<10.2%} {risk_contrib:<10.1%} {suggestion:<20}")

        print("="*60)


        #=====================分析完成总结==================
        print("\n🎉 风险归因分析完成！")
        print("=" * 60)
        print(f"\n📅 分析数据范围: {portfolio_returns.index[0].date()} 到 {portfolio_returns.index[-1].date()}")
        print(f"📊 包含股票: {len(portfolio_weights)}只")
        print(f"💰 总投资: ${sum(investment_amounts.values()):,}")

        print("\n📋 分析成果总结:")
        print("   ✅ 使用你的真实投资组合数据")
        print("   ✅ 数据时间范围到2025年12月2日")
        print("   ✅ Brinson风险贡献计算")
        print("   ✅ 因子风险归因分析")
        print("   ✅ 分散化效益评估")
        print("   ✅ 专业可视化图表")
        print("   ✅ 详细投资建议")

        print("\n💡 核心洞察:")
        max_contrib = risk_contributions.iloc[0]
        print(f"   1. 最大风险贡献者: {max_contrib['股票']} (权重{max_contrib['权重']:.1%}, "
              f"风险贡献{max_contrib['相对风险贡献']:.1%})")
        print(f"   2. 分散化效益: {diversification_metrics['分散化效益']:.1%}")
        print(f"   3. 风险集中度: {(risk_contributions['相对风险贡献'] ** 2).sum():.3f}")
        print(f"   4. 投资组合夏普比率: {risk_metrics['portfolio_annual_return'] / portfolio_volatility:.2f}")

        print("\n🔍 立即行动建议:")

        # 检查是否有需要立即调整的股票
        urgent_adjustments = []
        for _, row in risk_contributions.iterrows():
            risk_multiple = row['相对风险贡献'] / row['权重']
            if risk_multiple > 2.0:
                urgent_adjustments.append(f"  • {row['股票']}: 风险倍数{risk_multiple:.1f}倍，建议立即减仓")
            elif risk_multiple < 0.5:
                urgent_adjustments.append(f"  • {row['股票']}: 风险倍数{risk_multiple:.1f}倍，风险利用不足，建议加仓")

        if urgent_adjustments:
            print("   以下股票建议立即调整:")
            for adjustment in urgent_adjustments:
                print(adjustment)
        else:
            print("   暂无需要立即调整的股票，组合相对平衡")

        print("\n📋 后续监控:")
        print("   1. 每月重新计算风险贡献")
        print("   2. 关注高风险倍数股票的表现")
        print("   3. 定期检查相关性变化")
        print("   4. 根据市场环境调整风险预算")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        print("\n🔧 可能的原因:")
        print("   1. 股票数据文件不存在")
        print("   2. 文件格式不正确")
        print("   3. 数据量不足")
        print("   4. 权重配置问题")
        import traceback
        traceback.print_exc()

# ==================== 程序入口点 ====================
if __name__ == "__main__":
    """
        程序入口点说明:
        - 当直接运行此文件时执行main函数
        - 如果被其他文件导入则不执行
        - 这是Python的标准做法，确保代码的模块化和可重用性

        使用方法:
        1. 确保所有股票数据文件在当前目录下
        2. 直接运行此Python文件
        3. 查看输出结果和图表
        4. 根据分析结果调整投资策略
        """
    main()
    print("\n🙏 感谢使用风险归因分析系统！")
    print("风险管理的核心是理解而非规避风险。")

'''
============================总结===========================
一、风险归因图（Risk Attribution Chart）
一句话：用图表展示每只股票对总风险的"贡献份额"。

核心：
    瀑布图：看每只股票的累计风险贡献
    饼图：看风险在各股票间的分布比例
    气泡图：比较权重与风险贡献的关系

关键看什么：
    风险贡献 > 权重 → 高风险股票（考虑减仓）
    风险贡献 < 权重 → 低风险股票（考虑加仓）
    绿线：总投资组合风险水平

二、分解风险来源
一句话：分清风险是市场影响的还是公司自身的问题。

两种风险：
    1. 系统性风险（市场风险）：影响所有股票，无法消除
            来源：经济周期、利率、通胀、政策
            占比高 → 跟大盘走，分散化效果有限
    2. 特异性风险（个股风险）：只影响个别公司，可分散
            来源：公司管理、产品、竞争对手
            占比高 → 可通过分散投资降低风险

三、Brinson风险贡献分析
一句话：精确计算每只股票对组合风险的"责任大小"。

三个核心指标：
    边际风险贡献：权重变化1%，总风险变化多少
    绝对风险贡献：这只股票实际贡献了多少风险值
    相对风险贡献：这只股票占总风险的百分比

怎么用：
    风险倍数 = 相对风险贡献 ÷ 权重
    倍数 > 1.5 → 过度承担风险（减仓）
    倍数 < 0.7 → 风险利用不足（加仓）
    倍数 ≈ 1.0 → 风险与权重匹配（保持）

实际意义：
    找出发财时谁贡献多，亏钱时谁拖后腿
    知道调整哪只股票最有效
    避免"权重小但风险大"的隐形炸弹

四、三者的关系
    Brinson计算 → 得出具体数字
    风险分解 → 理解数字背后的原因
    风险归因图 → 把数字变成直观图表

就像健康报告：
    Brinson分析：化验单上的具体数值
    风险分解：判断是遗传病还是生活习惯病
    风险归因图：直观的体检图表展示
'''


