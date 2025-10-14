'''
Day 6：组合与风险管理基础
目标：
- 构建债券+期权组合
- 学习基本风险指标（VaR、CVaR）
任务：
- 使用真实数据计算组合净值
- 蒙特卡洛模拟计算组合VaR
输出：完整的组合风险分析脚本
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


class InteractivePortfolioRiskAnalyzer:
    '''
    交互式组合风险分析器
    功能：基于真实数据计算债券+期权组合的风险指标(VaR, CVaR)
    '''

    def __init__(self):
        '''初始化分析器'''
        pass

    def load_data(self):
        '''
        加载真实市场数据
        从三个文件读取：国债收益率、债券信息、AAPL期权数据
        '''
        print("加载真实数据中.....")
        try:
            # 加载国债收益率数据 - 用于无风险利率参考
            treasury = pd.read_excel('US_Treasury_Yields.xlsx')
            latest_yield = treasury.iloc[-1]['DGS10'] / 100  # 获取最新10年期收益率

            # 加载债券数据
            securities = pd.read_csv('./Securities.csv')

            # 加载AAPL期权数据，分别读取看涨和看跌期权
            calls = pd.read_excel('AAPL_options.xlsx', sheet_name='Calls')
            puts = pd.read_excel('AAPL_options.xlsx', sheet_name='Puts')
            # 合并看涨和看跌期权数据
            options = pd.concat([calls, puts], ignore_index=True)

            # 打印数据加载信息
            print(f"10年期国债收益率: {latest_yield * 100:.2f}%")
            print(f"证券数据: {len(securities)} 条记录")
            print(f"期权数量: {len(options)} 个合约")

            return treasury, securities, options, latest_yield

        except Exception as e:
            print(f"数据加载失败: {e}")
            return None, None, None, 0.04   # 返回4%的默认无风险利率

    def create_bond_portfolio(self, securities):
        '''
        创建债券投资组合
        让用户交互式选择债券和投资金额
        '''
        bonds = []  # 存储债券信息的列表

        if securities is not None and not securities.empty:
            # 过滤出有价格数据的债券
            valid_bonds = securities[securities['Price per $100'].notna()]
            print(f"找到 {len(valid_bonds)} 个有价格数据的债券")

            # 让用户选择要投资几个债券
            print(f"\n请设置债券投资:")
            try:
                # 获取用户输入的债券数量，默认3个
                num_bonds = int(input("要投资几个债券? (建议1-5个): ") or "3")
                num_bonds = max(1, min(10, num_bonds))  # 限制在1-10个之间
            except:
                num_bonds = 3
                print(f"使用默认值: 3个债券")

            # 遍历用户指定数量的债券
            for i, bond in valid_bonds.head(num_bonds).iterrows():
                price = bond['Price per $100']  # 债券价格(每100面值)

                # 让用户输入每个债券的投资金额
                try:
                    notional = float(input(
                        f"债券 {i + 1}: {bond['Security Type']} {bond.get('Security Term', '')} "
                        f"- 投资金额($)? (默认1000): ") or "1000")
                except:
                    notional = 1000
                    print(f"使用默认值: $1000")

                # 计算债券实际价值: 价格 × 面值 / 100
                value = price * notional / 100

                # 根据债券类型估计波动率
                bond_type = bond['Security Type']
                if "Bill" in bond_type:     # 短期票据波动率较低
                    vol = 0.05
                elif "Note" in bond_type:   # 中期票据波动率中等
                    vol = 0.08
                else:                       # 其他类型债券
                    vol = 0.07

                # 将债券信息添加到列表
                bonds.append({
                    'name': f"{bond['Security Type']} {bond.get('Security Term', '')}",
                    'value': value,             # 债券价值
                    'price': price,             # 债券价格
                    'notional': notional,       # 投资面值
                    'vol': vol                  # 估计波动率
                })

        # 处理无数据情况
        if len(bonds) == 0:
            print(f'没有找到有效的债券数据')
        else:
            # 打印债券投资详情
            print(f"\n债券投资详情:")
            for bond in bonds:
                print(f"✓ {bond['name']}: 价格${bond['price']:.2f}, "
                      f"投资${bond['notional']:,.0f}, 价值${bond['value']:,.0f}")

        return bonds

    def create_option_portfolio(self, options):
        """
        创建期权投资组合
        让用户交互式选择期权和购买数量
        """
        option_portfolio = []   # 存储期权信息的列表

        if options is not None and not options.empty:
            # 过滤有效的期权数据（有价格、行权价、隐含波动率）
            valid_options = options.dropna(subset=['lastPrice', 'strike', 'impliedVolatility'])
            # 过滤掉异常高的隐含波动率（>100%）
            valid_options = valid_options[valid_options['impliedVolatility'] <= 1.0]

            print(f"\n找到 {len(valid_options)} 个有效的期权数据")

            # 让用户选择期权数量
            try:
                num_options = int(input("\n要投资几个期权? (建议1-3个): ") or '2')
                num_options = max(1, min(5, num_options))   # 限制在1-5个
            except:
                num_options = 2
                print(f"使用默认值: 2个期权")

            selected_options = []       # 存储选中的期权数据

            # 分离看涨和看跌期权
            calls = valid_options[valid_options['optionType'].str.contains('call', case=False, na=False)]
            puts = valid_options[valid_options['optionType'].str.contains('put', case=False, na=False)]

            # 选择实值看涨期权（行权价低于当前股价）
            if not calls.empty:
                itm_call = calls[calls['strike'] < 170].head(1)   # 假设股价$180
                if not itm_call.empty:
                    selected_options.append(itm_call.iloc[0])

            # 选择虚值看跌期权（行权价低于当前股价）
            if not puts.empty:
                otm_put = puts[puts['strike'] < 170].head(1)
                if not otm_put.empty:
                    selected_options.append(otm_put.iloc[0])

            # 如果选择的期权数量不够，补充一些
            if len(selected_options) < num_options and not valid_options.empty:
                additional = valid_options.head(num_options - len(selected_options))
                selected_options.extend(additional.to_dict('records'))

            # 处理用户选择的期权
            for i, opt in enumerate(selected_options[:num_options]):
                # 判断期权类型：看涨或看跌
                opt_type = '看涨' if 'call' in str(opt['optionType']).lower() else '看跌'

                # 让用户输入购买数量
                try:
                    quantity = int(
                        input(f"期权 {i + 1}: AAPL {opt_type} ${opt['strike']} - 购买几手? (默认1): ") or "1")
                    quantity = max(1, min(10, quantity))    # 限制在1-10手
                except:
                    quantity = 1
                    print(f"使用默认值: 1手")

                # 计算期权总价值：价格 × 数量 × 100（每手100股）
                value = opt['lastPrice'] * quantity * 100

                # 限制隐含波动率在合理范围内
                volatility = min(opt['impliedVolatility'], 0.8)

                # 计算期权Delta值（对股价变动的敏感度）
                if opt_type == '看涨':
                    moneyness = (180 - opt['strike']) / 180  # 虚实值程度
                    delta = max(0.1, min(0.9, 0.5 + moneyness * 0.5))   # 看涨Delta在0.1-0.9之间
                else:
                    moneyness = (opt['strike'] - 180) / 180
                    delta = min(-0.1, max(-0.9, -0.5 + moneyness * 0.5))    # 看跌Delta在-0.9到-0.1之间

                # 判断期权虚实值状态
                moneyness_status = "实值" if (opt['strike'] < 180 and opt_type == '看涨') or (
                    opt['strike'] > 180 and opt_type == "看跌") else '虚值'

                # 创建期权信息字典
                option_info = {
                    'name': f"AAPL {opt_type} ${opt['strike']}",
                    'type': opt_type,        # 期权类型
                    'value': value,          # 期权总价值
                    'price': opt['lastPrice'], # 期权单价
                    'strike': opt['strike'],   # 行权价
                    'quantity': quantity,    # 购买手数
                    'delta': delta,          # Delta值
                    'vol': volatility        # 波动率
                }
                option_portfolio.append(option_info)

        # 处理无数据情况
        if len(option_portfolio) == 0:
            print(f"没有找到有效的期权数据")
        else:
            # 打印期权投资详情
            print("\n期权投资详情:")
            for option in option_portfolio:
                # 重新计算虚实值状态用于显示
                moneyness_status = '实值' if (option['strike'] < 180 and option['type'] == '看涨') or (
                    option['strike'] > 180 and option['type'] == '看跌') else '虚值'
                print(f"✓ {option['name']} ({moneyness_status}): 价格${option['price']:.2f}, "
                      f"{option['quantity']}手, 价值${option['value']:,.0f}")

        return option_portfolio

    def monte_carlo_var(self, bonds, options, simulations=5000, days=10):
        """
        蒙特卡洛模拟计算风险价值(VaR)
        通过模拟市场情景来估计组合的未来价值分布
        """
        # 计算当前组合价值
        bond_value = sum(b['value'] for b in bonds) if bonds else 0
        option_value = sum(o['value'] for o in options) if options else 0
        total_value = bond_value + option_value

        # 检查组合价值是否有效
        if total_value == 0:
            print(f"组合总价值为0，无法进行风险分析")
            return 0, np.array([0])

        # 打印组合价值分析
        print(f"\n" + "=" * 50)
        print("组合价值分析")
        print("=" * 50)
        print(f"债券总价值: ${bond_value:,.2f}")
        print(f"期权总价值: ${option_value:,.2f}")
        print(f"组合总价值: ${total_value:,.2f}")

        # 计算平均波动率
        bond_vol = np.mean([b['vol'] for b in bonds]) if bonds else 0
        option_vols = [o['vol'] for o in options] if options else [0.25]
        option_vol = np.mean(option_vols)

        # 计算期权组合平均Delta值
        option_deltas = [abs(o['delta']) for o in options] if options else [0.5]
        avg_delta = np.mean(option_deltas)

        print(f"债券波动率: {bond_vol:.1%}")
        print(f"期权波动率: {option_vol:.1%}")
        print(f"期权平均Delta: {avg_delta:.2f}")

        # 设置随机数种子确保结果可重现
        np.random.seed(42)
        # 计算时间调整因子（将年波动率转换为指定天数的波动率）
        time_factor = np.sqrt(days / 252)  # 252个交易日

        # 生成相关的随机数（债券和股票收益的相关性为30%）
        correlation = 0.3
        z1 = np.random.normal(0, 1, simulations)  # 债券随机冲击
        # 股票随机冲击（与债券相关）
        z2 = correlation * z1 + np.sqrt(1 - correlation ** 2) * np.random.normal(0, 1, simulations)

        # 计算债券和股票的收益率冲击
        bond_returns = z1 * bond_vol * time_factor
        stock_returns = z2 * option_vol * time_factor

        # 模拟未来组合价值
        future_values = []
        for i in range(simulations):
            # 债券价值变化
            bond_change = bond_value * bond_returns[i] if bonds else 0
            # 期权价值变化（考虑Delta暴露）
            option_change = option_value * avg_delta * stock_returns[i] if options else 0
            # 计算未来价值
            future_value = total_value + bond_change + option_change
            future_values.append(future_value)

        return total_value, np.array(future_values)

    def calculate_risk(self, current_value, future_values):
        """
        计算风险指标：VaR和CVaR
        VaR: 风险价值，在一定置信水平下的最大可能损失
        CVaR: 条件风险价值，超过VaR的期望损失
        """
        # 计算损益分布
        pnl = future_values - current_value

        # 计算95%置信水平的VaR（取第5百分位数的负值）
        var_95 = -np.percentile(pnl, 5)
        # 计算99%置信水平的VaR（取第1百分位数的负值）
        var_99 = -np.percentile(pnl, 1)

        # 计算CVaR（超过VaR的所有损失的平均值）
        cvar_95 = -pnl[pnl <= -var_95].mean()
        cvar_99 = -pnl[pnl <= -var_99].mean()

        return {
            '95% VaR': var_95,   # 95%置信水平下的风险价值
            '95% CVaR': cvar_95, # 95%置信水平下的条件风险价值
            '99% VaR': var_99,   # 99%置信水平下的风险价值
            '99% CVaR': cvar_99  # 99%置信水平下的条件风险价值
        }, pnl

    def plot_results(self, pnl, risk_metrics, bonds, options):
        '''
        绘制风险分析结果图表
        包含4个子图：损益分布、风险指标比较、组合成分、风险指标汇总
        '''
        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 子图1：损益分布直方图
        ax1.hist(pnl, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        # 标记VaR水平线
        ax1.axvline(-risk_metrics['95% VaR'], color='red', linestyle='--', label='95% VaR')
        ax1.axvline(-risk_metrics['99% VaR'], color='darkred', linestyle='--', label='99% VaR')
        ax1.set_xlabel('损益 ($)')
        ax1.set_ylabel('频率')
        ax1.set_title('组合损益分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：VaR和CVaR比较柱状图
        var_values = [risk_metrics['95% VaR'], risk_metrics['99% VaR']]
        cvar_values = [risk_metrics['95% CVaR'], risk_metrics['99% CVaR']]

        x = np.arange(2)  # x轴位置
        # 绘制VaR柱状图
        ax2.bar(x - 0.2, var_values, 0.4, label='VaR', alpha=0.7, color='orange')
        # 绘制CVaR柱状图
        ax2.bar(x + 0.2, cvar_values, 0.4, label='CVaR', alpha=0.7, color='red')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['95%', '99%'])
        ax2.set_ylabel('风险价值 ($)')
        ax2.set_title('VaR vs CVaR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3：组合成分饼图
        bond_total = sum(b['value'] for b in bonds) if bonds else 0
        option_total = sum(o['value'] for o in options) if options else 0
        if bond_total + option_total > 0:
            ax3.pie([bond_total, option_total], labels=['债券', '期权'], autopct='%1.1f%%',
                    colors=['lightblue', 'lightcoral'])
            ax3.set_title('组合成分占比')

        # 子图4：风险指标汇总表格
        metrics_data = [
            ['95% VaR', f"${risk_metrics['95% VaR']:,.0f}"],
            ['95% CVaR', f"${risk_metrics['95% CVaR']:,.0f}"],
            ['99% VaR', f"${risk_metrics['99% VaR']:,.0f}"],
            ['99% CVaR', f"${risk_metrics['99% CVaR']:,.0f}"]
        ]
        ax4.axis('off')  # 关闭坐标轴
        # 创建表格
        table = ax4.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # 调整表格大小
        ax4.set_title('风险指标汇总')

        # 调整布局并显示图表
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        '''
        运行完整的风险分析流程
        包括：数据加载、组合创建、风险计算、结果展示
        '''
        print(f"===== 交互式组合风险分析 =====")
        print(f"现在您可以自定义投资金额！\n")

        # 1. 加载市场数据
        treasury, securities, options, yield_rate = self.load_data()

        # 2. 创建投资组合（用户交互）
        bonds = self.create_bond_portfolio(securities)
        option_portfolio = self.create_option_portfolio(options)

        # 检查是否有有效数据
        if not bonds and not option_portfolio:
            print(f"没有找到有效的债券或期权数据，无法进行分析")
            return

        # 3. 蒙特卡洛模拟计算风险
        current_value, future_values = self.monte_carlo_var(bonds, option_portfolio)
        risk_metrics, pnl = self.calculate_risk(current_value, future_values)

        # 4. 显示风险分析结果
        print(f"\n📊 风险分析结果 (10天持有期):")
        for metric, value in risk_metrics.items():
            # 显示绝对金额和相对百分比
            print(f"  {metric}: ${value:,.2f} ({value / current_value * 100:.2f}%)")

        # 5. 绘制结果图表
        self.plot_results(pnl, risk_metrics, bonds, option_portfolio)

        print("\n✅ 分析完成!")


# 程序入口点
if __name__ == "__main__":
    # 创建分析器实例并运行分析
    analyzer = InteractivePortfolioRiskAnalyzer()
    analyzer.run_analysis()


'''
==============组合与风险管理基础总结==================
1️⃣ 学习目标
构建 债券 + 期权 的投资组合。
理解并计算组合的基本风险指标：VaR（风险价值） 和 CVaR（条件风险价值）。

2️⃣ 核心任务
1. 数据准备
    -使用真实市场数据：
        国债收益率（用于无风险利率参考）
        债券信息（证券价格、类型、期限）
        AAPL 期权数据（看涨、看跌合约）
    -数据清洗和筛选：过滤无价格、无隐含波动率或异常值的记录。

2. 组合构建
    -债券组合：用户可交互式选择投资的债券及金额，估算债券价值和波动率。
    -期权组合：用户可选择期权合约、购买手数，计算期权价值、Delta值及波动率，并标记期权虚实值状态。

3. 风险模拟
    -使用 蒙特卡洛模拟 模拟未来组合价值，考虑债券与期权波动率及二者相关性。
    -通过模拟生成未来价值分布，用于计算风险指标。

4. 风险指标计算
    -VaR（Value at Risk）：在指定置信水平下可能的最大损失。
    -CVaR（Conditional VaR）：超过 VaR 的平均损失，更关注尾部风险。
    -计算 95% 和 99% 两个置信水平下的 VaR 和 CVaR。

5. 结果可视化
    -绘制组合 损益分布直方图，标记 VaR 水平线。
    -绘制 VaR 与 CVaR 比较柱状图。
    -绘制 组合成分占比饼图（债券 vs 期权）。
    -绘制 风险指标汇总表格，方便快速查看。

3️⃣ 技术亮点
交互式组合构建：用户可自行选择投资金额、购买手数，灵活性高。
蒙特卡洛模拟：考虑债券与期权的波动率及相关性，生成更接近真实市场的风险分布。
风险指标全面：同时提供 VaR 和 CVaR，多置信水平分析组合潜在损失。
可视化：用图表和表格展示组合结构与风险结果，直观理解投资组合风险。

4️⃣ 收获与理解
学会了如何加载并处理真实金融数据，而不是仅依赖假数据或假设。
理解了债券和期权组合的价值构成及波动率对组合风险的影响。
掌握了 VaR 和 CVaR 的概念与计算方法，并学会用蒙特卡洛模拟估算风险。
掌握了交互式 Python 脚本设计，可以让用户自定义投资组合并动态计算风险指标。
'''