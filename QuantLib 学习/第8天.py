'''
Day 8：综合实战项目
目标：
- 将前面学的内容整合到一个小型实战项目。
任务：
- 构建债券+期权组合。
- 定价、收益率分析、风险管理、组合优化。
- 保存完整脚本到GitHub。
输出：完整QuantLib项目示例。
'''

import pandas as pd
import numpy as np
import QuantLib as ql
from scipy.optimize import minimize
import glob, os


# ----------------------------
# 定义投资组合分析类
# ----------------------------
class QuantitativePortfolioAnalyzer:
    def __init__(self):
        # 数据存储
        self.bonds = None              # 债券数据
        self.treasury = None           # 国债收益率数据
        self.options = None            # 期权数据
        self.bond_characteristics = [] # 每只债券的特征信息
        self.optimal_weights = None    # 优化后的组合权重
        self.portfolio_metrics = {}    # 投资组合指标
        self.rf_rate = 0.04            # 默认无风险利率

    # ----------------------------
    # 加载数据
    # ----------------------------
    def load_data(self):
        print(f"加载数据.......")
        # 加载债券数据
        try:
            self.bonds = pd.read_csv('./Securities.csv', encoding='latin1')
            print(f"债卷数据{len(self.bonds)}条")
        except Exception as e:
            print(f"债卷加载失败: {e}")
            return False

        # 加载国债收益率数据
        try:
            self.treasury = pd.read_excel('./US_Treasury_Yields.xlsx')
            self.rf_rate = self.treasury.iloc[-1]['DGS10'] /100
            print(f"国债收益率数据: 最近10年期收益率{self.rf_rate*100:.2f}%")
        except Exception as e:
            print(f"国债加载失败: {e}")
            return False

        # 加载期权数据
        option_files = glob.glob('./*_options.xlsx')
        option_data = []
        for f in option_files[:5]:   # 只取前5个期权文件
            try:
                calls = pd.read_excel(f, sheet_name='Calls').head(50)
                puts = pd.read_excel(f, sheet_name='Puts').head(50)
                calls['optionType']='Call'
                puts['optionType']='Put'
                df = pd.concat([calls, puts], ignore_index=True)
                df['Source_File'] = os.path.basename(f)
                option_data.append(df)
            except:
                continue
        if option_data:
            self.options = pd.concat(option_data, ignore_index=True)
            print(f"期权数据: {len(self.options)}个")
        else:
            print(f"没有有效期权数据")
            return False
        return True

    # ----------------------------
    # 债券分析
    # ----------------------------
    def analyze_bonds(self):
        latest_treasury = self.treasury.iloc[-1]
        self.bond_characteristics = []

        for idx, bond in self.bonds.iterrows():
            security_type = bond.get('Security Type', 'Unknown')
            security_term = bond.get('Security Term', '1-year')

            # 根据期限选择收益率
            if '10-year' in str(security_term):
                ytm = latest_treasury['DGS10']/100
                duration=10
            elif '5-year' in str(security_term):
                ytm = latest_treasury['DGS5']/100
                duration=5
            else:
                ytm = latest_treasury['DGS2']/100
                duration=2

            # QuantLib 定价
            try:
                today = ql.Date.todaysDate()
                ql.Settings.instance().evaluationDate = today
                schedule = ql.Schedule(today, today+ql.Period(duration, ql.Years),
                                       ql.Period(ql.Annual), ql.NullCalendar(),
                                       ql.Unadjusted, ql.Unadjusted,
                                       ql.DateGeneration.Backward, False)
                bond_ql = ql.FixedRateBond(1, 100, schedule, [ytm], ql.ActualActual())
                bond_price = bond_ql.cleanPrice()   # 获取债券价格
            except:
                bond_price = 100  # 默认价格

            self.bond_characteristics.append({
                'type': security_type,
                'term': security_term,
                'yield':ytm,
                'duration': duration,
                'price': bond_price
            })

        # 计算平均债券收益率
        self.avg_bond_yield = np.mean([b['yield'] for b in self.bond_characteristics])*100
        print(f"平均债卷收益率: {self.avg_bond_yield:.2f}%")
        return True

    # ----------------------------
    # 期权分析
    # ----------------------------
    def analyze_options(self):
        vols =[]
        strikes = []

        # 提取隐含波动率和行权价
        for idx, row in self.options.iterrows():
            iv = pd.to_numeric(row.get('impliedVolatility', np.nan), errors='coerce')
            strike=pd.to_numeric(row.get('strike', np.nan), errors='coerce')
            if not pd.isna(iv) and iv>0:
                vols.append(iv)
            if not pd.isna(strike):
                strikes.append(strike)

        # 使用中位数波动率和平均行权价
        self.avg_option_vol = np.median(vols)*100 if vols else 50
        self.underlying_price = np.mean(strikes)*1.05 if strikes else 200
        print(f"平均期权隐含波动率: {self.avg_option_vol:.2f}%")
        return True

    # ----------------------------
    # 期权希腊值计算
    # ----------------------------
    def calculate_optino_greeks(self):
        today=ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate=today

        # 定义欧式看涨期权
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.underlying_price)
        exercise = ql.EuropeanExercise(today+ql.Period(90, ql.Days))
        option = ql.VanillaOption(payoff, exercise)

        # 构建定价过程
        spot = ql.QuoteHandle(ql.SimpleQuote(self.underlying_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rf_rate, ql.Actual365Fixed()))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), self.avg_option_vol/100, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div_ts, flat_ts, vol_ts)

        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        # 计算希腊值
        self.option_delta = option.delta()
        self.option_gamma = option.gamma()
        self.option_vega = option.vega()/100
        self.option_theta = option.theta()/365
        print(f"期权希腊值计算完成: Delta={self.option_delta:.3f}, Gamma={self.option_gamma:.6f}")
        return True

    # ----------------------------
    # 投资组合优化
    # ----------------------------
    def optimize_portfolio(self):
        np.random.seed(42)

        # 模拟日收益
        bond_daily=self.avg_bond_yield/100/252
        option_daily = self.avg_option_vol/100/252
        bond_returns = np.random.normal(bond_daily, 0.001, 10000)
        option_returns = np.random.normal(option_daily, 0.02, 10000)

        # 最大夏普比率优化
        def objective(w):
            r = w[0]*bond_returns + w[1]*option_returns
            return -np.mean(r)/np.std(r)

        cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1})
        bounds=[(0,1),(0,1)]
        res = minimize(objective, [0.5,0.5], bounds=bounds, constraints=cons)
        self.optimal_weights = res.x

        # 计算组合收益和风险
        port_return = np.mean(self.optimal_weights[0]*bond_returns + self.optimal_weights[1]*option_returns)*252*100
        port_risk = np.std(self.optimal_weights[0]*bond_returns + self.optimal_weights[1]*option_returns)*np.sqrt(252)*100
        print(f"最优组合: 债卷{self.optimal_weights[0]:.2f}, 期权: {self.optimal_weights[1]:.2f}")
        print(f"组合年化收益{port_return:.2f}%, 年化风险{port_risk:.2f}%")
        return True

    # ----------------------------
    # 运行完整分析流程
    # ----------------------------
    def run_complete_analysis(self):
        steps = [self.load_data, self.analyze_bonds, self.analyze_options,
                 self.calculate_optino_greeks, self.optimize_portfolio]
        for step in steps:
            if not step():
                print(f"分析流程中断")
                return False
        print(f"综合实战项目完成")
        return True

# ----------------------------
# 主函数入口
# ----------------------------
def main():
    analyzer = QuantitativePortfolioAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()


'''
总结说明
1. 项目目标
    将债券和期权组合构建为一个量化投资组合。
    使用 QuantLib 进行债券定价和期权希腊值计算。
    分析债券收益率、期权隐含波动率。
    通过蒙特卡洛模拟和 scipy.optimize 计算最优组合权重。
    输出组合年化收益、年化风险、夏普比率。

2. 核心模块
    数据加载 (load_data)
        读取债券、国债、期权数据。
        支持多期权文件读取和合并。
    
    债券分析 (analyze_bonds)
        根据国债收益率计算每只债券的到期收益率 (YTM)。
        使用 QuantLib 定价债券，得到债券市场价值。
    
    期权分析 (analyze_options)
        提取隐含波动率 (IV) 和行权价。
        计算期权市场平均波动率和标的价格。
    
    期权希腊值计算 (calculate_optino_greeks)
        使用 QuantLib 构建欧式看涨期权。
        计算 Delta、Gamma、Vega、Theta，帮助理解期权风险敏感度。
    
    组合优化 (optimize_portfolio)
        利用蒙特卡洛模拟债券和期权日收益。
'''