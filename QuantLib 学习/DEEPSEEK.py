import pandas as pd
import numpy as np
import QuantLib as ql
from scipy.optimize import minimize
import glob, os
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

class QuantitativePortfolioAnalyzer:
    """Day 8 综合实战项目：债券+期权组合分析"""

    def __init__(self):
        self.bonds = None
        self.treasury = None
        self.options = None
        self.bond_characteristics = []
        self.optimal_weights = None
        self.portfolio_metrics = {}
        self.rf_rate = 0.04  # 默认无风险利率

    def load_data(self):
        """加载债券、国债和期权数据"""
        print("📁 加载数据...")

        try:
            self.bonds = pd.read_csv("./Securities.csv", encoding='latin1')
            print(f"✅ 债券数据: {len(self.bonds)} 条")
        except Exception as e:
            print(f"❌ 债券加载失败: {e}")
            return False

        try:
            self.treasury = pd.read_excel("./US_Treasury_Yields.xlsx")
            self.rf_rate = self.treasury.iloc[-1]['DGS10'] / 100
            print(f"✅ 国债收益率数据: 最新10年期收益率 {self.rf_rate*100:.2f}%")
        except Exception as e:
            print(f"❌ 国债加载失败: {e}")
            return False

        option_files = glob.glob("./*_options.xlsx")
        option_data = []
        for f in option_files[:5]:
            try:
                calls = pd.read_excel(f, sheet_name='Calls').head(50)
                puts = pd.read_excel(f, sheet_name='Puts').head(30)
                calls['optionType']='Call'
                puts['optionType']='Put'
                df = pd.concat([calls, puts], ignore_index=True)
                df['Source_File'] = os.path.basename(f)
                option_data.append(df)
            except:
                continue

        if option_data:
            self.options = pd.concat(option_data, ignore_index=True)
            print(f"✅ 期权数据: {len(self.options)} 个")
        else:
            print("⚠️ 没有有效期权数据")
            return False
        return True

    def analyze_bonds(self):
        """债券收益率分析 + QuantLib定价"""
        latest_treasury = self.treasury.iloc[-1]

        self.bond_characteristics = []

        for idx, bond in self.bonds.iterrows():
            security_type = bond.get('Security Type','Unknown')
            security_term = bond.get('Security Term','1-Year')

            # 计算债券收益率和久期
            if '10-Year' in str(security_term):
                ytm = latest_treasury['DGS10']/100
                duration = 10
            elif '5-Year' in str(security_term):
                ytm = latest_treasury['DGS5']/100
                duration = 5
            else:
                ytm = latest_treasury['DGS2']/100
                duration = 2

            # QuantLib定价
            try:
                today = ql.Date.todaysDate()
                ql.Settings.instance().evaluationDate = today
                schedule = ql.Schedule(today, today + ql.Period(duration, ql.Years),
                                       ql.Period(ql.Annual), ql.NullCalendar(),
                                       ql.Unadjusted, ql.Unadjusted,
                                       ql.DateGeneration.Backward, False)
                bond_ql = ql.FixedRateBond(1, 100, schedule, [ytm], ql.ActualActual())
                bond_price = bond_ql.cleanPrice()
            except:
                bond_price = 100

            self.bond_characteristics.append({
                'type': security_type,
                'term': security_term,
                'yield': ytm,
                'duration': duration,
                'price': bond_price
            })

        self.avg_bond_yield = np.mean([b['yield'] for b in self.bond_characteristics])*100
        print(f"📈 平均债券收益率: {self.avg_bond_yield:.2f}%")
        return True

    def analyze_options(self):
        """期权分析 + QuantLib定价"""
        vols = []
        strikes = []
        for idx, row in self.options.iterrows():
            iv = pd.to_numeric(row.get('impliedVolatility', np.nan), errors='coerce')
            strike = pd.to_numeric(row.get('strike', np.nan), errors='coerce')
            if not pd.isna(iv) and iv>0:
                vols.append(iv)
            if not pd.isna(strike):
                strikes.append(strike)
        self.avg_option_vol = np.median(vols)*100 if vols else 50
        self.underlying_price = np.mean(strikes)*1.05 if strikes else 200
        print(f"📉 平均期权隐含波动率: {self.avg_option_vol:.2f}%")
        return True

    def calculate_option_greeks(self):
        """QuantLib期权定价和希腊值计算"""
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.underlying_price)
        exercise = ql.EuropeanExercise(today + ql.Period(90, ql.Days))
        option = ql.VanillaOption(payoff, exercise)

        spot = ql.QuoteHandle(ql.SimpleQuote(self.underlying_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rf_rate, ql.Actual365Fixed()))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), self.avg_option_vol/100, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div_ts, flat_ts, vol_ts)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        self.option_delta = option.delta()
        self.option_gamma = option.gamma()
        self.option_vega = option.vega()/100
        self.option_theta = option.theta()/365
        print(f"✅ 期权希腊值计算完成: Delta={self.option_delta:.3f}, Gamma={self.option_gamma:.6f}")

        return True

    def optimize_portfolio(self):
        """组合优化（债券+期权）"""
        np.random.seed(42)
        bond_daily = self.avg_bond_yield/100/252
        option_daily = self.avg_option_vol/100/252

        bond_returns = np.random.normal(bond_daily, 0.001, 10000)
        option_returns = np.random.normal(option_daily, 0.02, 10000)

        def objective(w):
            r = w[0]*bond_returns + w[1]*option_returns
            return -np.mean(r)/np.std(r)  # 最大化夏普比率

        cons = ({'type':'eq','fun': lambda w: np.sum(w)-1})
        bounds = [(0,1),(0,1)]
        res = minimize(objective,[0.5,0.5],bounds=bounds,constraints=cons)
        self.optimal_weights = res.x

        port_return = np.mean(self.optimal_weights[0]*bond_returns + self.optimal_weights[1]*option_returns)*252*100
        port_risk = np.std(self.optimal_weights[0]*bond_returns + self.optimal_weights[1]*option_returns)*np.sqrt(252)*100
        print(f"🎯 最优组合: 债券 {self.optimal_weights[0]:.2f}, 期权 {self.optimal_weights[1]:.2f}")
        print(f"   组合年化收益 {port_return:.2f}%, 年化风险 {port_risk:.2f}%")
        return True

    def run_complete_analysis(self):
        """运行完整 Day 8 流程"""
        steps = [self.load_data, self.analyze_bonds, self.analyze_options,
                 self.calculate_option_greeks, self.optimize_portfolio]

        for step in steps:
            if not step():
                print("❌ 分析流程中断")
                return False
        print("✅ Day 8 综合实战项目完成！")
        return True

def main():
    analyzer = QuantitativePortfolioAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
