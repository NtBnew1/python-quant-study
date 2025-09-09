'''
第11天：

资金管理策略深入（动态仓位调整、风险预算）
加强止损止盈和风险控制模块
练习：实现动态仓位控制，回测验证风险指标变化

'''

'''我们回去第10天用回归的方式找比较好的股票代码.'''
'''2025 年各公司平均 FactorScore (按高到低)
company
SMCI, AVGO, ALAB, KC, TEM, TSLA, EDU, XPEV, HIMS, XNET, TAL, HOOD, LI, PLTR, IBRX 
用这些公司代码来计算回测. '''

import pandas as pd
import numpy as np
import os
import backtrader as bt    # 忘记用backtrader 回测了
from datetime import datetime, timedelta

# ================= 策略类 =================
class Strategy(bt.Strategy):
    # 策略参数（可以在调用 addstrategy 时修改）
    params = (
        ('ma5', 5),         # 5日均线
        ('ma20', 20),       # 20日均线
        ('ma50', 50),       # 50日均线
        ('stop_loss', 0.4), # 止损比例（跌 40% 卖出）
        ('take_profit', 0.5)# 止盈比例（涨 50% 卖出）
    )

    def __init__(self):
        # 初始化各类字典，分别存储不同股票的数据
        self.madata = {}      # 存储均线和波动率指标
        self.order = {}       # 存储每只股票的订单
        self.buyprice = {}    # 存储买入价
        self.cross_ma5 = {}   # 5日均线上穿 50日均线
        self.cross_ma20 = {}  # 20日均线上穿 50日均线

        # 给每只股票都设置指标
        for d in self.datas:
            ma5 = bt.indicators.MovingAverageSimple(d.close, period=self.p.ma5)
            ma20 = bt.indicators.MovingAverageSimple(d.close, period=self.p.ma20)
            ma50 = bt.indicators.MovingAverageSimple(d.close, period=self.p.ma50)
            vol = bt.indicators.StandardDeviation(d.close, period=20)       # 20日标准差（波动率）

            # 存到字典
            self.madata[d] = {'ma5': ma5, 'ma20': ma20, 'ma50': ma50, 'vol': vol}
            # CrossOver 指标
            self.cross_ma5[d] = bt.indicators.CrossOver(self.madata[d]['ma5'], self.madata[d]['ma50'])
            self.cross_ma20[d] = bt.indicators.CrossOver(self.madata[d]['ma20'], self.madata[d]['ma50'])

            # 初始化订单和买入价
            self.order[d] = None
            self.buyprice[d] = None
            d.plotinfo.plotmaster = None  # 单独绘制

    def next(self):
        """每个交易日都会调用"""
        for d in self.datas:
            pos = self.getposition(d).size  # 当前持仓量

            # === 动态仓位计算：波动率越大，仓位越小 ===
            vol = self.madata[d]['vol'][0]
            target_pct = max(0.1, min(1.0, 1/(1+vol*10)))   # 控制仓位比例（0.1~1.0之间）

            # === 买入逻辑 ===
            if pos == 0 and self.order[d] is None:  # 只在空仓时买入
                # 条件：5日 > 50日 且 20日 > 50日
                if ((self.cross_ma5[d][0] == 1) or (self.madata[d]['ma5'][0] > self.madata[d]['ma50'][0])) and \
                        ((self.cross_ma20[d][0] == 1) or (self.madata[d]['ma20'][0] > self.madata[d]['ma50'][0])):
                    size = int(self.broker.getcash() * target_pct / d.close[0])     # 动态仓位下的买入股数
                    if size > 0:
                        self.order[d] = self.buy(data=d, size=size)
                        self.buyprice[d] = d.close[0]

            # === 卖出逻辑（止损 / 止盈） ===
            if pos > 0 and self.order[d] is None and self.buyprice[d] is not None:
                if d.close[0] <= self.buyprice[d] * ( 1 - self.p.stop_loss) or \
                    d.close[0] >= self.buyprice[d] * ( 1 + self.p.take_profit):
                    self.order[d] = self.sell(data=d, size=pos)
                    self.buyprice[d] = None     # 清空买入价

    def notify_order(self, order):
        """订单执行通知"""
        if order.status in [order.Completed]:  # 订单完成
            dt = bt.num2date(order.executed.dt).date()  # 日期
            ticker = order.data._name  # 股票代码
            size = order.executed.size  # 成交数量
            price = order.executed.price  # 成交价格

            # 打印买卖信息
            if order.isbuy():
                print(f'{dt} | {ticker} 买入: {size}股, 价格{price:.2f}')
            elif order.issell():
                print(f'{dt} | {ticker} 卖出: {size}股, 价格{price:.2f}')
            self.order[order.data] = None


# ===============股票池=============
tickers = ['SMCI', 'AVGO', 'ALAB', 'KC', 'TEM', 'TSLA', 'EDU',
           'XPEV', 'HIMS', 'XNET', 'TAL', 'HOOD', 'LI', 'PLTR', 'IBRX']

# 创建回测引擎
cerebro = bt.Cerebro()
cerebro.addstrategy(Strategy)      # 加载策略
cerebro.broker.setcash(100000)     # 初始资金 10万
cerebro.broker.setcommission(commission=0.001)  # 手续费 0.1%

# ================= 批量读取 Excel 数据 =================
for ticker in tickers:
    # cerebro = bt.Cerebro()
    file_path = f"./{ticker}_all_data.xlsx"
    if not os.path.exists(file_path):
        print(f'文件不存在:{file_path}, 跳过{ticker}')
        continue

    # 读取 Excel，取 sheet_name = 'price'
    df = pd.read_excel(file_path, sheet_name='price')
    # print(df.columns)
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 只取最近 1 年的数据
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=365)
    df = df.loc[start_date:end_date]

    # 转换成 Backtrader 可识别的数据格式
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data, name=ticker)

    # cerebro.addstrategy(Strategy)
    # cerebro.broker.setcash(100000)  # 初始资金
    # cerebro.broker.setcommission(commission=0.001)

# ================= 运行回测 =================
print(f"============{ticker} 回测开始============")
print('初始资金: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print("回测结束资金: %.2f" % cerebro.broker.getvalue())
# cerebro.plot()    # 绘图（多股票会比较乱）


'''
=========================总结报告=======================

 一、实验目标
- 使用第10天回归筛选出的股票池（15家公司），在 Backtrader 框架中进行回测。  
- 深入学习资金管理策略，包括：动态仓位调整、风险预算、止损止盈。  

 二、策略设计
- **均线指标**  
  - MA5、MA20、MA50  
  - 买入条件：MA5 > MA50 且 MA20 > MA50  

- **动态仓位**  
  target\_pct = \max(0.1, \min(1.0, \frac{1}{1 + vol \times 10}))  
  波动率越大，仓位越小（范围 0.1 ~ 1.0）。  

- **风险控制**  
  - 止损：股价跌 40% 卖出  
  - 止盈：股价涨 50% 卖出  

 三、回测设置
- 股票池：SMCI, AVGO, ALAB, KC, TEM, TSLA, EDU, XPEV, HIMS, XNET, TAL, HOOD, LI, PLTR, IBRX  
- 数据周期：最近一年  
- 初始资金：100,000 美元  
- 手续费：0.1%  

 四、结果
- 初始资金：100,000 美元  
- 回测结束资金：由 `cerebro.broker.getvalue()` 输出  
- 交易日志：在 `notify_order` 打印买卖记录  

 五、评价
**不足**  
1. 仓位计算方法较简单  
2. 止损止盈参数固定，缺乏个股适应性  
3. 多标的绘图较乱  

 七、总结
本策略实现了 **趋势确认 + 动态仓位 + 风险控制** 的多股票回测框架，已具备初步实盘雏形。后续可通过参数优化与指标完善，进一步提升策略表现。

'''