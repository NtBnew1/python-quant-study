'''
整合多个技术指标，构建一个多因子交易策略。
练习：
- 综合使用均线（SMA）、MACD柱状图、成交量均线和RSI三大指标：
    - 满足均线金叉 + MACD柱状图翻红 + 成交量放大时买入；
    - 满足均线死叉或 MACD柱状图翻绿时卖出；
- 加入止盈止损规则，控制风险；
- 运行回测并绘制策略的收益曲线和买卖点图表。
'''

# 先导入库
import backtrader as bt
import pandas as pd


# 加载数据
def load_data():
    file_path = './AI_year_data.csv'          # 我这边可以改是因为我有保存这些股票的数据.
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    print(df.head())

    data = bt.feeds.PandasData(dataname=df)
    return data

# 添加SMA, MACD, 成交量均线, 和RSI策略
class Multi_Strategy(bt.Strategy):
    params = (
        ('sma_short', 10),      # SMA 10日
        ('sma_long', 50),       # SMA 50日 长线
        ('volume_sma_period', 20),   # 成交量均线20日
        ('rsi_period', 14),      # RSI 14日线
        ('rsi_sell', 70),        # RSI >70 以上卖
        ('take_profit', 0.2),   # 止盈20%
        ('stop_loss', 0.05)     # 止损5%
    )

    def __init__(self):
        # SMA 均线
        self.sma_short = bt.indicators.SMA(self.data.close, period=self.params.sma_short)
        self.sma_long = bt.indicators.SMA(self.data.close, period=self.params.sma_long)

        # MACD 柱状图
        self.macd = bt.indicators.MACD(self.data.close)
        self.macd_hist = self.macd.macd - self.macd.signal

        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # 成交量均线
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.params.volume_sma_period)

        self.buy_price = None

    def next(self):
        if not self.position:
            # 买入条件
            sma_cross = self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]
            macd_turn_up = self.macd_hist[0] > 0 and self.macd_hist[-1] <= 0
            volume_up = self.data.volume[0] > self.volume_sma[0]
            rsi_ok = self.rsi[0] > 40 and self.rsi[0] < 60

            if sma_cross and (macd_turn_up or volume_up):       # SMA_Cross 是主交易, ()里面是辅助交易
                self.buy()
                self.buy_price = self.data.close[0]     # 记录买入价格
        else:
            # 当前涨跌幅
            current_price = self.data.close[0]
            price_change = (current_price - self.buy_price) / self.buy_price

            # 止盈条件
            if price_change >= self.params.take_profit:
                self.sell()
                self.buy_price = None
                return


            # 止损条件
            elif price_change <= self.params.stop_loss:
                self.sell()
                self.buy_price = None
                return


            # 卖出条件
            dead_cross = self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]
            macd_turn_down = self.macd_hist[0] < 0 and self.macd_hist[-1] >= 0
            rsi_toohigt = self.rsi[0] > self.params.rsi_sell

            if dead_cross or macd_turn_down or rsi_toohigt:
                self.sell()
                self.buy_price = None

def run_testing():
    cerebro = bt.Cerebro()
    data = load_data()      # 需要调用
    cerebro.adddata(data)
    cerebro.addstrategy(Multi_Strategy)     # 需要加入策略, 不然就一直卡.

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")

    cerebro.plot()

# 调用
if __name__ == "__main__":
    run_testing()

