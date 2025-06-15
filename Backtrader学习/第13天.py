'''
第13天:
	**练习：**在策略中实现MACD并设定柱状图判断逻辑，回测并输出策略表现图表。
'''

import backtrader as bt
import pandas as pd

def load_data():
    file_path = './COIN_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

class MACD_Strategy(bt.Strategy):
    params = (
        ('take_profit', 0.05),    # 止盈
        ('stop_loss', 0.2)       # 止损
    )

    def __init__(self):
        macd = bt.indicators.MACD(self.data)
        self.macd_hist = macd.macd - macd.signal
        self.crossover = bt.indicators.CrossOver(self.macd_hist, 0)
        self.buy_price = None
        self.buy_date = None

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_price = self.data.close[0]
        if not self.position:
            if self.crossover > 0:
                self.buy()
                self.buy_price = current_price
                self.buy_date = current_date
                self.log(f"买入时间: {current_date}, 买入价格: {current_price:.2f}")

        else:
            # 当前涨跌幅
            change = (current_price - self.buy_price) / self.buy_price
            reason = None

            # 止盈
            if change >= self.params.take_profit:
                reason = '止盈触发'

            # 止损
            elif change <= -self.params.stop_loss:
                reason = '止损触发'

            # MACD 平仓
            elif self.crossover < 0:
                reason = 'MACD 死叉'

            if reason:
                self.sell()
                self.log(f"{reason}")
                self.log(f"买入时间: {self.buy_date}, 买入价格:{self.buy_price:.2f}")
                self.log(f"卖出时间: {current_date}, 卖出价格: {current_price:.2f}")
                self.log(f"涨跌幅: {change:.2%}")
                self.buy_price = None
                self.buy_date = None


def run_testing():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACD_Strategy)
    data = load_data()
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f" 初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f" 最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    run_testing()


'''' 算了就这样吧.   有显示时间就显示吧. 打印出来的前面有时间是卖出时间. '''