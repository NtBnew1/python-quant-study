'''
第12天：
	引入MACD指标，基于MACD柱状图制定交易信号。
'''

import backtrader as bt
import pandas as pd



# 读取数据
def load_data():
    file_path = './COIN_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0
    # print(df.head())

    data = bt.feeds.PandasData(dataname=df)
    return data


class MACD_Strategy(bt.Strategy):
    params = (
        ('take_profit', 0.06),  # 止盈: 上涨6%
        ('stop_loss', 0.2)      #止损: 下跌2%
    )
    '''止损还是20%比较好. '''

    def __init__(self):
        # 添加 MACD 指标
        macd = bt.indicators.MACD(self.data)

        # Backtrader 没有提供histo.  需要自己对手写
        self.macd_hist = macd.macd - macd.signal        # 柱状图 DIF - DEA
        self.buy_price = None


    # 加入追踪记录Log
    def log(self, txt):
        dt = self.datas[0].datetime.date(0) # 当前数据的时间
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        price = self.data.close[0]
        if not self.position:
            if self.macd_hist[0] > 0 and self.macd_hist[-1] <= 0:
                self.log(f"买入 @ {self.data.close[0]:.2f}")
                self.buy()
                self.buy_price = price
        else:
            # 用止盈止损来买卖.
            # 当前涨跌幅
            change = (price - self.buy_price) / self.buy_price

            if change >= self.params.take_profit:
                self.log(f"止盈 @ {price:.2f} (+{change*100:.2f}%)")
                self.sell()
                self.buy_price = None
            elif change <= -self.params.stop_loss:
                self.log(f" 止损 @ {price:.2f} ({change*100:.2f}%)")
                self.sell()
                self.buy_price = None

            # 或者柱状图反转信号也平仓
            elif self.macd_hist[0] < 0 and self.macd_hist[-1] >=0:
                self.log(f"MACD Exit @ {price:.2f}")
                self.sell()
                self.buy_price = None


def run_testing():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACD_Strategy)
    data = load_data()
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == "__main__":
    run_testing()


    '''试一试其它的股票'''