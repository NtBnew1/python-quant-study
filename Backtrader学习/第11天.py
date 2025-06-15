'''
第11天:
	练习：*在Backtrader中加入成交量和成交量均线，设定基于成交量变化的买卖规则，并运行回测。
'''

import backtrader as bt
import pandas as pd

def load_data():
    file_path = './AAL_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0
    # print(df.head())

    data = bt.feeds.PandasData(dataname=df)
    return data

class volume_strategy(bt.Strategy):
    ''' 加入其它指标(MA)'''

    def __init__(self):
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=20)  # 成交量平均线20天
        """加入ma"""
        self.price_ma = bt.indicators.MovingAverageSimple(self.data.close, period=20)

    def next(self):
        ''' 优化代码. 看起来比较好读.'''
        current_date = self.datas[0].datetime.date(0)
        current_volume = self.data.volume[0]
        volume_avg = self.vol_ma[0]
        current_close_price = self.data.close[0]
        close_avg = self.price_ma

        if not self.position:
            # 成交量 > 均量 并且 收盘价 > 收盘均价: 买入
            if current_volume >= volume_avg * 1.5 and current_close_price > close_avg:
                self.buy()
                # 加入买入时间, 成交量, 均量, 和价格.
                print(f"买入时间: {current_date}, "
                      f"成交量: {current_volume}, "
                      f"均量:{volume_avg}, "
                      f"价格: {current_close_price}")

        else:
            # 成交量 < 均量 并且 收盘价 < 收盘均价: 卖出
            if current_volume <= volume_avg and current_close_price < close_avg:
                self.sell()
                # 加入卖出时间, 成交量, 均量, 和价格
                print(f"卖出时间: {current_date}, "
                      f"成交量: {current_volume}, "
                      f"均量: {volume_avg}, "
                      f"价格: {current_close_price}")

def run_testing():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(volume_strategy)

    data = load_data()
    cerebro.adddata(data)

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    run_testing()