'''
第14天
练习:
    学习策略参数优化：使用 Backtrader 自动搜索最优参数
'''

import backtrader as bt
import pandas as pd

# 加载数据
def load_data():
    file_path = './DIS_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

# results 是全局变量
results = []    # 把优化参数数据存储到这里来

class MACD_strategy(bt.Strategy):
    params = (
        ('fast', 12),
        ('slow', 26),
        ('signal', 9),
        ('take_profit', 0.05),
        ('stop_loss', 0.1)
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1 = self.params.fast,
            period_me2 = self.params.slow,
            period_signal = self.params.signal
        )

        self.cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.buy_price = None

    def next(self):
        if not self.position:
            if self.cross > 0:  #交叉买入
                self.buy()
                self.buy_price = self.data.close[0]
        else:
            change = (self.data.close[0] - self.buy_price) / self.buy_price
            if self.cross < 0 or change >= self.params.take_profit or change <= -self.params.stop_loss:
                self.close()

    def stop(self):
        # 现在 这里是局部变量.  我需要把下面的代码加入到全局变量里.
        global results
        final_value = self.broker.getvalue()
        # print(f"fast={self.params.fast}, "
        #       f"slow={self.params.slow}, "
        #       f"signal={self.params.signal}, "
        #       f"tp={self.params.take_profit}, "
        #       f"sl={self.params.stop_loss}, "
        #       f"Final Value={self.broker.getvalue():.2f}")

        result = {
            'fast': self.params.fast,
            'slow': self.params.slow,
            'signal': self.params.signal,
            'take_profit': self.params.take_profit,
            'stop_loss': self.params.stop_loss,
            'Final Value': final_value
                  }
        results.append(result)

def run_testing():
    cerebro = bt.Cerebro()

    data = load_data()   # 需要调用load_data
    cerebro.adddata(data)

    # 优化参数
    cerebro.optstrategy(
        MACD_strategy,
        fast=range(10, 15, 1),     # fast EMA 参数
        slow=range(24, 30, 1),     # slow EMA 参数
        signal=range(8, 15, 1),    # signal EMA 参数
        take_profit=[0.05, 0.1, 0.5],
        stop_loss=[0.02, 0.1, 0.2]
    )



    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.run(maxcpus=1)

    #优化参数太多, 绘制图只能绘制一个.
    # cerebro.plot()

    # 找出最优参数
    best_result = max(results, key=lambda x: x['Final Value'])
    print(f"\n最优参数组合:")
    for k, v in best_result.items():
        print(f"{k}: {v}")

# 忘记调用
if __name__ == "__main__":
    run_testing()








