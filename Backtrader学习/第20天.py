'''
第20天：
任务目标：
掌握Backtrader的参数优化功能，批量回测不同参数组合。

练习内容：
    学习cerebro.optstrategy()用法，设计参数网格；
    跑多组参数回测，采集绩效指标；
    筛选最优参数组合；
    输出优化回测报告，分析参数对策略表现的影响。
'''


# 需要导入库
import pandas as pd
import backtrader as bt

''' 添加时间'''
import time

def load_data():
    file_path = './BABA_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

# 用来收集所有回测结果
results = []            # 把results放到全局变量

class MACD(bt.Strategy):
    params = (
        ('fast', 12),
        ('slow', 26),
        ('signal', 9)
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position: # 如果没有建仓
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

    def stop(self):
        pnl = round(self.broker.getvalue() - self.broker.startingcash, 2)
        results.append({
            'fast': self.p.fast,
            'slow': self.p.slow,
            'signal': self.p.signal,
            'pnl': pnl,
            '开始日期': self.data.datetime.date(0),
            '结束日期': self.data.datetime.date(-1)         # 把回测日期输出到results里
        })
        print(f' Fast: {self.p.fast}, Slow: {self.p.slow}, Signal: {self.p.signal}, pnl: {pnl}')

def run_testing():
    global results
    cerebro = bt.Cerebro()
    data = load_data()
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    cerebro.optstrategy(
        MACD,
        fast=range(10, 17, 2),           # 10, 12, 14, 16
        slow = range(20, 31, 5),         # 20, 25, 30
        signal=range(6, 13, 3)           # 6, 9, 12
    )

    ''' 我把results改成全局变量.  就不需要用MACD调用results.  只要results就新. '''
    # 要先run, 之后再保存到results里.
    results.clear()        # 先清空, 确保运行干净

    start = time.time()
    cerebro.run(maxcpus=1)      # 再运行
    print(f'参数优化完成, 用时: {round(time.time() - start, 2)}秒')

    # 分析最佳参数
    df = pd.DataFrame(results)
    print(df.head())        # 打印看看内容
    best = df.sort_values(by='pnl', ascending=False).iloc[0]
    print(f"最佳参数组合")
    print(f"{len(results)}组 => Fast: {best['fast']}, Slow: {best['slow']}, Signal: {best['signal']}, pnl: {best['pnl']}")

    #  保存优化结果
    df.to_csv('./BABA_MACD_优化结果.csv', index=False, encoding='utf-8-sig')        # ./ 这是添加到目录里

    ''' 保存成功.'''

    # cerebro.plot()    # 优化不能绘制图

# 忘记调用了
if __name__ == "__main__":
    run_testing()