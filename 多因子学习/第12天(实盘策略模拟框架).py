'''
第12天：实盘策略模拟框架

目标：
1. 学习实盘环境搭建基础（数据更新、券商API简介）
2. 模拟单只股票策略交易，输出现金、持仓、总资金
3. 绘制组合价值曲线，观察策略效果
'''

# =================== 导入库 ===================
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为 SimHei（黑体），解决负号显示问题
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# =================== 数据加载函数 ===================
def load_stock_data(file_path, start_date=None, end_date=None):
    """
    加载股票数据并计算指标
    - file_path: Excel文件路径
    - start_date, end_date: 数据起止日期（可选）
    返回:
    - df: 带指标的DataFrame
    - price_col: 收盘价列名
    """
    df = pd.read_excel(file_path)

    # 自动识别日期和收盘价列
    date_col = next((c for c in df.columns if 'date' in c.lower() or '时间' in c), df.columns[0])
    price_col = next((c for c in df.columns if 'close' in c.lower() or '收盘' in c), df.columns[1])

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # 筛选日期范围
    if start_date: df = df[df.index >= pd.to_datetime(start_date)]
    if end_date: df = df[df.index <= pd.to_datetime(end_date)]

    # 计算20日均线（MA20）
    df['MA20'] = df[price_col].rolling(20).mean()

    # 计算RSI指标（14日）
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - 100 / (1 + rs)

    return df, price_col

# =================== 模拟券商类 ===================
class SimpleBroker:
    """简易模拟券商"""
    def __init__(self, cash=100000):
        self.cash = cash  # 初始现金
        self.positions = 0  # 持仓股数
        self.history = []  # 交易记录
        self.portfolio_values = []  # 每日组合价值

    def buy(self, qty, price):
        """买入操作"""
        cost = qty * price
        if self.cash >= cost:
            self.cash -= cost
            self.positions += qty
            self.history.append(('BUY', qty, price))
            print(f"📈 买入 {qty}股 @ {price:.2f}")
            return True
        return False

    def sell(self, qty, price):
        """卖出操作"""
        if self.positions >= qty:
            self.positions -= qty
            self.cash += qty * price
            self.history.append(('SELL', qty, price))
            print(f"📉 卖出 {qty}股 @ {price:.2f}")
            return True
        return False

    def record_portfolio(self, date, price):
        """记录每日组合总价值"""
        total_value = self.cash + self.positions * price
        self.portfolio_values.append((date, total_value))

# =================== 简单交易策略 ===================
def macd_rsi_strategy(df, price_col, broker):
    """
    简单策略：
    - 买入条件：价格 > MA20 且 RSI < 70，使用现金20%买入
    - 卖出条件：RSI > 70，全部卖出
    """
    for date, row in df.iterrows():
        price = row[price_col]
        ma20 = row['MA20']
        rsi = row['RSI']

        # 买入信号
        if price > ma20 and rsi < 70:
            qty = int(broker.cash / price * 0.2)
            if qty > 0:
                broker.buy(qty, price)

        # 卖出信号
        elif rsi > 70 and broker.positions > 0:
            broker.sell(broker.positions, price)

        # 记录每日组合价值
        broker.record_portfolio(date, price)

# =================== 绘制组合价值曲线 ===================
def plot_portfolio(broker):
    """绘制组合价值曲线，并标记最终总资金"""
    if not broker.portfolio_values:
        print("没有组合数据可绘制")
        return
    df = pd.DataFrame(broker.portfolio_values, columns=['date', 'value'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['value'], label='组合价值', color='blue')

    # 标记最终总资金
    final_date = df['date'].iloc[-1]
    final_value = df['value'].iloc[-1]
    plt.scatter(final_date, final_value, color='red', s=50, label=f'最终总资金: {final_value:.2f}')
    plt.text(final_date, final_value, f'{final_value:.2f}', fontsize=10, color='red', ha='right', va='bottom')

    plt.xlabel('日期')
    plt.ylabel('组合价值')
    plt.title('第12天：AAPL简易实盘策略组合价值')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =================== 主程序 ===================
if __name__ == "__main__":
    # 1. 加载AAPL数据
    file_path = 'AAPL_all_data.xlsx'
    df, price_col = load_stock_data(file_path, start_date='2020-01-01', end_date='2023-12-31')

    # 2. 初始化券商
    broker = SimpleBroker(cash=100000)

    # 3. 执行策略
    macd_rsi_strategy(df, price_col, broker)

    # 4. 输出最终结果
    print(f"\n最终现金: {broker.cash:.2f}")
    print(f"最终持仓: {broker.positions} 股")

    # 计算最终总资金（现金 + 持仓市值）
    if broker.positions > 0:
        last_price = df[price_col].iloc[-1]
        total_value = broker.cash + broker.positions * last_price
    else:
        total_value = broker.cash
    print(f"最终总资金: {total_value:.2f}")
    print(f"交易记录: {broker.history}")

    # 5. 绘制组合价值
    plot_portfolio(broker)

'''
=================== 总结 ===================
1. 本代码实现了一个简易实盘模拟框架：
   - 通过Excel读取历史股价
   - 计算MA20和RSI指标
   - 按条件买卖股票，模拟资金变化
2. 输出内容包括：
   - 最终现金、最终持仓股数、最终总资金
   - 每日组合总价值曲线
   - 每次交易记录
3. 可以在此基础上扩展：
   - 多股票组合策略
   - 更复杂的买卖信号（MACD、布林线等）
   - 参数优化和风险管理
'''
