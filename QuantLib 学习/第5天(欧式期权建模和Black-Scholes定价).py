'''
Day 5：期权基础
目标：
-学习欧式期权建模和Black-Scholes定价。
-计算期权价格及希腊值（Greeks）。
任务：
-创建欧式看涨/看跌期权对象。
-使用不同波动率、到期时间测试价格变化。
输出：期权定价及敏感性分析脚本。
'''

# 期权大家应该清楚吧.  如果不清楚可以去youtube关注 "阳光财经".  她有一个视频时实盘讲解期权的.

import pandas as pd
import numpy as np
import datetime as dt
from yahooquery import Ticker
from math import log, sqrt, exp
from scipy.stats import norm

# =========Black-Scholes 定价函数===========
'''black-scholes是用于定价欧式期权的数学模型'''
'''
S = 股票价格
K = 执行价
T = 到期时间(年)
r = 无风险利率
sigma = 波动率
option_type = 看涨/看跌
'''
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (log(S/K) + (r + 0.5*sigma**2)* T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type == "call":
        price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else: # put
        price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

# 获取期权数据
symbol = 'AAPL'
ticker = Ticker(symbol)

opt_chain = ticker.option_chain # 获取期权链
print(opt_chain.head())

today = dt.datetime.today() # 定义今天.

# =====选择一个合约=======
future_options = opt_chain[opt_chain.index.get_level_values('expiration') > (today + pd.Timedelta(days=180))]
first_row = future_options.iloc[0]   # 去明年第一个期权
idx = first_row.name
expiration = pd.to_datetime(idx[1]) # 到期日
option_type = idx[2]    # 看涨 或 看跌


# ======获取市场参数===========
S0 = ticker.history(period='1d')['close'].iloc[-1]  # 标的现价
K = first_row['strike']     # 执行价
T = (expiration - today).days /365  # 距离到期年数

#   r 无风险利率, 不用自定义.  用真实数据
import pandas_datareader.data as web
start = today - dt.timedelta(days=30)   # 最近30天的数据
# 获取3个月美国国债利率 (DGS3MO)
df_r = web.DataReader("DGS3MO", 'fred', start, today)
r = df_r.iloc[-1, 0] / 100      # 转成小数
print(f"真实无风险利率: r = {r:.4f}")

hist = ticker.history(period='6mo')['close']
sigma = np.log(hist/hist.shift(1)).std() * np.sqrt(252)     # 历史波动率

# ==============计算Black_Scholes 价格============
bs_price = black_scholes(S0, K, T, r, sigma, option_type='call' if option_type=='calls' else 'put')
print(f"Black_Scholes 模型价格: {bs_price:.4f}({option_type})")
print(f"市场价格: {first_row['lastPrice']}")


# ==========计算Greeks(敏感性)==================
d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

Delta = norm.cdf(d1) if option_type=="calls" else norm.cdf(d1) -1
Gamma = norm.pdf(d1) / (S0*sigma*sqrt(T))
Vega = S0 * norm.pdf(d1) * sqrt(T) / 100
Theta = (-S0*norm.pdf(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)) / 365 if option_type=='calls' else ...
Rho = K*T*exp(-r*T)*norm.cdf(d2)/100 if option_type=='calls' else ...

# =======敏感性分析 ( 波动率 和 到期时间)
print(f"=====波动率敏感性分析 (Vega)=====")
for vol in [0.1, 0.2, 0.3, 0.5]:
    price = black_scholes(S0, K, T, r, vol, option_type)
    print(f"波动率{vol*100:.0f}% -> 价格: {price:.4f}")

print(f"=====到期时间敏感性分析 (Theta)===== ")
for t_year in [0.25, 0.5, 1, 2]:
    price = black_scholes(S0, K, t_year, r, sigma, option_type)
    print(f"到期: {t_year:.2f}年 -> 价格: {price:.4f}")


'''             我们用明天的来计算吧. '''



'''
AAPL250926C00110000
AAPl - 股票代码
250926 - 到期日
C - 看涨, 如果是P 是看跌
00110000 - 行权价  $ 110.00 美元

Black_Scholes 模型价格: 135.5486(calls)
市场价格: 130.1
135 是我们计算出来的.  市场是130.  看涨.  有可能是波动率或者其它的一些原因吧. 

=====波动率敏感性分析 (Vega)=====
波动率10% -> 价格: 0.0000
波动率20% -> 价格: 0.0000
波动率30% -> 价格: 0.0000
波动率50% -> 价格: 0.0000
我们就算1天, 所以波动率没用.       这个应该是计算问题. 


=====到期时间敏感性分析 (Theta)===== 
到期: 0.25年 -> 价格: 0.0001
到期: 0.50年 -> 价格: 0.0165
到期: 1.00年 -> 价格: 0.3196
到期: 2.00年 -> 价格: 1.8921
期权到期时间越长, 价格越高. 

'''

