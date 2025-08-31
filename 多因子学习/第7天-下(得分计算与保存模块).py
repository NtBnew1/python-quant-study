"""
Day 7-下：多因子得分计算与标准化模块

📌 本模块的目标是把“计算得分”的过程单独拿出来，
以后每次做任务时就不需要重复计算了，节省时间，也更规范。

✅ 步骤说明：
1. 每次导入最新的多因子数据（CSV 或 Excel 格式）
2. 对各个因子进行 Z-score 标准化
3. 对于负向因子（例如 pe、pb、volatility 等），取负号统一方向
   - 越小越好的因子，统一成“得分越高越好”
4. 把所有因子的标准化得分加总，形成综合得分
5. 按照得分从高到低排序，选出前 5 家公司
6. 将选出的公司和得分保存为 CSV 文件，供后续回测直接使用
"""

# === 导入所需的库 ===
import pandas as pd                     # 用于数据处理
from scipy.stats import zscore         # 用于 Z-score 标准化

# === 1. 读取多因子数据 ===
file_path = './Day4_factor_all_stocks.xlsx' # Excel 文件路径
df = pd.read_excel(file_path)          # 读取 Excel 文件为 DataFrame

# === 2. 设置因子列表和需要反向处理的因子 ===
factor_columns = ['PE', 'PB', 'EV_EBITDA', '12m_return', '6m_return', '3m_return',
                  'ROE', 'ROA', 'NetMargin', 'Volatility', 'MaxDrawdown']
# 这是所有用来打分的因子列名

reverse_factor = ['PE', 'PB', 'EV_EBITDA', 'Volatility', 'MaxDrawdown']
# 这些因子是“越小越好”的，需要取负号反转方向，使得“高分 = 好”

# === 3. 提取因子数据子集 ===
factor_data = df[factor_columns].copy()  # 只保留因子列，复制出一个新的 DataFrame

# === 4. 对每个因子做 Z-score 标准化 ===
factor_zscore = factor_data.apply(zscore)  # 每一列做 z-score，使不同量纲的因子可以比较

# === 5. 对反向因子取负，使得所有因子方向一致（越大越好）===
for f in reverse_factor:
    if f in factor_zscore.columns:        # 先检查该因子是否存在，避免报错
        factor_zscore[f] = -factor_zscore[f]  # 对于越小越好的因子，取负

# === 6. 计算综合得分：所有因子的标准化值加总 ===
df['综合得分'] = factor_zscore.sum(axis=1)  # 横向求和，得到每家公司的综合打分

# === 7. 排序选股：选出综合得分最高的前 5 家公司 ===
top_n = 5
selected = df.sort_values(by='综合得分', ascending=False).head(top_n)

# === 8. 保存选股结果到 Excel 文件 ===
selected[['Stock', '综合得分']].to_excel('./Day7-下_selected_top5.xlsx', index=False)
print(f"\n已保存成功")  # 提示保存成功

# === 输出所有公司和综合得分（从高到低）===
print(f"\n所有公司和综合得分: ")
print(df[['Stock', '综合得分']].sort_values(by='综合得分', ascending=False))

# === 只输出前 5 家公司 ===
print(f" \n前5家公司:")
print(selected[['Stock', '综合得分']])

'''
📌 提示：
- 如果你想扩展选股范围，可以下载更多公司数据（例如用 Alpha Vantage）。
- 本脚本只需运行一次，之后的回测策略可以直接使用 saved_top5.xlsx。
- 如果你熟悉 Python 模块化，也可以将此逻辑写成函数或打包成模块调用。
'''
