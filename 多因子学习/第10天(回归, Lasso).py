'''
第10天：

机器学习辅助因子筛选基础（如岭回归、Lasso）
探索用机器学习模型给因子赋权的方法
练习：用简单回归模型做因子权重拟合
'''

import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 读取数据 ==================
file_path = './Day8-2_factors_and_standardized.xlsx'
df = pd.read_excel(file_path, sheet_name='Standardized_Factors')

# 因子列和目标列
factor_cols = [
    '12m_return', '6m_return', '3m_return',
    'volatility_12m', 'MaxDrawdown',
    'PE', 'PB', 'EV_EBITDA',
    'ROE', 'ROA', 'NetMargin'
]
target = 'future_return_20'

# 添加 Month 列，用于按月滚动
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')

# 定义回归模型
models = {
    'Linear': LinearRegression(),  # 普通线性回归
    'Ridge': Ridge(alpha=1.0),     # 岭回归，防止多重共线性
    'Lasso': Lasso(alpha=0.01)     # Lasso 回归，可做特征选择
}

ROLLING_WINDOW = None  # 可选：使用最近 N 条数据做滚动回归
weights_list = []      # 用于存储每个月各模型因子权重

# ================== 2. 按月份滚动回归计算因子权重 ==================
months = sorted(df['Month'].unique())
for month in months:
    # 训练数据为当月及以前所有数据
    train_data = df[df['Month'] <= month].dropna(subset=factor_cols + [target])
    if ROLLING_WINDOW is not None:
        train_data = train_data.tail(ROLLING_WINDOW)    # 可选：仅取最近 N 条数据
    if train_data.empty:
        continue    # 如果没有数据就跳过

    x_train = train_data[factor_cols].values
    y_train = train_data[target].values

    scaler = StandardScaler()       # 标准化特征，保证每个因子同等量纲
    x_train_scaled = scaler.fit_transform(x_train)

    # 用字典存储当月各模型因子权重
    month_weights = {'Month': str(month)}
    for name, model in models.items():
        try:
            model.fit(x_train_scaled, y_train)  # 拟合模型
            # 将每个因子的系数保存到字典
            month_weights.update({f"{name}_{f}": coef for f, coef in zip(factor_cols, model.coef_)})
        except:
            # 如果模型报错，则对应因子权重设为 None
            month_weights.update({f"{name}_{f}": None for f in factor_cols})
    weights_list.append(month_weights)

weights_df = pd.DataFrame(weights_list)  # 最终存储每月各模型因子权重

# ================== 3. 计算每只股票每日因子得分 ==================
scores_list = []
for month in months:
    month_data = df[df['Month'] == month].dropna(subset=factor_cols + [target])
    if month_data.empty:
        continue

    # 使用 Ridge 回归权重计算当月因子得分
    ridge_weights = weights_df.loc[weights_df['Month']==str(month), [f'Ridge_{f}' for f in factor_cols]]
    if ridge_weights.empty:
        continue
    ridge_weights = ridge_weights.iloc[0].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(month_data[factor_cols].values)

    # 因子得分 = 标准化因子值 dot 权重
    month_data['FactorScore'] = X_scaled.dot(ridge_weights)

    # 选出前20%股票
    top_n = int(len(month_data) * 0.2) if len(month_data) > 5 else len(month_data)
    top_stocks = month_data.nlargest(top_n, 'FactorScore')

    # 保存结果
    scores_list.append(top_stocks[['Date', 'Month', 'company', 'FactorScore']])

scores_df = pd.concat(scores_list, ignore_index=True)

# ================== 4. 打印最近数据 ==================
# 整年 2025 年各公司平均得分
recent_avg = scores_df[scores_df['Month'].dt.year == 2025].groupby('company')['FactorScore'].mean().sort_values(ascending=False)
print('2025 年各公司平均 FactorScore (按高到低)')
print(recent_avg)

# 最近月份前20%股票
last_month = scores_df['Month'].max()
recent_top = scores_df[scores_df['Month'] == last_month]
print(f"{last_month} 前20%股票")
print(recent_top)

# 2025-07 月各公司平均因子得分
last_month = '2025-07'
top_july = scores_df[scores_df['Month'] == last_month]
avg_score_july = top_july.groupby('company')['FactorScore'].mean().sort_values(ascending=False)
print(f"{last_month} 月各公司平均 FactorScore: ")
print(avg_score_july)

# ================== 5. 生成月度统计 ==================
monthly_summary = []
for month in months:
    month_data = scores_df[scores_df['Month'] == month]
    if month_data.empty:
        continue
    # 统计每只股票当月出现天数和平均因子得分
    summary = month_data.groupby('company').agg(
        AppearDays=('Date', 'count'),
        AvgFactorScore=('FactorScore', 'mean')
    ).reset_index()
    summary['Month'] = str(month)
    monthly_summary.append(summary)

monthly_summary_df = pd.concat(monthly_summary, ignore_index=True)

# ================== 6. 保存 Excel ==================
output_file = './Day10_results_with_monthly_summary.xlsx'
with pd.ExcelWriter(output_file) as writer:
    # 每个模型权重
    for name in models.keys():
        model_cols = [c for c in weights_df.columns if c.startswith(name)]
        sheet_df = weights_df[['Month'] + model_cols]
        sheet_df.to_excel(writer, sheet_name=name+'_Weights', index=False)
    # 每日因子得分前20%股票
    top_stock_df = pd.concat(scores_list, ignore_index=True)
    top_stock_df.to_excel(writer, sheet_name='TopStock', index=False)
    # 月度统计
    monthly_summary_df.to_excel(writer, sheet_name='MonthlySummary', index=False)

print(f"结果已保存到 {output_file}")

# ================== 7. 绘制 2025 年热力图 ==================
scores_2025 = scores_df[scores_df['Month'].dt.year == 2025]

if scores_2025.empty:
    print("2025 年没有数据, 无法绘制热力图")
else:
    # 转换日期格式便于显示
    scores_2025['DateStr'] = scores_2025['Date'].dt.strftime('%Y-%m-%d')
    heatmap_data = scores_2025.pivot(index='company', columns='DateStr', values='FactorScore')
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0)
    plt.title('FactorScore Heatmap - 2025')
    plt.xlabel('Date')
    plt.ylabel('Company')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ================== 8. 可视化因子权重随时间变化 ==================
# 8.1 Ridge折线图
plt.figure(figsize=(14, 6))
for f in factor_cols:
    plt.plot(weights_df['Month'], weights_df['Ridge_' + f], label=f)
plt.xticks(rotation=45)
plt.title('Monthly Factor Weights (Ridge Regression)')
plt.ylabel('Coefficient')
plt.xlabel('Month')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()

# 8.2 Ridge热力图
plt.figure(figsize=(14, 6))
ridge_cols = ['Ridge_' + f for f in factor_cols]
heat_data = weights_df[ridge_cols].T
heat_data.columns = weights_df['Month']
sns.heatmap(heat_data, cmap='RdBu_r', center=0, annot=False)
plt.title("Heatmap of Ridge Regression Factor Weights Over Time")
plt.xlabel('Month')
plt.ylabel('Factors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


'''
## 第10天：机器学习辅助因子筛选总结

### 1. 学习目标
- 使用简单回归、Ridge 和 Lasso 模型对多因子进行权重拟合。
- 探索每只股票因子得分的计算方法。
- 按月份滚动回归，观察因子权重随时间变化。

### 2. 数据处理
- 数据来源：`Day8-2_factors_and_standardized.xlsx`。
- 选取的因子包括收益率、波动率、估值指标和财务指标等 11 个因子。
- 添加 `Month` 列，用于按月份计算因子权重。

### 3. 模型与计算
- 回归模型：LinearRegression、Ridge(alpha=1.0)、Lasso(alpha=0.01)。
- 按月份滚动回归计算因子权重。
- 使用 Ridge 回归权重计算每只股票每日因子得分 (`FactorScore`)。
- 选出每月前 20% 的股票，作为因子表现最好的股票。

### 4. 输出内容
- **年度平均因子得分**：2025 年各公司平均 `FactorScore` 排序。
- **月度前20%股票**：2025-07 月度前20%股票及其得分。
- **月度统计**：每只股票当月出现天数及平均因子得分。
- **Excel 保存**：
  - 各模型每月因子权重 (`Linear_Weights`, `Ridge_Weights`, `Lasso_Weights`)  
  - 每日因子得分前20%股票 (`TopStock`)  
  - 月度统计汇总 (`MonthlySummary`)

### 5. 可视化
- **2025 年因子得分热力图**：显示每只股票每日因子得分变化。
- **Ridge 因子权重折线图**：观察因子权重随月份变化趋势。
- **Ridge 因子权重热力图**：清晰展示各因子在不同月份的重要性。

### 6. 总结
- Ridge 回归权重相对平稳，不易受极端值影响。
- 因子得分高的股票在特定月份更容易被识别，方便后续量化策略使用。
- 可视化热力图帮助快速识别因子表现和股票优劣趋势。

'''