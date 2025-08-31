import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 读取数据 ==================
file_path = './Day8-2_factors_and_standardized.xlsx'
df = pd.read_excel(file_path, sheet_name='Standardized_Factors')

# 定义因子列和目标列
factor_cols = [
    '12m_return', '6m_return', '3m_return',
    'volatility_12m', 'MaxDrawdown',
    'PE', 'PB', 'EV_EBITDA',
    'ROE', 'ROA', 'NetMargin'
]
target = 'future_return_20'

# 添加 Month 列，类型为 Period，方便按月滚动
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')

# 定义回归模型
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01)
}

ROLLING_WINDOW = None  # 可设置只用最近 N 个月数据训练
weights_list = []

# ================== 2. 按月份滚动回归计算因子权重 ==================
months = sorted(df['Month'].unique())
for month in months:
    # 取当前月份及以前的数据训练
    train_data = df[df['Month'] <= month].dropna(subset=factor_cols + [target])

    if ROLLING_WINDOW is not None:
        train_data = train_data.tail(ROLLING_WINDOW)  # 只保留最近 N 条数据

    if train_data.empty:
        continue

    x_train = train_data[factor_cols].values
    y_train = train_data[target].values
    if x_train.shape[0] == 0:
        continue

    # 标准化因子值，保证系数可比
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # 存储每个月各模型因子权重
    month_weights = {'Month': str(month)}
    for name, model in models.items():
        try:
            model.fit(x_train_scaled, y_train)
            month_weights.update({f"{name}_{f}": coef for f, coef in zip(factor_cols, model.coef_)})
        except:
            month_weights.update({f"{name}_{f}": None for f in factor_cols})
    weights_list.append(month_weights)

weights_df = pd.DataFrame(weights_list)

# ================== 3. 计算每只股票每日因子得分 ==================
scores_list = []

for month in months:
    month_data = df[df['Month'] == month].dropna(subset=factor_cols + [target])
    if month_data.empty:
        continue

    # 使用 Ridge 回归权重计算多因子得分
    ridge_weights = weights_df.loc[weights_df['Month'] == str(month), [f'Ridge_{f}' for f in factor_cols]]
    if ridge_weights.empty:
        continue
    ridge_weights = ridge_weights.iloc[0].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(month_data[factor_cols].values)

    # 计算 FactorScore
    month_data['FactorScore'] = X_scaled.dot(ridge_weights)

    # 取前20%股票
    top_n = int(len(month_data) * 0.2) if len(month_data) > 5 else len(month_data)
    top_stocks = month_data.nlargest(top_n, 'FactorScore')
    scores_list.append(top_stocks[['Date', 'Month', 'company', 'FactorScore']])

scores_df = pd.concat(scores_list, ignore_index=True)

# ================== 4. 打印最近数据 ==================
# 2025 年各公司平均 FactorScore
recent_avg = scores_df[scores_df['Month'].dt.year == 2025].groupby('company')['FactorScore'].mean().sort_values(
    ascending=False)
print("2025 年各公司平均 FactorScore（按高到低）：")
print(recent_avg)

# 最近月份前20%股票
last_month = scores_df['Month'].max()
recent_top = scores_df[scores_df['Month'] == last_month]
print(f"{last_month} 前20%股票：")
print(recent_top)

# 2025-07 月各公司平均 FactorScore
last_month = '2025-07'
top_july = scores_df[scores_df['Month'] == last_month]
avg_score_july = top_july.groupby('company')['FactorScore'].mean().sort_values(ascending=False)
print(f"{last_month} 月各公司平均 FactorScore：")
print(avg_score_july)

# ================== 5. 生成月度统计 ==================
monthly_summary = []
for month in months:
    month_data = scores_df[scores_df['Month'] == month]
    if month_data.empty:
        continue
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
    for name in models.keys():
        model_cols = [c for c in weights_df.columns if c.startswith(name)]
        sheet_df = weights_df[['Month'] + model_cols]
        sheet_df.to_excel(writer, sheet_name=name + '_Weights', index=False)
    top_stocks_df = pd.concat(scores_list, ignore_index=True)
    top_stocks_df.to_excel(writer, sheet_name='TopStocks', index=False)
    monthly_summary_df.to_excel(writer, sheet_name='MonthlySummary', index=False)

print(f"结果已保存到 {output_file}")

# ================== 7. 绘制 2025 年热力图 ==================
scores_2025 = scores_df[scores_df['Month'].dt.year == 2025]

if scores_2025.empty:
    print("2025 年没有数据，无法绘制热力图")
else:
    # 将日期转换为字符串，方便在热力图上显示
    scores_2025['DateStr'] = scores_2025['Date'].dt.strftime('%Y-%m-%d')

    # pivot：行是公司，列是日期，值是 FactorScore
    heatmap_data = scores_2025.pivot(index='company', columns='DateStr', values='FactorScore')

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0)
    plt.title('FactorScore Heatmap - 2025')
    plt.xlabel('Date')
    plt.ylabel('Company')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ================== 8. 可视化因子权重随时间变化 ==================
# 8.1 Ridge回归折线图
plt.figure(figsize=(14, 6))
for f in factor_cols:
    plt.plot(weights_df['Month'], weights_df['Ridge_' + f], label=f)
plt.xticks(rotation=45)
plt.title('Monthly Factor Weights (Ridge Regression)')
plt.ylabel('Coefficient')
plt.xlabel("Month")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()
# 折线图显示每个因子权重随时间的变化
# Ridge回归系数一般比较平稳，不易受极端值影响

# 8.2 Ridge回归热力图
plt.figure(figsize=(12, 6))
ridge_cols = ['Ridge_' + f for f in factor_cols]
heat_data = weights_df[ridge_cols].T  # 转置，使因子为行，月份为列
heat_data.columns = weights_df['Month']
sns.heatmap(heat_data, cmap='RdBu_r', center=0, annot=False)
plt.title('Heatmap of Ridge Regression Factor Weights Over Time')
plt.xlabel('Month')
plt.ylabel('Factors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
