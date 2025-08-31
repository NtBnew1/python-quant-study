'''
第10天：

机器学习辅助因子筛选基础（如岭回归、Lasso）
探索用机器学习模型给因子赋权的方法
练习：用简单回归模型做因子权重拟合

'''
# ================== 机器学习辅助多因子权重计算 ==================
# 目标：通过机器学习方法（线性回归、Ridge、Lasso）计算多因子在预测未来收益率中的权重
# 并观察这些权重随时间的变化，从而辅助构建多因子投资组合。
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 读取标准化因子数据 ==================
file_path = './Day8-2_factors_and_standardized.xlsx'
df = pd.read_excel(file_path, sheet_name='Standardized_Factors')
print("DataFrame 的列名：", df.columns.tolist())

# 这里读取Excel文件中的数据，sheet里存放了标准化后的各类因子值
# 为什么标准化？因为不同因子量纲不同，比如PE、ROE量级不同，直接回归会影响系数大小
# 标准化后，所有因子均值为0，标准差为1，回归系数可以直接比较因子重要性

# 定义因子列和目标列
factor_cols = [
    '12m_return', '6m_return', '3m_return',  # 历史收益类因子
    'volatility_12m', 'MaxDrawdown',  # 风险类因子
    'PE', 'PB', 'EV_EBITDA',  # 估值类因子
    'ROE', 'ROA', 'NetMargin'  # 盈利能力类因子
]
target = 'future_return_20'  # 未来20天收益率作为回归目标

# 添加月份列, 方便按月滚动回归
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
# 把Date转换为月份周期（Period），方便做按月滚动回归
# 这样每个月可以用当月及之前的数据做训练，观察系数变化趋势

# ================== 2. 准备回归模型 ==================
models = {
    'Liner': LinearRegression(),  # 普通最小二乘回归
    'Ridge': Ridge(alpha=1.0),  # 岭回归，增加L2正则化，防止多重共线性导致系数不稳定
    'Lasso': Lasso(alpha=0.01)  # Lasso回归，L1正则化，可稀疏化系数，有助于特征选择
}

# 滚动窗口参数: 只使用最近N个月的数据训练模型
ROLLING_WINDOW = None
# 如果设置成整数N，就只取最近N个月的数据训练
# 如果为None，则使用从数据开始到当前月份的全部历史数据

# 用于存储每个月因子权重
weights_list = []

# ================== 3. 按月份滚动回归 ==================
months = sorted(df['Month'].unique())  # 获取所有月份并排序
for month in months:
    # 训练集: 当月及之前的数据
    train_data = df[df['Month'] <= month].dropna(subset=factor_cols + [target])
    # dropna保证没有缺失值，否则回归会报错
    # subset=factor_cols + [target] 表示只检查因子和目标列的缺失值

    # 滚动窗口: 只保留最近 ROLLING_WINDOW 个月的数据
    if ROLLING_WINDOW is not None:
        train_data = train_data.tail(ROLLING_WINDOW)
        # tail(N)保留最近N行数据，相当于最近N个月的数据

    if train_data.empty:
        continue  # 如果训练集为空，直接跳过

    x_train = train_data[factor_cols].values  # 因子矩阵
    y_train = train_data[target].values  # 目标向量

    # 检查是否还有有效数据
    if x_train.shape[0] == 0:
        continue

    # 标准化因子 (均值0, 标准差1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # 标准化后，系数更可比，不同因子量纲不同会导致系数大小不合理
    # 例如PE是两位数，ROE可能是百分数，直接回归系数会被PE“掩盖”

    # 整合每个模型, 并记录系数
    month_weights = {'Month': str(month)}  # 初始化本月权重字典，记录月份
    for name, model in models.items():
        try:
            model.fit(x_train_scaled, y_train)  # 训练模型
            # zip(factor_cols, model.coef_) 是把因子名称和对应回归系数对应起来
            month_weights.update({f"{name}_{f}": coef for f, coef in zip(factor_cols, model.coef_)})
        except Exception as e:
            print(f"Warning: {name} model failed for month {month} -> {e}")
            # 如果训练失败（可能是样本太少或矩阵奇异），就用None填充
            month_weights.update({f"{name}_{f}": None for f in factor_cols})

    weights_list.append(month_weights)  # 把本月权重加入列表

# ================== 4. 转为DataFrame ==================
weights_df = pd.DataFrame(weights_list)
print("滚动回归每月因子权重(前5行): \n", weights_df.head())
# 转为DataFrame方便查看和处理，列名格式如 Ridge_12m_return
# 可以用来分析各因子随时间变化趋势

# ================== 5. 可视化因子权重随时间变化 ==================
# 5.1 Ridge回归折线图
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


# 5.2 Ridge回归热力图
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
# 热力图能快速看出哪些因子权重一直为正、负或波动较大


# ================== 7. 计算每家公司每个月的多因子得分 ==================
scores_list = []

for month in months:
    # 当月数据
    month_data = df[df['Month'] == month].dropna(subset=factor_cols + [target])
    if month_data.empty:
        continue

    # 获取当月 Ridge 权重
    ridge_weights = weights_df.loc[weights_df['Month'] == str(month), [f"Ridge_{f}" for f in factor_cols]]
    if ridge_weights.empty:
        continue
    ridge_weights = ridge_weights.iloc[0].values  # 取系数值

    # 标准化因子值（和训练时保持一致）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(month_data[factor_cols].values)

    # 计算多因子得分 = X * 权重
    month_data['FactorScore'] = X_scaled.dot(ridge_weights)

    # 按得分排序，取前20%
    top_n = int(len(month_data) * 0.2) if len(month_data) > 5 else len(month_data)
    top_stocks = month_data.nlargest(top_n, 'FactorScore')

    # 保存结果
    scores_list.append(top_stocks[['Date', 'Month', 'company', 'FactorScore']])

# 合并结果
scores_df = pd.concat(scores_list, ignore_index=True)

# 输出结果
print("每个月多因子得分最高的股票（前20%）：\n", scores_df.head(20))

# 最后统一保存到一个 Excel，不同 Sheet
output_file = './Day10_results.xlsx'
with pd.ExcelWriter(output_file) as writer:
    # 保存每个模型权重
    for name in models.keys():
        model_cols = [c for c in weights_df.columns if c.startswith(name)]
        sheet_df = weights_df[['Month'] + model_cols]
        sheet_df.to_excel(writer, sheet_name=name + '_Weights', index=False)

    # 保存前20%股票
    top_stocks_df = pd.concat(scores_list, ignore_index=True)
    top_stocks_df.to_excel(writer, sheet_name='TopStocks', index=False)

print(f"\n所有结果已保存到 {output_file}，不同数据在不同Sheet里")

'''
# ===================== 总结 ==========================
1. 本代码按月滚动训练三个回归模型，计算每个因子在预测未来收益率的权重
2. 使用标准化确保不同量纲因子系数可比
3. 可视化折线图和热力图帮助理解因子稳定性和重要性
4. 保存到Excel方便长期跟踪和策略优化
这种方法能辅助构建多因子投资组合，同时观察因子权重随时间的变化趋势



===================绘制图解读================
1️⃣ 横轴和纵轴
-横轴（Month）：每个月的时间，从 2021-08 到 2025-08。
-纵轴（Coefficient）：回归系数，也就是每个因子的权重。
-解释：权重越大，说明这个因子对预测未来收益率的影响越大；权重为负，说明这个因子对未来收益是负相关。

2️⃣ 每条线的含义
-每条彩色折线对应一个因子：
    比如 12m_return、6m_return、ROE 等。
-折线上的点表示每个月 Ridge 回归算出的该因子权重。
-越高越重要，越低越不重要或反向影响。

3️⃣ 权重变化趋势
-稳定的因子：
    -如 12m_return、6m_return 等历史收益类因子，权重一直为正，波动不大 → 对未来收益预测较稳定。
-波动较大的因子：
    -如 3m_return、ROA 等，权重在早期波动大 → 表明初期数据不稳定或因子对短期收益敏感。
-负权重因子：
    -比如 NetMargin 或 ROA，权重为负 → 说明它们对未来收益可能有抑制作用。

4️⃣ 为什么会有波动
-Ridge 回归会平滑系数，但因数据是滚动回归（每个月重新训练一次），所以：
    -样本数据变化：新数据加入，旧数据移出（如果滚动窗口有限）。
    -因子关系变化：市场行情、行业表现不同，因子贡献也不同。
-总结：这就是因子权重随时间变化的“动态多因子权重”。
'''


