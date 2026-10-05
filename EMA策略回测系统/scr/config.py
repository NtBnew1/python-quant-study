"""
配置文件 - 只存固定配置, 股票代码由运行时传入
        - 所有参数集中管理
"""

from pathlib import Path
from datetime import datetime

# ===================================
# 项目路径
# ===================================

PROJECT_IDR = Path(__file__).parent.parent

DATA_RAW_DIR = PROJECT_IDR / 'data' / 'raw'
DATA_PROCESSED_DIR = PROJECT_IDR / 'data' / 'processed'
DATA_SIGNALS_DIR = PROJECT_IDR / 'data' / 'signals'
DATA_SIGNALS_EMA_DIR = PROJECT_IDR / 'data' / 'signals_ema'
DATA_SIGNALS_EMA_WEEKLY_DIR = PROJECT_IDR / 'data' / 'signals_ema_周线'

RESULTS_DIR = PROJECT_IDR / 'results'
CHARTS_DIR = PROJECT_IDR / 'charts'

# ============================
# 时间配置
# ============================
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")  #自动获取当天日期



# ================================
# EMA策略参数
# ================================
EMA_PERIODS = (5, 10, 60)
THRESHOLD = 0.02


# =================================
# 回测参数
# =================================
INITIAL_CASH = 30000
STOP_LOSS = 0.20
POSITION_RATIO = 0.30
COMMISSION = 0.001


# ================================
# 优化参数
# ================================
EMA_COMBINATIONS = [
    (3, 10, 30),
    (5, 10, 60),
    (5, 15, 30),
    (8, 21, 55),
    (10, 20, 60),
]

STOP_LOSS_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
POSITION_RATIO_RANGE = [0.20, 0.30, 0.40, 0.50, 1.00]
THRESHOLD_RANGE = [0.00, 0.01, 0.02, 0.03, 0.05]




# =================================
# 工具函数
# =================================
def create_directories():
    """创建所有必要的目录"""
    for directory in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SIGNALS_DIR,
        DATA_SIGNALS_EMA_DIR,
        DATA_SIGNALS_EMA_WEEKLY_DIR,
        RESULTS_DIR,
        CHARTS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f" 目录已创建: {directory}")

def print_config():
    """打印当前的配置"""
    print(f"=" * 60)
    print(f"当前配置")
    print(f"=" * 60)
    print(f"开始日期: {START_DATE}")
    print(f"结束日期: {END_DATE} (今天)")
    print(f"EMA周期: {EMA_PERIODS}")
    print(f"信号阈值: {THRESHOLD*100}%")
    print(f"初始资金: ${INITIAL_CASH:,}")
    print(f"止损比例: {STOP_LOSS*100}%")
    print(f"仓位比例: {POSITION_RATIO*100}%")
    print(f"手续费: {COMMISSION*100}%")
    print(f"=" * 60)


if __name__ == "__main__":
    # 直接运行config.py 时, 创建所有目录并打印配置
    create_directories()
    print()
    print_config()
    print(f"\n 所有目录创建完成! ")


"""
## 配置文件

**任务目标**：
- 集中管理所有项目配置参数
- 统一管理项目路径
- 自动创建所有必要的目录
- 提供配置打印功能

**实现方案**：
1. **路径配置**：
   - 自动获取项目根目录
   - 定义所有数据目录路径（raw、processed、signals等）
   - 定义结果和图表输出目录

2. **时间配置**：
   - 开始日期固定（2020-01-01）
   - 结束日期自动获取当天

3. **策略参数**：
   - EMA周期组合（5, 10, 60）
   - 信号阈值（2%）
   - 回测参数（初始资金、止损比例、仓位比例、手续费）

4. **优化参数**：
   - 多组EMA组合用于优化
   - 止损范围、仓位比例范围、阈值范围

5. **工具函数**：
   - `create_directories()`：一键创建所有目录
   - `print_config()`：打印当前配置信息

**核心代码结构**：
```python
# 路径配置
DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SIGNALS_DIR, ...

# 时间配置
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# EMA策略参数
EMA_PERIODS = (5, 10, 60)
THRESHOLD = 0.02

# 回测参数
INITIAL_CASH = 30000
STOP_LOSS = 0.20
POSITION_RATIO = 0.30
COMMISSION = 0.001

# 优化参数
EMA_COMBINATIONS = [...]
STOP_LOSS_RANGE = [...]
POSITION_RATIO_RANGE = [...]
THRESHOLD_RANGE = [...]

# 工具函数
create_directories()   # 创建所有目录
print_config()         # 打印配置信息
"""





