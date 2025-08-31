import random

cash = 10000  # 初始资金
stock_price = 100  # 初始股票价格
stocks_owned = 0

print("欢迎来到Python炒股小游戏！你有$10,000资金，初始股价$100")

for day in range(1, 11):
    stock_price += random.randint(-10, 10)  # 模拟价格波动
    print(f"\n第{day}天，当前股价：${stock_price}，你持有：{stocks_owned} 股，现金：${cash}")

    action = input("请输入操作：买入（buy），卖出（sell），跳过（pass）: ").strip().lower()

    if action == "buy":
        amount = int(input("买入多少股？"))
        cost = amount * stock_price
        if cost <= cash:
            cash -= cost
            stocks_owned += amount
        else:
            print("资金不足，买入失败。")
    elif action == "sell":
        amount = int(input("卖出多少股？"))
        if amount <= stocks_owned:
            cash += amount * stock_price
            stocks_owned -= amount
        else:
            print("你没有这么多股票。")
    elif action == "pass":
        print("跳过这一天。")
    else:
        print("无效操作，跳过。")

final_value = cash + stocks_owned * stock_price
print(f"\n游戏结束！你最终资产为：${final_value}")
