'''用pandas-datareader 获取股票数据. '''



from pandas_datareader import data as pdr
from datetime import datetime

def fetch_multiple_stock_to_excel():
    tickers_input = input(f"请输入股票代码: ").strip()
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    start_date = input(f"请输入开始日期 (YYYY-MM-DD): ").strip()
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"结束日期自动设置为: {end_date}")


    # 为每个股票代码获取数据保存
    successful_tickers = []
    failed_tickers = []

    for ticker in tickers:
        try:
            print(f"正在获取{ticker}数据.....", end=" ")
            stock_data = pdr.DataReader(ticker, 'stooq', start=start_date, end=end_date)

            if not stock_data.empty:
                filename = f"./{ticker}_stock_data.xlsx"
                stock_data.to_excel(filename, engine='openpyxl')
                successful_tickers.append(ticker)
                print(f"获取{ticker}数据成功 --> 保存到{filename}")

            else:
                failed_tickers.append(ticker)
                print(f"获取{ticker}数据失败")

        except Exception as e:
            failed_tickers.append(ticker)
            print(f"失败:{e}")

    print(f"数据获取完成...")
    print(f"成功获取: {len(successful_tickers)}只股票")
    print(f"获取失败: {len(failed_tickers)}只股票")

    if successful_tickers:
        print(f"已保存到Excel文件")
        for ticker in successful_tickers:
            print(f"-{ticker}_stock_data.xlsx")

    if failed_tickers:
        print(f"失败的股票: {failed_tickers}" )

if __name__ == "__main__":
    fetch_multiple_stock_to_excel()














