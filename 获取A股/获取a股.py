"""
A股数据下载器
功能：下载任意A股历史数据，保存为Excel文件

核心功能：
1. 下载A股历史数据（使用akshare库）
2. 支持单只/多只股票下载
3. 保存为Excel格式
4. 友好的用户交互界面
"""




import akshare as ak
import pandas as pd
import os
import time

class AStockDownloader:
    """A股数据下载器"""

    def __init__(self):
        # 创建保存文件夹
        self.save_folder = "data_stock"
        os.makedirs(self.save_folder, exist_ok=True)

    def show_welcome(self):
        """显示欢迎信息"""
        print("=" * 50)
        print("      A股历史数据下载器")
        print("=" * 50)
        print("功能说明：")
        print("1. 下载任意A股历史数据")
        print("2. 保存为Excel格式")
        print("3. 支持批量下载")
        print("=" * 50)
        print(f"数据将保存到: {self.save_folder}/ 文件夹")
        print("=" * 50)

    def get_user_input(self):
        """获取用户输入的股票代码"""
        print("\n📝 输入股票代码")
        print("-" * 30)
        print("支持格式：")
        print("• 单只股票：000001")
        print("• 多只股票：000001 600519 000858")
        print("• 逗号分隔：000001,600519,000858")
        print("-" * 30)

        while True:
            codes_input = input("\n请输入股票代码（输入q退出）: ").strip()

            if codes_input.lower() in ['q', 'quit', 'exit']:
                return None

            if not codes_input:
                print("⚠ 请输入股票代码")
                continue

            # 解析股票代码
            codes = self.parse_codes(codes_input)

            if codes:
                print(f"✅ 识别到 {len(codes)} 只股票：")
                for i, code in enumerate(codes, 1):
                    print(f"  {i}. {code}")
                return codes
            else:
                print("⚠ 未识别到有效股票代码")

    def parse_codes(self, input_str):
        """解析股票代码输入"""
        # 替换所有分隔符为空格
        for sep in [',', '，', ';', '；', '、']:
            input_str = input_str.replace(sep, ' ')

        codes = []
        for code in input_str.split():
            code = code.strip()
            if code and self.is_valid_stock_code(code):
                codes.append(code)

        # 去重
        return list(set(codes))

    def is_valid_stock_code(self, code):
        """验证股票代码格式"""
        # 必须是6位数字
        if len(code) != 6 or not code.isdigit():
            return False

        # 检查交易所前缀
        first_char = code[0]
        valid_prefixes = ['0', '3', '6', '4', '8', '2']  # A股代码前缀

        return first_char in valid_prefixes

    def download_stock_data(self, code, start_date="20240101"):
        """下载单只股票数据"""
        try:
            # 获取当前日期
            end_date = time.strftime("%Y%m%d")

            # 下载数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )

            return df, None  # 返回数据和空错误信息

        except Exception as e:
            return None, str(e)  # 返回错误信息

    def save_to_excel(self, df, code):
        """保存数据到Excel"""
        filename = f"{code}_data_stock.xlsx"
        filepath = os.path.join(self.save_folder, filename)

        # 保存为Excel
        df.to_excel(filepath, index=False)

        return filename

    def batch_download(self, codes):
        """批量下载股票数据"""
        print(f"\n⏳ 开始下载 {len(codes)} 只股票数据...")
        print("-" * 40)

        results = {
            'success': [],
            'failed': []
        }

        # 设置时间范围（2024年至今）
        start_date = "20240101"

        for i, code in enumerate(codes, 1):
            print(f"\n[{i}/{len(codes)}] 正在下载 {code}...")

            # 下载数据
            df, error = self.download_stock_data(code, start_date)

            if error:
                print(f"  ❌ 下载失败: {error}")
                results['failed'].append({'code': code, 'error': error})
                continue

            if df.empty:
                print(f"  ⚠ 无数据")
                results['failed'].append({'code': code, 'error': '无数据'})
                continue

            # 保存数据
            filename = self.save_to_excel(df, code)

            print(f"  ✅ 成功下载 {len(df)} 条数据")
            print(f"     保存为: {filename}")

            results['success'].append({
                'code': code,
                'records': len(df),
                'filename': filename
            })

            # 添加短暂延迟，避免请求过快
            if i < len(codes):
                time.sleep(0.5)

        return results

    def show_results(self, results, total_count):
        """显示下载结果"""
        print("\n" + "=" * 50)
        print("📊 下载结果")
        print("=" * 50)

        success_count = len(results['success'])
        failed_count = len(results['failed'])

        print(f"\n统计信息：")
        print(f"  总计: {total_count} 只")
        print(f"  成功: {success_count} 只")
        print(f"  失败: {failed_count} 只")

        if success_count > 0:
            total_records = sum(item['records'] for item in results['success'])
            print(f"  总数据量: {total_records} 条")

            print(f"\n📁 文件保存在: {os.path.abspath(self.save_folder)}/")

            print(f"\n✅ 成功下载的股票：")
            for item in results['success'][:5]:  # 只显示前5个
                print(f"  • {item['code']}: {item['records']}条数据 -> {item['filename']}")

            if success_count > 5:
                print(f"  ... 等共 {success_count} 只股票")

        if failed_count > 0:
            print(f"\n❌ 下载失败的股票：")
            for item in results['failed'][:5]:  # 只显示前5个
                print(f"  • {item['code']}: {item['error']}")

            if failed_count > 5:
                print(f"  ... 等共 {failed_count} 只股票")

        print("\n" + "=" * 50)

    def run(self):
        """运行主程序"""
        self.show_welcome()

        while True:
            try:
                # 1. 获取股票代码
                codes = self.get_user_input()

                if codes is None:
                    print("\n👋 感谢使用！")
                    break

                # 2. 确认下载
                print(f"\n📋 准备下载 {len(codes)} 只股票")
                confirm = input("开始下载？(y/n): ").strip().lower()

                if confirm not in ['y', 'yes', '是']:
                    print("取消下载")
                    continue

                # 3. 批量下载
                results = self.batch_download(codes)

                # 4. 显示结果
                self.show_results(results, len(codes))

                # 5. 是否继续
                continue_download = input("\n继续下载其他股票？(y/n): ").strip().lower()
                if continue_download not in ['y', 'yes', '是']:
                    print("\n👋 感谢使用！")
                    break

                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n\n程序中断")
                break
            except Exception as e:
                print(f"\n程序错误: {e}")
                retry = input("是否继续？(y/n): ").strip().lower()
                if retry not in ['y', 'yes', '是']:
                    break

# 主程序
if __name__ == "__main__":
    # 检查依赖
    try:
        import akshare
        import pandas
        print("✅ 依赖检查通过")
    except ImportError:
        print("❌ 缺少依赖库")
        print("请运行: pip install akshare pandas openpyxl")
        exit(1)

    # 运行下载器
    downloader = AStockDownloader()
    downloader.run()