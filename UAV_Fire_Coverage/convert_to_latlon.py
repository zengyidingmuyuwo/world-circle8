import pandas as pd
from pyproj import Transformer
import os


def convert_and_overwrite(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    # 1. 读取原CSV文件
    df = pd.read_csv(csv_path)

    # 记录原始的表头列名（为了防止覆盖后破坏你其他代码的读取，比如 'circle id' 中间的空格我们要保留）
    original_columns = df.columns.tolist()

    # 临时把列名里的空格替换成下划线，方便下面提取数据
    temp_columns = [col.strip().replace(' ', '_') for col in original_columns]
    df.columns = temp_columns

    # 2. 定义坐标转换器 (从 UTM投影 转换为 WGS84 标准经纬度)
    # always_xy=True 保证顺序为 (X, Y) -> (经度, 纬度)
    transformer = Transformer.from_crs("epsg:32647", "epsg:4326", always_xy=True)

    # 3. 执行转换并直接覆盖原列的值
    # df['center_x'] 会变成 100.xxxx (经度 Longitude)
    # df['center_y'] 会变成 22.xxxx (纬度 Latitude)
    lon, lat = transformer.transform(df['center_x'].values, df['center_y'].values)

    df['center_x'] = lon
    df['center_y'] = lat

    # 4. 把表头列名还原成一开始的样子
    df.columns = original_columns

    # 5. 覆盖保存回原路径（index=False 表示不保存行号）
    df.to_csv(csv_path, index=False)

    # 打印成功提示
    file_name = os.path.basename(csv_path)
    print(f"✅ 成功覆盖文件: {file_name}")
    print(f"   此时文件内的中心点已变更为 -> X(经度): {lon[0]:.6f}°, Y(纬度): {lat[0]:.6f}°\n")


if __name__ == "__main__":
    # 你的文件绝对路径
    path_circle_1 = r"E:\lzd\python\贪心圆\111-copilot-process-fire-data-and-cluster\output\circle_1_center.csv"
    path_circle_8 = r"E:\lzd\python\贪心圆\111-copilot-process-fire-data-and-cluster\output\circle_8_center.csv"

    print("开始执行经纬度覆盖操作...\n")

    convert_and_overwrite(path_circle_1)
    convert_and_overwrite(path_circle_8)

    print("🎉 全部完成！CSV文件已被覆盖。")
    print("你现在可以打开 CSV 文件确认，或者直接运行你的原项目代码了！")