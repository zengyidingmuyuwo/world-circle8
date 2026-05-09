import geopandas as gpd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PREPARE_DIR = os.path.join(BASE_DIR, "prepare")
shp_path = os.path.join(PREPARE_DIR, "circle_1_points.shp")
gdf = gpd.read_file(shp_path)

print("\n🔍 火点数据检查结果：")
print("一共读取到", len(gdf), "个火点。")
print("前三个火点的实际坐标是：")
for i in range(3):
    print(f"火点 {i+1}:", gdf.geometry.iloc[i])