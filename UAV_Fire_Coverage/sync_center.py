import pandas as pd
import geopandas as gpd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
base_dir = os.path.join(BASE_DIR, "prepare")
pairs = [
    ("circle_1_center.csv", "circle_1_points.shp"),
    ("circle_8_center.csv", "circle_8_points.shp")
]

for csv_name, shp_name in pairs:
    csv_path = os.path.join(base_dir, csv_name)
    shp_path = os.path.join(base_dir, shp_name)

    if os.path.exists(csv_path) and os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        # 兼容老版本和新版本的 geopandas
        try:
            centroid = gdf.geometry.union_all().centroid
        except:
            centroid = gdf.geometry.unary_union.centroid

        df = pd.read_csv(csv_path)

        # 【关键修复】X 是经度 (100.5)，Y 是纬度 (22.4)
        df['latitude'] = centroid.y
        df['longitude'] = centroid.x
        if 'center_x' in df.columns:
            df['center_x'] = centroid.x  # 修复：x 对应经度
        if 'center_y' in df.columns:
            df['center_y'] = centroid.y  # 修复：y 对应纬度

        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ {csv_name} 最终对齐！")