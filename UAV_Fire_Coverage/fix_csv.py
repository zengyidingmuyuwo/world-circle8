import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
base_dir = os.path.join(BASE_DIR, "prepare")
files = ["circle_1_center.csv", "circle_8_center.csv"]
original_crs = "EPSG:32647"

for file in files:
    csv_path = os.path.join(base_dir, file)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 注意：在 UTM47N 中，X(Easting)通常是几十万，Y(Northing)是几百万
        # 从你的日志看，center_x 是 2481739(Y/纬度向)，center_y 是 658222(X/经度向)
        geometry = [Point(y, x) for x, y in zip(df['center_x'], df['center_y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=original_crs)
        gdf_4326 = gdf.to_crs(epsg=4326)

        # 直接暴力覆盖原来的 center_x 和 center_y！
        # 让代码无论读哪个列，读到的都是 25.xxx 和 101.xxx
        df['center_x'] = gdf_4326.geometry.y  # 纬度 (25.xxx)
        df['center_y'] = gdf_4326.geometry.x  # 经度 (101.xxx)
        df['latitude'] = gdf_4326.geometry.y
        df['longitude'] = gdf_4326.geometry.x

        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ {file} 暴力覆盖成功！现在的中心点是: {df['center_x'].iloc[0]:.4f}, {df['center_y'].iloc[0]:.4f}")