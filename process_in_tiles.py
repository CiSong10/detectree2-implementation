import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path

def process_in_tiles(gdf, func, nx=5, ny=5, buffer=0):
    """
    Split a large GeoDataFrame into a regular grid tiles and apply a function
    to features within each tile. Returns a concatenated GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): input data
        func (callable): function to apply, must take a GeoDataFrame and return a GeoDataFrame
        nx, ny (int): number of tiles in x and y direction (e.g. 5x5)
        buffer (float): optional buffer around each tile to avoid boundary clipping

    Returns:
        GeoDataFrame: merged output after applying func to each tile
    """
    
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Compute tile width/height
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    results = []

    for i in range(nx):
        for j in range(ny):
            print(f"Processing tile {(i * nx + j + 1)} / {(i+1) * (j+1)}")
            # Create tile bounds
            x1 = xmin + i * dx
            x2 = x1 + dx
            y1 = ymin + j * dy
            y2 = y1 + dy

            tile = box(x1 - buffer, y1 - buffer, x2 + buffer, y2 + buffer)

            # Subset GDF by intersection with tile
            subset = gdf[gdf.intersects(tile)].copy()

            if subset.empty:
                continue
            
            # Apply your cleaning function
            result = func(subset)

            results.append(result)

    return gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=gdf.crs)


if __name__ == "__main__":
    from detectree2.models.outputs import clean_crowns
    from utility import secondary_cleaning

    data_dir = Path("data/akron/")
    # gdf = gpd.read_file(data_dir / "akron_prediction.gpkg")
    # clean1 = process_in_tiles(gdf, clean_crowns, buffer=5)
    # clean1.to_file(data_dir / "clean1.gpkg", driver="GPKG")

    clean1 = gpd.read_file(data_dir / "clean1.gpkg")
    clean2 = process_in_tiles(clean1, secondary_cleaning, nx=2, ny=2)
    clean2.to_file(data_dir / "akron_tree_segmentation.gpkg", driver="GPKG")