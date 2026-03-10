"""
Helper functions
"""

import json
from os import PathLike
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from pyogrio.errors import DataSourceError
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.data.transforms as T
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from detectree2.preprocessing.tiling import image_details, is_overlapping_box
from detectree2.models.train import register_train_data
from detectree2.models.predict import get_tree_dicts, get_filenames
from detectree2.models.outputs import GeoFile, box_filter, filename_geoinfo


class MultiBandPredictor:
    """
    A custom predictor supporting arbitrary number of input bands.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image with shape (H, W, C)
        Returns:
            dict: prediction output
        """
        assert img.ndim == 3, f"Expected (H, W, C), got {img.shape}"
        H, W, C = img.shape
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": H, "width": W}
        with torch.no_grad():
            predictions = self.model([inputs])[0]
        return predictions


def secondary_cleaning(crowns, 
                       containing_threshold=0.8, 
                       small_area_ratio=0.8):
    """
    Performs secondary cleaning on tree crown detections to handle complex overlapping scenarios,
    where model outputs a larger geometry as well as its subsets as crowns.
    
    Args:
        crowns_filepath (str): Path to the GPKG file with crown geometries.
        containing_threshold (float): Ratio threshold for considering a crown B contained in crown A.
        small_area_ratio (float): Area ratio threshold to decide which crown to keep.
        
    Returns:
        GeoDataFrame: Cleaned crown detections.
    """
    
    assert isinstance(crowns, (str, gpd.GeoDataFrame))

    if isinstance(crowns, str):
        crowns = gpd.read_file(crowns)
    
    crowns['area'] = crowns.geometry.area
    crowns = crowns.reset_index(drop=True)
    
    # Create spatial index and perform a spatial join with itself to get candidate pairs.
    # We exclude self-joins by later filtering out pairs with same index.
    joined = gpd.sjoin(crowns, crowns, how="inner", predicate="intersects", lsuffix="A", rsuffix="B")
    joined = joined[joined.index != joined.index_B].copy()
    joined = joined.merge(crowns[['geometry']], left_on='index_B', right_index=True, suffixes=('_A', '_B'))
    
    # Compute intersection area for each candidate pair using apply.
    def compute_intersection(row):
        geom_A = row['geometry_A']
        geom_B = row['geometry_B']
        return geom_A.intersection(geom_B).area
    
    joined['intersection_area'] = joined.apply(compute_intersection, axis=1)
    
    # Compute ratio: how much of crown B is contained in crown A
    # (Using crown B's area from the right-hand side dataset)
    joined['contain_ratio'] = joined['intersection_area'] / joined['area_B']
    
    # Filter candidate pairs where crown B is significantly contained within crown A
    contained_pairs = joined[joined['contain_ratio'] >= containing_threshold]
    
    # Prepare a DataFrame to help decide which crowns to remove.
    # The following logic follows your original idea:
    #   - For each crown A, if it "contains" crown(s) B, compare confidence and area.
    #   - Note: In cases where multiple crown Bs are contained in a single A, you can group them.
    decisions = []
    
    contained_pairs = contained_pairs.reset_index().rename(columns={'index':'index_A'})

    # Iterate over unique crown A candidates (using a grouped approach)
    for crown_A_idx, group in tqdm(contained_pairs.groupby('index_A'), desc="Secondary Cleaning"):
        # Get attributes for crown A from the main crowns df
        crown_A = crowns.loc[crown_A_idx]
        conf_A = crown_A.Confidence_score
        area_A = crown_A.area
        
        # List to hold candidate information from crown B
        candidate_B = group[['index_B', 'area_B', 'Confidence_score_B']].to_dict('records')
        
        if len(candidate_B) == 0:
            continue
        
        # For a single contained crown:
        if len(candidate_B) == 1:
            crown_B = candidate_B[0]
            if conf_A >= crown_B['Confidence_score_B']: # Larger crown A is better
                decisions.append((crown_B['index_B'], 'remove'))  # remove smaller B
            elif crown_B['area_B'] >= small_area_ratio * area_A: # Slightly smaller crown B is better
                decisions.append((crown_A_idx, 'remove'))  # remove A
            else: # Smaller crown is too small despite better confidence
                decisions.append((crown_B['index_B'], 'remove'))
        else:
            # Multiple crown Bs: decide based on average confidence
            avg_conf_B = sum(item['Confidence_score_B'] for item in candidate_B) / len(candidate_B)
            if avg_conf_B > conf_A:
                decisions.append((crown_A_idx, 'remove'))
            else:
                for item in candidate_B:
                    decisions.append((item['index_B'], 'remove'))
    
    crowns_to_remove = set(idx for idx, action in decisions if action == 'remove')
    cleaned_crowns = crowns[~crowns.index.isin(crowns_to_remove)]
    
    print(f"Total crowns removed: {len(crowns_to_remove)} of {len(crowns)} ({len(crowns_to_remove)/len(crowns)*100:.1f}%)")
    
    return cleaned_crowns


def safe_register_train_data(train_location, name: str = "tree", val_fold=None, class_mapping_file=None):
    # First unregister if already exists
    for d in ["train", "val", "full"]:
        dataset_name = f"{name}_{d}"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)

    # Then re-register as usual
    register_train_data(train_location, name, val_fold, class_mapping_file)


def canopy_mask_filter(crowns_path, canopy_mask_path, threshold=0.5):

    if isinstance(crowns_path, PathLike):
        crowns = gpd.read_file(crowns_path)
    elif isinstance(crowns_path, gpd.GeoDataFrame):
        crowns = crowns_path
    
    with rasterio.open(canopy_mask_path) as src:
        tree_fracs = []

        for geom in tqdm(crowns.geometry, desc="Filtering crowns with canopy mask"):
            try:
                out_image, out_transform = mask(src, [mapping(geom)], crop=True, filled=False)
                data = out_image[0]

                # Keep only valid pixels
                valid = data[data >= 0]  # skip nodata
                if valid.size == 0:
                    tree_frac = 0.0
                else:
                    tree_frac = np.sum(valid == 1) / valid.size

            except (rasterio.errors.WindowError, ValueError) as e:
                print(f"Error with geometry {geom}: {str(e)}")
                tree_frac = 0.0
            
            tree_fracs.append(tree_frac)

    # Add it as a new column
    crowns["tree_frac_in_mask"] = tree_fracs

    # Keep polygons where tree coverage >= 0.5 (i.e. ≤50% non-tree)
    crowns_filtered = crowns[crowns["tree_frac_in_mask"] >= threshold].copy()
    crowns_filtered.drop(columns=['tree_frac_in_mask'], errors="ignore", inplace=True)

    print(f"Canopy Mask filtered {len(crowns) - len(crowns_filtered)} polygons out of {len(crowns)}.")
    
    return crowns_filtered


def predict_on_data(
    directory: str | Path,
    out_folder: str | Path | None = None,    
    predictor=DefaultPredictor,
    trees_metadata=None,
    eval: bool = False,
    geos_exist: bool = True,
    save: bool = True,
    scale: float = 1,
    visualize: bool = False,
    num_predictions: int = 0,
    ) -> None:
    """Make predictions on data.
    This function is a combination of detectree2.models.train.predictions_on_data and 
    detectree2.models.predict.predict_on_data

    Args:
        directory (str): Directory containing tiled data. Input test data directory when evaluating with test data
        predictor (DefaultPredictor): The predictor object.
        trees_metadata: Metadata for trees.
        save (bool): Whether to save the predictions.
        scale (float): Scale of the image for visualization.
        geos_exist (bool): Determines if geojson files exist.
        num_predictions (int): Number of predictions to make.
    
    Returns:
        None
    """
    directory = Path(directory)
    if out_folder is None:
        pred_dir = directory / "predictions"
    else:
        pred_dir = Path(out_folder)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if eval or geos_exist:
        dataset_dicts = get_tree_dicts(directory)
        if dataset_dicts:
            sample_file = Path(dataset_dicts[0]["file_name"])
            _, mode = get_filenames(sample_file.parent)
        else:
            mode = None
    else:
        dataset_dicts, mode = get_filenames(directory)

    # Decide how many items to predict on
    num_to_pred = len(dataset_dicts) if num_predictions == 0 else num_predictions

    for d in tqdm(dataset_dicts[:num_to_pred],
                  desc=f"Predicting files in mode {mode}"):
        file_name = Path(d["file_name"])
        file_ext = file_name.suffix.lower()

        # --- Read image ---
        if file_ext == ".png":
            img = cv2.imread(str(file_name))
            if img is None:
                print(f"Failed to read {file_name}.")
                continue
            img_vis = img[:, :, ::-1] if visualize else None

        elif file_ext == ".tif":
            with rasterio.open(file_name) as src:
                img = src.read().transpose(1, 2, 0)
            img_vis = img[:, :, :3] if visualize and img.shape[2] >= 3 else img if visualize else None

        else:
            print(f"Unsupported file type: {file_ext}")
            continue

        # --- Prediction ---
        outputs = predictor(img)

        # --- Visualization if needed ---
        if visualize and img_vis is not None:
            v = Visualizer(
                img_vis,
                metadata=trees_metadata,
                scale=scale,
                instance_mode=ColorMode.SEGMENTATION,
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # --- Save JSON ---
        if save:
            output_file = pred_dir / f"Prediction_{file_name.stem}.json"
            evaluations = instances_to_coco_json(
                outputs["instances"].to("cpu"), str(file_name)
            )
            output_file.write_text(json.dumps(evaluations))

##########################################

def polygon_from_mask(mask):
    """
    Convert a binary mask to a polygon.
    
    Returns
    -------
    list of float or 0
        Flattened polygon coordinates in [x1, y1, x2, y2, ...] format,
        or 0 if no valid polygon found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0
    
    contour = max(contours, key=cv2.contourArea) # Pick the largest contour instead of the first one

    # Flatten to (x1, y1, x2, y2, ...)
    contour = contour.reshape(-1, 2)

    if len(contour) < 10:
        return 0
    
    # Ensure closed polygon
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    
    return contour.flatten().tolist()

# ================= PROJECT ========================

def project_to_geojson_parallel(
    tiles_path,
    pred_fold,
    output_fold,
    # multi_class=False,
    max_workers=16
):

    pred_fold = Path(pred_fold)
    output_fold = Path(output_fold)
    output_fold.mkdir(parents=True, exist_ok=True)

    entries = [f for f in pred_fold.iterdir() if f.suffix == ".json"]
    if not entries:
        raise RuntimeError(f"No prediction .json files in {pred_fold}")

    # Submit jobs to workers
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for json_file in entries:
            future = executor.submit(_to_geojson_worker, json_file, tiles_path, output_fold)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Projecting files"):
            future.result() 


def _to_geojson_worker(file, tiles_path, output_fold): # multi_class=multi_class
    # ### TEMPORARY ###
    # if output_file.exists() and output_file.stat().st_size > 20:
    #     return
    # #################
    output_file = output_fold / f"{file.stem}.geojson"
    tif_path = Path(tiles_path) / f"{file.stem.removeprefix('Prediction_')}.tif"
    with rasterio.open(tif_path) as src:
        epsg = src.crs.to_epsg()
        transform = src.transform

    with file.open("r") as f:
        predictions = json.load(f)
    
    features = []

    for crown_data in predictions:
        crown = crown_data['segmentation']
        score = crown_data['score']

        mask = mask_util.decode(crown)
        poly = polygon_from_mask(mask)
        if not poly:
            continue

        coords = np.array(poly).reshape(-1, 2)
        x_coords, y_coords = rasterio.transform.xy(
            transform, rows = coords[:, 1], cols = coords[:, 0]
        )
        moved_coords = list(zip(x_coords, y_coords))

        feature = {
            "type": "Feature",
            "properties": {"Confidence_score": score},
            "geometry": {"type": "Polygon", "coordinates": [moved_coords],}
            }

        # if multi_class:
        #     feature['properties']['category'] = crown_data['category_id']
        
        features.append(feature)

    geofile = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}
        },
        "features": features
    }

    with output_file.open("w") as f:
        json.dump(geofile, f)


########### STITCH  ###############

def _stitch(file, shift):
    try:
        crowns_tile = gpd.read_file(file)
    except DataSourceError:
        return
    geo = box_filter(file, shift) 
    crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
    return crowns_tile


def stitch_crowns_parallel(
        folder: str = None, 
        files: list = None, 
        shift: int = 1, 
        max_workers=10, 
        chunk_size=1000):
    if folder:
        crowns_path = Path(folder)
        files = list(crowns_path.glob("*.geojson"))
    elif files:
        files = files
    
    if not files:
        raise FileNotFoundError(f"No geojson files found in {crowns_path}.")

    _, _, _, _, crs = filename_geoinfo(files[0])

    list_of_chunks = []
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(files), chunk_size), desc="Stitching crowns", unit="chunk"):
            chunk = files[i : i + chunk_size] # Process in chunks
            in_chunk_list = []

            futures = {executor.submit(_stitch, f, shift): f for f in chunk}

            for future in as_completed(futures):
                df = future.result()
                if df is None:
                    failed_files.append(str(futures[future]))
                else:
                    in_chunk_list.append(df)

            chunk_crowns = pd.concat(in_chunk_list, ignore_index=True) # .drop(columns=["index_right"], errors="ignore")
            list_of_chunks.append(chunk_crowns)

    crowns = pd.concat(list_of_chunks, ignore_index=True)
    crowns = crowns.drop(columns=["index_right"], errors="ignore")
    crowns = gpd.GeoDataFrame(crowns, crs=f"EPSG:{crs}")

    if failed_files:
        log_path = crowns_path.parent / "failed_stitching_files.txt"
        with log_path.open("w") as f:
            f.write("\n".join(failed_files))
        print(f"{len(failed_files)} files failed. See {str(log_path)}")

    return crowns


def _compute_fraction_window(args):
    """Worker receives geometry plus metadata + full raster array"""
    geom, mask_arr, transform = args

    # Bounding box of polygon
    minx, miny, maxx, maxy = geom.bounds

    # Convert bounds to pixel indices
    row_min, col_min = ~transform * (minx, maxy)  # upper-left corner
    row_max, col_max = ~transform * (maxx, miny)  # lower-right corner

    # Pixel windows must be int indices
    r0, c0 = max(0, int(np.floor(row_min))), max(0, int(np.floor(col_min)))
    r1, c1 = int(np.ceil(row_max)), int(np.ceil(col_max))

    # Skip invalid window
    if r1 <= r0 or c1 <= c0:
        return 0.0

    # Clip the numpy mask region
    sub = mask_arr[r0:r1, c0:c1]

    if sub.size == 0:
        return 0.0

    # Clip the mask by polygon EXACTLY (needed for accuracy)
    # Create a boolean mask using rasterized geometry
    from rasterio.features import rasterize
    poly_mask = rasterize(
        [(geom, 1)],
        out_shape=sub.shape,
        transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, sub.shape[1], sub.shape[0]),
        fill=0,
        dtype="uint8"
    )

    # Extract only pixels inside polygon
    valid = sub[poly_mask == 1]

    if valid.size == 0:
        return 0.0

    # Calculate fraction of canopy (mask pixels == 1)
    return float(np.sum(valid == 1) / valid.size)


def canopy_mask_filter_fast(
    crowns_path,
    mask_arr,
    transform,
    threshold: float = 0.25,
    workers: int = 12,
):
    """Fast canopy mask filter using window slicing + parallel processing."""

    if isinstance(crowns_path, gpd.GeoDataFrame):
        crowns = crowns_path.copy()
    else:
        crowns = gpd.read_file(crowns_path)

    geoms = list(crowns.geometry)

    args = [(g, mask_arr, transform) for g in geoms]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(_compute_fraction_window, args),
                total=len(args),
                desc="Filtering crowns with canopy mask",
            )
        )

    crowns["tree_frac_in_mask"] = results
    filtered = crowns[crowns["tree_frac_in_mask"] >= threshold].copy()
    filtered.drop(columns=["tree_frac_in_mask"], errors="ignore", inplace=True)
    print(f"Canopy Mask filtered {len(crowns)-len(filtered)} polygons out of {len(crowns)}.")

    return filtered


if __name__=="__main__":
    pass