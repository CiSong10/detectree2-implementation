"""
Helper functions
"""

from pathlib import Path
from os import PathLike
import geopandas as gpd
from tqdm import tqdm
import shutil
import random
import rasterio
import cv2
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.engine import DefaultPredictor

from detectree2.preprocessing.tiling import image_details, is_overlapping_box
from detectree2.models.train import register_train_data
from detectree2.models.predict import get_tree_dicts, get_filenames


def find_final_model(path: str | PathLike[str]) -> str:
    """
    Given a path to either:
    - a checkpoint file (.pth), or
    - a directory containing multiple checkpoints named model_*.pth,

    Returns the path to `model_final.pth`.  
    If `model_final.pth` does not exist in the directory, it will create
    a symlink to the latest model (based on the numeric suffix).
    """
    path = Path(path)

    if path.suffix == ".pth":
        return str(path)
    
    if path.is_dir():
        model_final_path = path / 'model_final.pth'

        if not model_final_path.exists():
            model_files = sorted(
                path.glob("model_*.pth"),
                key=lambda f: int(f.stem.split("_")[1])
            )
            if not model_files:
                raise FileNotFoundError(f"No model_*.pth files found in {path}")
            
            latest_model = model_files[-1]
            model_final_path.symlink_to(latest_model.name)

        return str(model_final_path)
    
    raise ValueError(f"{path} is neither a .pth file nor a directory.")


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
    for crown_A_idx, group in tqdm(contained_pairs.groupby('index_A')):
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


def to_traintest_folders_sample(
        tiles_dir: Path,
        tiles_root: Path,
        test_frac: float = 0.15,
        folds: int = 1,
        strict: bool = False,
        seed: int = None,
        sample_n_tiles: int = None) -> None:
    """
    Split tiles into train/test folders, with optional sampling of training tiles
    before splitting into folds.

    Args:
        tiles_dir: folder with tiles
        tiles_root: folder to save train and test folders
        test_frac: fraction of tiles to be used for testing (not affected by sampling)
        folds: number of folds to split the data into
        strict: if True, remove overlapping train tiles
        seed: random seed
        sample_n_tiles: number of tiles to sample for training (after test selection).
                        If None, use all remaining tiles.

    Returns:
        None
    """

    if not tiles_dir.exists():
        raise IOError(f"Tiles folder does not exist: {tiles_dir}")

    # Clean previous train/test folders
    shutil.rmtree(tiles_root / "train", ignore_errors=True)
    shutil.rmtree(tiles_root / "test", ignore_errors=True)
    (tiles_root / "train").mkdir(parents=True, exist_ok=True)
    (tiles_root / "test").mkdir(parents=True, exist_ok=True)

    # Collect all tile stems
    tile_names = [p.stem for p in tiles_dir.glob("*.geojson")]
    if seed is not None:
        random.seed(seed)
    random.shuffle(tile_names)

    # --- 1) Split into test and train pools ---
    n_test = int(len(tile_names) * test_frac)
    test_tiles = tile_names[:n_test]
    train_tiles = tile_names[n_test:]  # remaining available for training

    # --- 2) Copy test tiles ---
    test_boxes = []
    for tile in test_tiles:
        test_boxes.append(image_details(tile))
        shutil.copy(tiles_dir / f"{tile}.geojson", tiles_root / "test")

    # --- 3) Optionally sample train tiles ---
    if sample_n_tiles is not None and sample_n_tiles < len(train_tiles):
        train_tiles = random.sample(train_tiles, sample_n_tiles)

    # --- 4) Copy training tiles (respecting "strict" mode) ---
    for tile in train_tiles:
        train_box = image_details(tile)
        if strict:
            if not is_overlapping_box(test_boxes, train_box):
                shutil.copy(tiles_dir / f"{tile}.geojson", tiles_root / "train")
        else:
            shutil.copy(tiles_dir / f"{tile}.geojson", tiles_root / "train")

    # --- 5) Split training into folds ---
    train_roots = [p.stem for p in (tiles_root / "train").glob("*.geojson")]
    random.shuffle(train_roots)
    folds_split = np.array_split(train_roots, folds)

    for i, fold in enumerate(folds_split, start=1):
        fold_dir = tiles_root / f"train/fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name in fold:
            shutil.move(tiles_root / f"train/{name}.geojson", fold_dir / f"{name}.geojson")


def safe_register_train_data(train_location, name: str = "tree", val_fold=None, class_mapping_file=None):
    # First unregister if already exists
    for d in ["train", "val", "full"]:
        dataset_name = f"{name}_{d}"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)

    # Then re-register as usual
    register_train_data(train_location, name, val_fold, class_mapping_file)


def parallel_predict_on_data(
        directory: str | Path = "./",
        out_folder: str = "predictions",
        predictor=DefaultPredictor,
        eval: bool=False,
        num_predictions=0,
        max_workers=4
        ) -> None:
    """
    Make predictions on tiled data in parallel using multiple processes.
    One global progress bar is shown.
    """
    directory = Path(directory)
    pred_dir = directory / out_folder
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset dictionaries
    if eval:
        dataset_dicts = get_tree_dicts(directory)
        if dataset_dicts:
            sample_file = Path(dataset_dicts[0]["file_name"])
            _, mode = get_filenames(sample_file.parent)
        else:
            mode = None
    else:
        dataset_dicts, mode = get_filenames(directory)

    # Subset if needed
    num_to_pred = len(dataset_dicts) if num_predictions == 0 else num_predictions

    # Run parallel prediction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_predict_file, d, pred_dir, predictor)
            for d in dataset_dicts
        ]

        # Global progress bar
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"Predicting files in mode {mode}",
                           unit="file"):
            future.result()  # raise exception if any


def _predict_file(d, pred_dir, predictor):
    """
    Worker function: create predictor, run inference, save results.
    Each process loads its own model (CUDA-safe).
    """

    file_name = Path(d['file_name'])
    file_ext = file_name.suffix.lower()

    if file_ext == ".png":
        img = cv2.imread(str(file_name))
    elif file_ext == ".tif":
        with rasterio.open(file_name) as src:
            img = src.read().transpose(1, 2, 0)
    else:
        return 
    
    outputs = predictor(img)

    output_file = pred_dir / f"Prediction_{file_name.stem}.json"
    evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), str(file_name))
    output_file.write_text(json.dumps(evaluations))
