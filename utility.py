"""
Helper functions
"""

from pathlib import Path
from os import PathLike
import geopandas as gpd
from tqdm import tqdm

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
