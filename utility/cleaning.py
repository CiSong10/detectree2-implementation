from os import PathLike
import geopandas as gpd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm


def secondary_cleaning(crowns, containing_threshold=0.8, small_area_ratio=0.8):
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

    crowns["area"] = crowns.geometry.area
    crowns = crowns.reset_index(drop=True)

    # Create spatial index and perform a spatial join with itself to get candidate pairs.
    # We exclude self-joins by later filtering out pairs with same index.
    joined = gpd.sjoin(
        crowns, crowns, how="inner", predicate="intersects", lsuffix="A", rsuffix="B"
    )
    joined = joined[joined.index != joined.index_B].copy()
    joined = joined.merge(
        crowns[["geometry"]], left_on="index_B", right_index=True, suffixes=("_A", "_B")
    )

    # Compute intersection area for each candidate pair using apply.
    def compute_intersection(row):
        geom_A = row["geometry_A"]
        geom_B = row["geometry_B"]
        return geom_A.intersection(geom_B).area

    joined["intersection_area"] = joined.apply(compute_intersection, axis=1)

    # Compute ratio: how much of crown B is contained in crown A
    # (Using crown B's area from the right-hand side dataset)
    joined["contain_ratio"] = joined["intersection_area"] / joined["area_B"]

    # Filter candidate pairs where crown B is significantly contained within crown A
    contained_pairs = joined[joined["contain_ratio"] >= containing_threshold]

    # Prepare a DataFrame to help decide which crowns to remove.
    # The following logic follows your original idea:
    #   - For each crown A, if it "contains" crown(s) B, compare confidence and area.
    #   - Note: In cases where multiple crown Bs are contained in a single A, you can group them.
    decisions = []

    contained_pairs = contained_pairs.reset_index().rename(columns={"index": "index_A"})

    # Iterate over unique crown A candidates (using a grouped approach)
    for crown_A_idx, group in tqdm(
        contained_pairs.groupby("index_A"), desc="Secondary Cleaning"
    ):
        # Get attributes for crown A from the main crowns df
        crown_A = crowns.loc[crown_A_idx]
        conf_A = crown_A.Confidence_score
        area_A = crown_A.area

        # List to hold candidate information from crown B
        candidate_B = group[["index_B", "area_B", "Confidence_score_B"]].to_dict(
            "records"
        )

        if len(candidate_B) == 0:
            continue

        # For a single contained crown:
        if len(candidate_B) == 1:
            crown_B = candidate_B[0]
            if conf_A >= crown_B["Confidence_score_B"]:  # Larger crown A is better
                decisions.append((crown_B["index_B"], "remove"))  # remove smaller B
            elif (
                crown_B["area_B"] >= small_area_ratio * area_A
            ):  # Slightly smaller crown B is better
                decisions.append((crown_A_idx, "remove"))  # remove A
            else:  # Smaller crown is too small despite better confidence
                decisions.append((crown_B["index_B"], "remove"))
        else:
            # Multiple crown Bs: decide based on average confidence
            avg_conf_B = sum(item["Confidence_score_B"] for item in candidate_B) / len(
                candidate_B
            )
            if avg_conf_B > conf_A:
                decisions.append((crown_A_idx, "remove"))
            else:
                for item in candidate_B:
                    decisions.append((item["index_B"], "remove"))

    crowns_to_remove = set(idx for idx, action in decisions if action == "remove")
    cleaned_crowns = crowns[~crowns.index.isin(crowns_to_remove)]

    print(
        f"Total crowns removed: {len(crowns_to_remove)} of {len(crowns)} ({len(crowns_to_remove)/len(crowns)*100:.1f}%)"
    )

    return cleaned_crowns


def canopy_mask_filter(crowns_path, canopy_mask_path, threshold=0.5):

    if isinstance(crowns_path, PathLike):
        crowns = gpd.read_file(crowns_path)
    elif isinstance(crowns_path, gpd.GeoDataFrame):
        crowns = crowns_path

    with rasterio.open(canopy_mask_path) as src:
        tree_fracs = []

        for geom in tqdm(crowns.geometry, desc="Filtering crowns with canopy mask"):
            try:
                out_image, out_transform = mask(
                    src, [mapping(geom)], crop=True, filled=False
                )
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
    crowns_filtered.drop(columns=["tree_frac_in_mask"], errors="ignore", inplace=True)

    print(
        f"Canopy Mask filtered {len(crowns) - len(crowns_filtered)} polygons out of {len(crowns)}."
    )

    return crowns_filtered


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
        transform=rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, sub.shape[1], sub.shape[0]
        ),
        fill=0,
        dtype="uint8",
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
    print(
        f"Canopy Mask filtered {len(crowns)-len(filtered)} polygons out of {len(crowns)}."
    )

    return filtered
