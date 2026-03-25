import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, TypedDict

# from detectree2.models.outputs import project_to_geojson, to_eval_geojson, stitch_crowns, clean_crowns, post_clean
from detectree2.models.outputs import calc_iou


def post_clean(
    unclean_df: gpd.GeoDataFrame,
    clean_df: gpd.GeoDataFrame,
    iou_threshold: float = 0.3,
    field: str = "Confidence_score",
    max_iterations: int = 5,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Fill in the gaps left by clean_crowns.

    Takes the original (unclean) crowns and the cleaned set, then iteratively adds back crowns
    from the unclean set that do not significantly overlap with any cleaned crown. Each round,
    the combined result is re-cleaned to handle mutual overlaps among the newly added crowns.
    Iteration continues until no new crowns are added or ``max_iterations`` is reached.

    Args:
        unclean_df (gpd.GeoDataFrame): Unclean crowns.
        clean_df (gpd.GeoDataFrame): Clean crowns.
        iou_threshold (float, optional): IoU threshold that determines whether predictions are
            considered overlapping. Defaults to 0.3.
        field (str): Field used to prioritise selection of crowns. Defaults to "Confidence_score".
        max_iterations (int, optional): Maximum number of gap-filling rounds. Defaults to 5.
        verbose (bool, optional): Print progress information. Defaults to True.
    """
    # Fix invalid geometries once upfront, not per-row
    unclean_df = unclean_df.copy()
    unclean_df["geometry"] = unclean_df.geometry.buffer(0)

    current_clean = clean_df.copy()
    current_clean["geometry"] = current_clean.geometry.buffer(0)

    for iteration in range(1, max_iterations + 1):
        prev_count = len(current_clean)

        # Spatial join to find candidate overlapping pairs (bbox intersection)
        joined_df = gpd.sjoin(
            unclean_df, current_clean, how="inner", predicate="intersects"
        )

        # Use a set for O(1) lookup; skip further pairs once an unclean crown is marked
        to_remove = set()
        for idx, row in joined_df.iterrows():
            if idx in to_remove:
                continue  # Already marked for removal, skip remaining pairs

            iou = calc_iou(
                unclean_df.loc[idx, "geometry"],
                current_clean.loc[row["index_right"], "geometry"],
            )
            if iou > iou_threshold:
                to_remove.add(idx)

        reduced_unclean_df = unclean_df.drop(index=to_remove)

        # Concatenate the reduced unclean dataframe with the clean dataframe
        result_df = pd.concat([current_clean, reduced_unclean_df], ignore_index=True)
        result_df.reset_index(drop=True, inplace=True)

        # Re-clean the combined set to resolve any mutual overlaps among newly added crowns
        current_clean = clean_crowns(
            result_df, iou_threshold=iou_threshold, field=field, verbose=verbose
        )
        current_clean.reset_index(drop=True, inplace=True)

        new_count = len(current_clean)
        if verbose:
            print(
                f"post_clean: iteration {iteration} — {prev_count} → {new_count} crowns "
                f"(+{new_count - prev_count})"
            )

        if new_count == prev_count:
            if verbose:
                print("post_clean: converged, no new crowns added.")
            break

    return current_clean


def clean_crowns(
    crowns,
    iou_threshold=0.7,
    confidence=0.2,
    area_threshold=2,
    field="Confidence_score",
    containment_threshold=0.85,
    verbose=True,
) -> gpd.GeoDataFrame:
    # 1. Filter out invalid geometries and tiny artifacts.
    crowns = crowns[~crowns.is_empty & crowns.is_valid].copy()
    crowns = crowns[crowns.area > area_threshold].copy()

    if confidence:
        crowns = crowns[crowns[field] > confidence]

    crowns.reset_index(drop=True, inplace=True)

    # 2. Use a spatial join to quickly find all candidate overlapping pairs.
    #    The join will pair each crown with any crown whose bounding box intersects.
    print("clean_crowns: Performing spatial join...")
    join = gpd.sjoin(crowns, crowns, how="inner", predicate="intersects")
    # Remove self-joins (where a crown is paired with itself).
    join = join[join.index != join.index_right]

    # 3. Build a conflict graph: edges between crowns that truly conflict:
    # (high IoU OR containment).
    from collections import defaultdict

    conflicts = defaultdict(set)  # crowns_idx -> set of conflicting crown_idxs

    for _, row in tqdm(
        join.iterrows(),
        total=len(join),
        desc="clean_crowns: Building conflict graph",
        smoothing=0,
        disable=not verbose,
    ):
        i = row.name  # index from left table (crowns)
        j = row["index_right"]  # index from right table (crowns)
        # To avoid duplicate work, skip if i and j are already in the same group.
        if i >= j:
            continue

        geom_i = crowns.at[i, "geometry"]
        geom_j = crowns.at[j, "geometry"]
        intersection_area = geom_i.intersection(geom_j).area

        # IoU check
        # union_area = geom_i.area + geom_j.area - intersection_area
        # iou_val = intersection_area / union_area if union_area > 0 else 0

        iou_val = calc_iou(geom_i, geom_j)

        is_conflict = iou_val > iou_threshold

        # Containment check: is the smaller crown mostly inside the larger one?
        if not is_conflict and containment_threshold is not None:
            min_area = min(geom_i.area, geom_j.area)
            if min_area > 0 and (intersection_area / min_area) > containment_threshold:
                is_conflict = True

        if is_conflict:
            conflicts[i].add(j)
            conflicts[j].add(i)

    # 4. Greedy NMS: sort by confidence descending,
    #  keep a crown if it doesn't conflict with any already-kept crown.
    sorted_indices = crowns[field].sort_values(ascending=False).index.to_list()
    kept = set()
    removed = set()

    for i in sorted_indices:
        if i in removed:
            continue
        kept.add(i)

        for j in conflicts[i]:
            removed.add(j)

    cleaned_crowns = crowns.loc[sorted(kept)].copy()
    return gpd.GeoDataFrame(cleaned_crowns, crs=crowns.crs).reset_index(drop=True)


if __name__ == "__main__":
    pass
