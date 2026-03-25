from pathlib import Path
import geopandas as gpd
from shapely.geometry import box


def compute_iou(geom_a, geom_b) -> float:
    intersection_area = geom_a.intersection(geom_b).area
    if intersection_area == 0:
        return 0.0
    union_area = geom_a.union(geom_b).area
    return intersection_area / union_area if union_area > 0 else 0.0


def match_polygons(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Greedy one-to-one matching between GT and prediction polygons.

    Strategy
    --------
    For every GT polygon, find all candidate predictions whose bounding box
    overlaps (via spatial index), compute IoU, and keep the best match that
    exceeds *iou_threshold*.  Each prediction can be matched to at most one
    GT polygon.

    Returns
    -------
    dict with keys:
        tp          - number of true positives
        fp          - number of false positives
        fn          - number of false negatives
        matches     - list of (gt_idx, pred_idx, iou) tuples
        unmatched_gt   - list of unmatched GT indices
        unmatched_pred - list of unmatched pred indices
        iou_scores  - list of IoU values for matched pairs
    """
    matched_pred_indices = set()
    matches = []
    unmatched_gt = []

    for gt_idx, gt_row in gt.iterrows():
        gt_geom = gt_row.geometry

        # candidate predictions via spatial index
        cand_idx = list(pred.sindex.intersection(gt_geom.bounds))
        if not cand_idx:
            unmatched_gt.append(gt_idx)
            continue

        best_iou = 0.0
        best_pred_idx = None

        for pi in cand_idx:
            if pi in matched_pred_indices:
                continue
            iou = compute_iou(gt_geom, pred.geometry.iloc[pi])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pi

        if best_pred_idx is not None and best_iou >= iou_threshold:
            matches.append((gt_idx, best_pred_idx, best_iou))
            matched_pred_indices.add(best_pred_idx)
        else:
            unmatched_gt.append(gt_idx)

    unmatched_pred = [i for i in range(len(pred)) if i not in matched_pred_indices]

    tp = len(matches)
    fn = len(unmatched_gt)
    fp = len(unmatched_pred)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "iou_scores": [m[2] for m in matches],
    }


def evaluate_tree_crowns(pred, gt, iou_threshold=0.5):
    """
    Evaluate predicted tree crowns polygons against gournd truth polygons.

    Returns
    -------
    dict
        Dictionary containing Precision, Recall, and F1 score.
    """

    # 1. Read polygons

    print(f"Ground truth : {len(gt):,} polygons")
    print(f"Predictions  : {len(pred):,} polygons")

    if pred.crs != gt.crs:
        pred = pred.to_crs(gt.crs)

    # 2. Crop predictions to ground truth bbox
    minx, miny, maxx, maxy = gt.total_bounds
    bbox = box(minx, miny, maxx, maxy)

    pred = pred[pred.geometry.intersects(bbox)].copy()
    pred.reset_index(drop=True, inplace=True)
    gt = gt.copy()
    gt.reset_index(drop=True, inplace=True)

    pred["area"] = pred.geometry.area
    gt["area"] = gt.geometry.area

    # 3. IoU matching
    match_result = match_polygons(gt, pred, iou_threshold)

    # Counts
    tp, fp, fn = match_result["tp"], match_result["fp"], match_result["fn"]

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return {
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
    }


if __name__ == "__main__":
    data = "kent_3in"
    model = "flexi"

    print(f"/n Evaluating {data}")
    data_dir = Path("data") / data
    pred_file = data_dir / "{data}_prediction.gpkg"
    gt_file = list((data_dir / "crowns").glob("*.gpkg"))[0]

    pred = gpd.read_file(pred_file, layer=f"{data}_{model}_postclean")
    gt = gpd.read_file(gt_file)

    metrics = evaluate_tree_crowns(pred, gt, iou_threshold=0.5)

    for k, v in metrics.items():
        print(f"{k}: {v}")
