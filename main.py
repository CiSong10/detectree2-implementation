from pipeline import Pipeline
from configs import Configs
import geopandas as gpd
from utility.evaluate_iou import evaluate_tree_crowns
from pathlib import Path


configs = Configs(
    model=None,
    data=None,
    mode="ms",
    threshold=0,
    # force_retile=True,
    workers=15,
    resize="rand_fixed",
    confidence=0.2,
    intersection=0.4,
    containment=0.6,
    mask_filter_threshold=0.5,
)
pipeline = Pipeline(configs)
pipeline.train()
pipeline.predict()


# -- Eval --
from shapely.geometry import box

print(f"\n Evaluating model {configs.model}")

for data in configs.data:
    print(f"\nEvaluating {data}")
    data_dir = Path("data") / data
    pred_file = data_dir / f"{data}_prediction.gpkg"
    gt_file = list((data_dir / "crowns").glob("*.gpkg"))[0]

    pred = gpd.read_file(pred_file, layer=f"{data}_{configs.model}_postclean")
    gt = gpd.read_file(gt_file)

    minx, miny, maxx, maxy = gt.total_bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    pred_clip = pred[pred.geometry.within(bbox_geom)]

    metrics = evaluate_tree_crowns(pred_clip, gt, iou_threshold=0.5)

    for k, v in metrics.items():
        print(f"{k}: {v}")
