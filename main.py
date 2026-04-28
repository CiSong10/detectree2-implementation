import geopandas as gpd
import logging
from pathlib import Path
from shapely.geometry import box

from configs import Configs
from pipeline import Pipeline
from utility.evaluate_iou import evaluate_tree_crowns

logging.getLogger("detectree2.preprocessing.tiling").setLevel(logging.ERROR)


configs = Configs(
    model=None,
    data=None,
    mode="rgb",
    threshold=0,
    # force_retile=True,
    workers=15,
    resize="rand_fixed",
    confidence=0.2,
    intersection=0.4,
    containment=0.6,
    mask_filter_threshold=0.5,
    # resume=True
)
pipeline = Pipeline(configs)
pipeline.train()
pipeline.predict()

# -- Eval --
print(f"\n Evaluating model {configs.model}")
tp = 0
fp = 0
fn = 0
output_lines = []

for data in configs.data:
    print(f"\nEvaluating {data}")
    output_lines.append(f"\nEvaluating {data}")
    data_dir = Path("data") / data
    pred_file = data_dir / f"{data}_prediction.gpkg"
    gt_file = list((data_dir / "crowns").glob("*.gpkg"))[0]

    pred = gpd.read_file(pred_file, layer=f"{data}_{configs.model}_postclean")
    gt = gpd.read_file(gt_file)

    minx, miny, maxx, maxy = gt.total_bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    pred_clip = pred[pred.geometry.within(bbox_geom)]

    metrics = evaluate_tree_crowns(pred_clip, gt, iou_threshold=0.5)
    tp += metrics["True Positives"]
    fp += metrics["False Positives"]
    fn += metrics["False Negatives"]

    for k, v in metrics.items():
        print(f"{k}: {v}")
        output_lines.append(f"{k}: {v}")

f1 = 2 * tp / (2 * tp + fp + fn)
print(f"F1 Score for model {configs.model}: {f1:.2f}")
output_lines.append(f"F1 Score for model {configs.model}: {f1:.2f}")

# Write eval to eval.txt
if configs.model_dir:
    eval_file = configs.model_dir / "eval.txt"
    if not eval_file.exists():
        with open(eval_file, "w") as f:
            f.write("\n".join(output_lines))
