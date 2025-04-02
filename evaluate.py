#!/usr/bin/env python3
import os
import argparse
import shutil
import glob
import logging
import sys
from typing import Tuple, Optional

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import (register_train_data, register_test_data, setup_cfg, 
                                    combine_dicts, load_json_arr, predictions_on_data)
from detectree2.models.outputs import to_eval_geojson
from detectree2.models.evaluation import site_f1_score2
from detectron2.engine import DefaultPredictor


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the evaluation script.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='DetecTree2 Evaluating Script')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('-i', '--input-dir', type=str, default='./data/train',
                        help='Root directory containing all train site folders')
    parser.add_argument('-b', '--buffer', type=int, default=30,
                        help='Buffer size for tiling')
    parser.add_argument('-s', '--tile-size', type=int, default=40,
                        help='Height and Width of tiles')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Threshold for tiling')
    parser.add_argument('--dem', type=str, default='./data/DEM_m.tif',
                        help='Path to DEM layer (relative to site directory)')
    parser.add_argument('--evaluation-mode', type=str, choices=['standard', 'coco'], default='standard',
                        help='Evaluation method to use (standard or COCO evaluator)')
    parser.add_argument('--output-dir', type=str, default='.data/evaluation/',
                        help='Directory to save evaluation results')
    return parser.parse_args()


def evaluate_model(cfg, tiles_dir, dem=None) -> Tuple[float, float, float]:
    """Evaluate a model on tiled data.
    
    Args:
        cfg: Model configuration
        tiles_dir (str): Directory containing tiles
        dem (str, optional): Path to DEM layer
        
    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score
    """
    pred_folder = os.path.join(tiles_dir, "predictions")
    shutil.rmtree(pred_folder, True) if os.path.exists(pred_folder) else None
    shutil.rmtree('./eval', True) if os.path.exists('./eval') else None
    predictions_on_data(tiles_dir, DefaultPredictor(cfg))
    to_eval_geojson(pred_folder)

    prec, recall, f1 = site_f1_score2(
        tile_directory=tiles_dir, 
        test_directory=os.path.join(tiles_dir, "test"),
        pred_directory=pred_folder,
        lidar_img=dem,
        IoU_threshold= 0.5,
        border_filter=[False, 1],
        conf_threshold=0.2,
        area_threshold=16,
    )

    return prec, recall, f1


def coco_evaluator(cfg, tiles_dir, output_dir="./evaluation_output/"):
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    logger.info("Use Detectron2 built-in COCO Evaluator")

    #register your data
    test_folder = os.path.join(tiles_dir, "test")
    register_test_data(test_folder)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("tree_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "tree_test")

    inference_on_dataset(predictor.model, val_loader, evaluator)
    

def main():
    args = parse_arguments()

    sites = [d for d in os.listdir(args.input_dir)
             if os.path.isdir(os.path.join(args.input_dir, d))]
    
    cfg = setup_cfg(update_model=args.model_path)
    
    results = {} # This currently doesn't do anything. Can develop to output average prec, recall, f1 in the future.
    for site in sites:
        logger.info(f"Processing site: {site}")
        tiles_dir = os.path.join(args.input_dir, site, f"tiles_{args.tile_size}_{args.buffer}_{args.threshold}")
    
        if args.evaluation_mode == 'standard':
            prec, recall, f1 = evaluate_model(cfg, tiles_dir, args.dem)
            results[site] = {'precision': prec,
                             'recall': recall,
                             'f1': f1}
        else: # coco evaluation
            coco_evaluator(cfg, tiles_dir, args.output_dir)
            evaluate_model(cfg, tiles_dir, args.dem)


if __name__ == "__main__":
    main()
