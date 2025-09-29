#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
import shutil
import cv2
import rasterio
import geopandas as gpd
from PIL import Image
from datetime import datetime
from detectree2.preprocessing.tiling import tile_data, to_traintest_folders
from detectree2.models.train import (register_train_data, MyTrainer, setup_cfg, 
                                    combine_dicts, load_json_arr, predictions_on_data)
import iopath.common.file_io as file_io
from glob import glob

from evaluate import evaluate_model
from pathlib import Path

from pipeline import find_final_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='DetecTree2 Fine-tuning Script')
    parser.add_argument('-i', '--train-dir', type=str, default='./data/train/',
                        help='Root training directory containing all site folders')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Custom output models directory name')    
    parser.add_argument('-b', '--buffer', type=int, default=30,
                        help='Buffer size for tiling')
    parser.add_argument('-s', '--tile-size', type=int, default=40,
                        help='Height and Width of tiles')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Threshold for tiling')
    parser.add_argument('--test-frac', type=float, default=0.15,
                        help='Fraction of data for testing')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--val-fold', type=int, default=4,
                        help='Validation fold number')
    parser.add_argument('--base_model', type=str, 
                        default='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                        help='Base model from detectron2 model_zoo')
    parser.add_argument('-m', '--pretrained_model', type=str, 
                        default='models/pretrained/250312_flexi.pth',
                        help='Path to pre-trained model weights')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--eval-period', type=int, default=100,
                        help='Evaluation period')
    parser.add_argument('--max-iter', type=int, default=3000,
                        help='Maximum iterations')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--force-retile', action='store_true',
                        help='Force re-tiling even if tiles directory already exists')
    parser.add_argument('--dem', type=str, default='',
                        help='Path to DEM layer')
    parser.add_argument('--strict', action='store_true',
                        help='use strict train/test split')
    return parser.parse_args()


def prepare_data(args):
    """Prepare data by tiling and organizing into train/test folders"""
    train_dir = Path(args.train_dir)
    sites = [site_dir for site_dir in train_dir.iterdir() if site_dir.is_dir()]
    site_data = []

    for site_dir in sites:
        print(f"Processing site: {site_dir.stem}")
        
        appends = f"{args.tile_size}_{args.buffer}_{args.threshold}"
        tiles_dir = site_dir / f"tiles_{appends}"
    
        if tiles_dir.is_dir() and not args.force_retile:
            train_dir = tiles_dir / "train"
            if train_dir.exists():
                print(f"  Tiles directory already exists: {tiles_dir}")
                print(f"  Skipping tiling for this site (use --force-retile to override)")

        else:
            if tiles_dir.is_dir():
                shutil.rmtree(tiles_dir) 
            
            crown_path = next((site_dir / "crowns").glob("*.shp"))
            img_path = next((site_dir / "rgb").glob("*.tif"))
            crowns = gpd.read_file(crown_path)
            image = rasterio.open(img_path)

            if crowns.crs != image.crs:
                print(f"CRS mismatch. Transforming crowns from {crowns.crs} to {image.crs.data}")
                crowns = crowns.to_crs(image.crs.data) # ensure CRS match
            
            tile_data(img_path, tiles_dir, args.buffer, args.tile_size, args.tile_size, 
                    crowns, args.threshold, mode="rgb")
            
        to_traintest_folders(tiles_dir, tiles_dir, test_frac=args.test_frac, 
                                 folds=args.folds, strict=args.strict)
            
        site_data.append({"site_name": site_dir.stem,
                          "tiles_dir": tiles_dir,
                          "appends": appends})
    
    return site_data


def train_model(args, models_dir, site_data):
    train_datasets = []
    val_datasets = []

    for site_info in site_data:
        site_name = site_info["site_name"]
        train_location = site_info["tiles_dir"] / "train"
        register_train_data(train_location, site_name, val_fold=args.val_fold)

        train_datasets.append(f"{site_name}_train")
        val_datasets.append(f"{site_name}_val")
    
    file_io.g_pathmgr._DISABLE_TELEMETRY = True
    
    trains = tuple(train_datasets)
    tests = tuple(val_datasets)
    
    cfg = setup_cfg(args.base_model, trains, tests, args.pretrained_model, 
                   workers=args.workers, eval_period=args.eval_period, 
                   max_iter=args.max_iter, out_dir=str(models_dir),
                   resize="rand_fixed")
    
    trainer = MyTrainer(cfg, patience=args.patience)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg


def plot_metrics(models_dir, site_data):
    """Plot and save training metrics"""
    # Create plots directory
    plots_dir = models_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load metrics
    metrics_path = models_dir / 'metrics.json'
    experiment_metrics = load_json_arr(metrics_path)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 7))
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], 
        label='Total Validation Loss', color='red')
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], 
        label='Total Training Loss')
    
    plt.legend(loc='upper right')
    plt.title('Comparison of the training and validation loss of detectree2')
    plt.ylabel('Total Loss')
    plt.xlabel('Number of Iterations')
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_validation_loss.png', dpi=300)
    plt.close()
    
    # Plot AP50 metrics
    colors = plt.cm.tab10.colors

    for i, site_info in enumerate(site_data):
        site = site_info['site_name']
        site_metrics = [x for x in experiment_metrics if site + '_val/segm/AP50' in x]
        iterations = [x['iteration'] for x in site_metrics]
        ap50_values = [x[site + '_val/segm/AP50'] for x in site_metrics]

        
        plt.plot(iterations, 
                 ap50_values,
                 label=f'Site {site} Validation AP50',
                 color = colors[i % len(colors)],
                 marker = 'o',
                 linewidth = 2
        )
        
    plt.legend(loc="best")
    plt.title('Comparison of the training and validation loss of Mask R-CNN')
    plt.ylabel('AP50')
    plt.xlabel('Number of Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / 'val_fold_ap50.png', dpi=300)
    plt.close()
    
    print(f"Saved plots to {plots_dir}")


def main():
    args = parse_arguments()
    now = datetime.now().strftime('%y%m%d_%H')
    output_dir_name = args.output_dir if args.output_dir else f"{now}_models"
    models_dir = Path('models/finetuned') / output_dir_name

    site_data = prepare_data(args)
    cfg = train_model(args, models_dir, site_data)
    plot_metrics(models_dir, site_data)

    if Path("train_outputs").is_dir() and not any(Path("train_outputs").iterdir()):
        Path("train_outputs").rmdir()

    print(f"Fine-tuning completed. Results saved in {models_dir}")

    # Evaluation
    # Find the finalist model
    update_model = find_final_model(models_dir)
    print(f'\n Evaluating model {update_model}...')
    cfg = setup_cfg(update_model=update_model)
    for site_info in site_data:
        print(f"\n Evaluating site: {site_info['site_name']}")
        print(f"\n tiles dir: {site_info['tiles_dir']}")

        evaluate_model(cfg, site_info['tiles_dir'])


if __name__ == "__main__":
    os.environ["IOPATH_DISABLE_TELEMETRY"] = "1"
    main()
