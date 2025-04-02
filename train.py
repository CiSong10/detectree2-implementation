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
# from detectree2.models.outputs import to_eval_geojson
# from detectree2.models.evaluation import site_f1_score2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
# from detectron2.engine import DefaultPredictor
import iopath.common.file_io as file_io
from glob import glob

from evaluate import evaluate_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='DetecTree2 Fine-tuning Script')
    parser.add_argument('-i', '--train-dir', type=str, default='./data/train/',
                        help='Root training directory containing all site folders')
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
    parser.add_argument('--val-fold', type=int, default=5,
                        help='Validation fold number')
    parser.add_argument('--base_model', type=str, 
                        default='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                        help='Base model from detectron2 model_zoo')
    parser.add_argument('-m', '--pretrained_model', type=str, 
                        default='./models/pretrained/250312_flexi.pth',
                        help='Path to pre-trained model weights')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of workers')
    parser.add_argument('--eval-period', type=int, default=100,
                        help='Evaluation period')
    parser.add_argument('--max-iter', type=int, default=3000,
                        help='Maximum iterations')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--force-retile', action='store_true',
                        help='Force re-tiling even if tiles directory already exists')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Custom output models directory name')
    parser.add_argument('--dem', type=str, default='./data/DEM_m.tif',
                        help='Path to DEM layer')
    return parser.parse_args()


def prepare_data(args):
    """Prepare data by tiling and organizing into train/test folders"""
    sites = [d for d in os.listdir(args.train_dir)
             if os.path.isdir(os.path.join(args.train_dir, d))]
    site_data = []

    for site in sites:
        print(f"Processing site: {site}")
        site_dir = os.path.join(args.train_dir, site)
        
        appends = f"{args.tile_size}_{args.buffer}_{args.threshold}"
        tiles_dir = os.path.join(site_dir, f"tiles_{appends}")
    
        if os.path.isdir(tiles_dir) and not args.force_retile:
            train_dir = os.path.join(tiles_dir, "train")
            if os.path.exists(train_dir):
                print(f"  Tiles directory already exists: {tiles_dir}")
                print(f"  Skipping tiling for this site (use --force-retile to override)")
            else:
                to_traintest_folders(tiles_dir, tiles_dir, test_frac=args.test_frac, 
                                     folds=args.folds, strict=False)
        else:
            shutil.rmtree(tiles_dir) if os.path.isdir(tiles_dir) else None
            
            crown_path = glob(os.path.join(site_dir, "crowns", "*.shp"))[0]
            img_path = glob(os.path.join(site_dir, "rgb", "*.tif"))[0]
            crowns = gpd.read_file(crown_path)
            image = rasterio.open(img_path)

            if crowns.crs != image.crs:
                print(f"CRS mismatch. Transforming crowns from {crowns.crs} to {image.crs.data}")
                crowns = crowns.to_crs(image.crs.data) # ensure CRS match
            
            tile_data(img_path, tiles_dir, args.buffer, args.tile_size, args.tile_size, 
                    crowns, args.threshold, mode="rgb")
            to_traintest_folders(tiles_dir, tiles_dir, test_frac=args.test_frac, 
                                 folds=args.folds, strict=False)
            
        site_data.append({"site_name": site,
                          "tiles_dir": tiles_dir,
                          "appends": appends})
    
    return site_data


def train_model(args, site_data):
    train_datasets = []
    val_datasets = []

    for site_info in site_data:
        site_name = site_info["site_name"]
        train_location = os.path.join(site_info["tiles_dir"], "train")
        register_train_data(train_location, site_name, val_fold=args.val_fold)

        train_datasets.append(f"{site_name}_train")
        val_datasets.append(f"{site_name}_val")
    
    file_io.g_pathmgr._DISABLE_TELEMETRY = True
    
    now = datetime.now().strftime('%y%m%d_%H')
    models_dir = args.output_dir if args.output_dir else f"{now}_models"
    models_dir = os.path.join('finetuned_models', models_dir)
    
    trains = tuple(train_datasets)
    tests = tuple(val_datasets)
    
    cfg = setup_cfg(args.base_model, trains, tests, args.pretrained_model, 
                   workers=args.workers, eval_period=args.eval_period, 
                   max_iter=args.max_iter, out_dir=models_dir,
                   resize="rand_fixed")
    
    trainer = MyTrainer(cfg, patience=args.patience)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg


def plot_metrics(models_dir, site_data):
    """Plot and save training metrics"""
    # Create plots directory
    plots_dir = os.path.join(models_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load metrics
    metrics_path = os.path.join(models_dir, 'metrics.json')
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
    plt.savefig(os.path.join(plots_dir, 'training_validation_loss.png'), dpi=300)
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
    plt.savefig(os.path.join(plots_dir, 'val_fold_ap50.png'), dpi=300)
    plt.close()
    
    print(f"Saved plots to {plots_dir}")


def find_final_model(output_dir):
    model_final_path = os.path.join(output_dir, 'model_final.pth')
    if os.path.isfile(model_final_path):
        return model_final_path
    else:
        model_files = glob(os.path.join(output_dir, 'model_*.pth'))
        max_suffix = -1
        selected_model = None

        for model_file in model_files:
            try:
                # Extract numerical suffix from filename
                filename = os.path.basename(model_file)
                suffix = int(filename.split('_')[1].split('.')[0])
                if suffix > max_suffix:
                    max_suffix = suffix
                    selected_model = model_file
            except (IndexError, ValueError):
                # Skip files that don't match the expected pattern
                continue
    
        if selected_model is None:
            raise FileNotFoundError(f"No valid model found in {output_dir}")

        return selected_model


def main():
    args = parse_arguments()
    site_data = prepare_data(args)
    cfg = train_model(args, site_data)
    plot_metrics(cfg.OUTPUT_DIR, site_data)
    os.rmdir('train_outputs') if os.path.isdir('train_outputs') and len(os.listdir('train_outputs'))==0 else None
    print(f"Fine-tuning completed. Results saved in {cfg.OUTPUT_DIR}")

    # Evaluation
    # Find the finalist model
    update_model = find_final_model(cfg.OUTPUT_DIR)
    print(f'\n Evaluating model {update_model}...')
    cfg = setup_cfg(update_model=update_model)
    for site_info in site_data:
        print(f"\n Evaluating site: {site_info['site_name']}")
        print(f"\n tiles dir: {site_info['tiles_dir']}")

        evaluate_model(cfg, site_info['tiles_dir'], args.dem)


if __name__ == "__main__":
    main()
