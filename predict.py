#!/usr/bin/env python
"""
A script for running tree crown detection using Detectree2.
#TODO: rewrite tile_data so it can use parellel processing
"""

import argparse
import shutil
import logging
from pathlib import Path
import geopandas as gpd
from datetime import datetime

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns, post_clean
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor

from utility import find_final_model, secondary_cleaning

logging.getLogger("detectree2.preprocessing.tiling").setLevel(logging.ERROR)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tree crown detection using Detectree2")
    
    parser.add_argument("-i", "--input-path", required=True, 
                        help="Path to the site data directory")
    parser.add_argument("-o", "--output-suffix", default="", 
                        help="Suffix to add to output filename (default: '')")
    parser.add_argument("-m", "--model-path", default="models/pretrained/250312_flexi.pth",
                        help="Path to the trained model")
    parser.add_argument("-s", "--tile-size", type=int, default=40, 
                        help="Width and Height of tiles (default: 40)")
    parser.add_argument("-b", "--buffer", type=int, default=30, 
                        help="Buffer around tiles (default: 30)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Confidence threshold for filtering crowns (default: 0.2)")
    parser.add_argument("--min-area", type=float, default=2,
                        help="Minimum area of crowns to be retained. (default: 2 m^2)")    
    parser.add_argument("--simplify", type=float, default=0.3,
                        help="Tolerance for simplifying crown geometries (default: 0.3). The higher this value, the smaller the number of vertices in the resulting geometry.")
    parser.add_argument("--intersection", type=float, default=0.5,
                        help="Threshold for crown intersection (default: 0.5)")
    parser.add_argument('--force-retile', action='store_true',
                        help='Force re-tiling even if tiles directory already exists')
    return parser.parse_args()
    

def process_site(args) -> None:
    """
    Process a site for tree crown detection.
    """
    input_path = Path(args.input_path)
    img_dir = input_path / "rgb"
    tiles_dir = input_path / f"tiles_pred_{args.tile_size}_{args.buffer}"
    predictions_json_path = tiles_dir / "predictions"
    predictions_geojson_path = tiles_dir / "predictions_geo"
    model_path = find_final_model(args.model_path)

    if not args.output_suffix:
        timestamp = datetime.now().strftime("%y%m%d_%H")
        output_file = input_path / f"crowns_out_{timestamp}.gpkg"
    else:
        output_file = input_path / f"crowns_out_{args.output_suffix}.gpkg"

    # Step 1: Tiling
    if not args.force_retile and tiles_dir.is_dir():
        pass
    else:
        if tiles_dir.is_dir():
            shutil.rmtree(tiles_dir)
        img_files = list(img_dir.glob("*.tif"))
        for img_path in img_files:
            print(f"Tiling image: {img_path}")
            tile_data(img_path, tiles_dir, args.buffer, args.tile_size, args.tile_size, dtype_bool=True)
    
    # Step 2: Predicting
    cfg = setup_cfg(update_model=model_path)
    predict_on_data(tiles_dir, predictor=DefaultPredictor(cfg))
    project_to_geojson(tiles_dir, 
                       predictions_json_path,
                       predictions_geojson_path)
    shutil.rmtree(predictions_json_path, ignore_errors=True) # Use this to save disk space 
    
    # Step 3: Stitching
    crowns = stitch_crowns(predictions_geojson_path, 1)
    stitched_crowns = input_path / f"stitched_crowns_{args.output_suffix}.gpkg"
    crowns.to_file(stitched_crowns) # Temperarily save stitched crowns
    shutil.rmtree(predictions_geojson_path, ignore_errors=True) # Use this to save disk space 

    # Step 4: Cleaning duplicates
    clean = clean_crowns(crowns, args.intersection, confidence=args.confidence, area_threshold=args.min_area)
    clean = clean.set_geometry(clean.simplify(args.simplify)) 
    stitched_crowns.unlink(missing_ok=True)

    clean_2 = secondary_cleaning(clean)
    clean_2.to_file(output_file)

    print(f"Done predicting. Results saved in {output_file}")

    
def main():
    args = parse_arguments()
    process_site(args)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")