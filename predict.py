#!/usr/bin/env python
"""
A script for running tree crown detection using Detectree2.
"""

import os
import argparse
import shutil
import glob

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns, post_clean
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
from secondary_cleaning import secondary_cleaning


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tree crown detection using Detectree2")
    
    parser.add_argument("-i", "--input-path", default="./data/predict", 
                        help="Path to the site data directory (default: ./data/predict)")
    parser.add_argument("-o", "--output-suffix", default="", 
                        help="Suffix to add to output filename (default: '')")
    parser.add_argument("-m", "--model-path", default="models/finetuned/250402_15_models/model_final.pth",
                        help="Path to the trained model")
    parser.add_argument("-s", "--tile-size", type=int, default=40, 
                        help="Width and Height of tiles (default: 40)")
    parser.add_argument("-b", "--buffer", type=int, default=30, 
                        help="Buffer around tiles (default: 30)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Confidence threshold for filtering crowns (default: 0.2)")
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
    input_path=args.input_path

    img_dir = os.path.join(input_path, "rgb")
    appends = f"{args.tile_size}_{args.buffer}"
    tiles_dir = os.path.join(input_path, f"tiles_pred_{appends}")
    predictions_geo_path = os.path.join(tiles_dir, "predictions_geo")
    output_file = os.path.join(input_path, f"crowns_out_{args.output_suffix}.gpkg")
    
    if not args.force_retile and os.path.isdir(tiles_dir):
        pass
    else:
        shutil.rmtree(tiles_dir) if os.path.isdir(tiles_dir) else None
        img_files = glob.glob(os.path.join(img_dir, "*.tif"))
        for img_path in img_files:
            print(f"Tiling image: {img_path}")
            tile_data(img_path, tiles_dir, args.buffer, args.tile_size, args.tile_size, dtype_bool=True)
    
    cfg = setup_cfg(update_model=args.model_path)
    predict_on_data(tiles_dir, predictor=DefaultPredictor(cfg))
    project_to_geojson(tiles_dir, 
                       os.path.join(tiles_dir, "predictions"),
                       predictions_geo_path)
    
    crowns = stitch_crowns(predictions_geo_path, 1)
    stitched_crowns = os.path.join(input_path, f"stitched_crowns_{args.output_suffix}.gpkg")
    crowns.to_file(stitched_crowns) # Temperarily save stitched crowns

    clean = clean_crowns(crowns, args.intersection, confidence=args.confidence)
    clean = clean.set_geometry(clean.simplify(args.simplify)) 
    os.remove(stitched_crowns) if os.path.exists(stitched_crowns) else None

    clean_2 = secondary_cleaning(clean)
    clean_3 = post_clean(unclean_df=crowns, clean_df=clean_2)
    clean_3.to_file(output_file)

    print(f"Done predicting. Results saved in {output_file}")

    
def main():
    args = parse_arguments()
    process_site(args)


if __name__ == "__main__":
    main()
