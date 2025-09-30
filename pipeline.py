import logging
import shutil
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import rasterio
from detectree2.preprocessing.tiling import tile_data, to_traintest_folders, image_details, is_overlapping_box
from detectree2.models.train import (register_train_data, MyTrainer, setup_cfg, 
                                     load_json_arr, predictions_on_data)
from detectree2.models.outputs import project_to_geojson, to_eval_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.evaluation import site_f1_score2
from detectron2.engine import DefaultPredictor
from configs import Configs

from utility import (find_final_model, secondary_cleaning, to_traintest_folders_sample, 
                     safe_register_train_data, parallel_predict_on_data)

""" data structure

- data # data_dir
    - site_1 # site_dir
        - crowns
        - rgb # img_dir
        - tiles_{self.appends}  # tiles_root
            - tiles # tiles_dir
                - t_1.tif
                - t_1.geojson
                - t_1.png
                - t_2. ...
            - train # train_location
                - fold_1
                - ...
            - test
            - predictions
            - predictions_geo
    - site_2

- models
    - pretrained
    - finetuned
        - configs.output_dir
            - xxx.pth
"""


class Pipeline:
    def __init__(self, configs: Configs):
        self.configs = configs

        self.appends = f"{self.configs.tile_size}_{self.configs.buffer}_{self.configs.threshold}"
        self.model_dir = Path("models/finetuned") / (
            configs.model or datetime.now().strftime("%y%m%d_%H")
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(configs.data, str):
            data_sites = [configs.data]
        self.sites = [Path('data') / site for site in data_sites]

        random.seed(configs.seed)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._tile_data()

    def _tile_data(self):
        """
        Tile data. Skip if tiles already exists, unless force_retile config is True.

        Args:
            sites (str | Path | list[str | Path] | None): 
                One or more site directories. If None, uses self.sites.
        """

        for site_dir in self.sites:
            tiles_root = site_dir / f"tiles_{self.appends}"
            tiles_dir = tiles_root / "tiles"

            if tiles_root.is_dir() and any(tiles_root.iterdir()) and not self.configs.force_retile:
                self.logger.info(f"Tiles already exist: {tiles_root}, skipping tiling.")
            else:
                if tiles_root.is_dir():
                    shutil.rmtree(tiles_root)

                # Allows a imgs_dir with multiple imgs & only one crown_path 
                # Saves tiles into one tiles_root
                imgs_dir = list((site_dir / "rgb").glob("*.tif"))
                
                try: 
                    crown_path = next((site_dir / "crowns").glob("*.shp")) # Allow crown_path not exist, just simply tile with crowns = None
                    crowns = gpd.read_file(crown_path)
                    # Check projection
                    with rasterio.open(imgs_dir[0]) as img:
                        if crowns.crs != img.crs:
                            self.logger.warning(f"CRS mismatch. Transforming crowns from {crowns.crs} to {img.crs.data}")
                            crowns = crowns.to_crs(img.crs.data) # ensure CRS match
                except StopIteration:
                    crowns = None

                for img_path in imgs_dir:
                    tile_data(img_path, tiles_dir, 
                              self.configs.buffer, self.configs.tile_size, self.configs.tile_size,
                              crowns, self.configs.threshold, mode="rgb",
                              multithreaded = True) # TODO: configurable

    def train(self, sample_n_tiles=None):
    
        self._tile_data()

        for site_dir in self.sites:
            tiles_root = site_dir / f"tiles_{self.appends}"
            tiles_dir = tiles_root / "tiles"

            # TODO: edit to_traintest_folders so that it can select only a few samples (sample_n_tiles) to train test folders
            to_traintest_folders_sample(tiles_dir, tiles_root, test_frac=self.configs.test_frac, 
                                        folds=self.configs.folds, strict=self.configs.strict, 
                                        sample_n_tiles=sample_n_tiles)


        train_datasets, val_datasets = [], []
        for site_dir in self.sites:
            site_name = site_dir.stem
            tiles_root = site_dir / f"tiles_{self.appends}"
            train_location = tiles_root / "train"
            safe_register_train_data(train_location, site_name, val_fold=self.configs.val_fold)

            train_datasets.append(f"{site_name}_train")
            val_datasets.append(f"{site_name}_val")


        cfg = setup_cfg(
            self.configs.base_model, tuple(train_datasets), tuple(val_datasets),
            self.configs.pretrained_model, self.configs.workers, 
            eval_period=self.configs.eval_period, max_iter=self.configs.max_iter,
            out_dir=str(self.model_dir), resize=self.configs.resize
        )
        trainer = MyTrainer(cfg, self.configs.patience)
        trainer.resume_or_load(resume=False)
        trainer.train()
        return cfg

    def predict(self, parallel=False):
        for site_dir in self.sites:
            tiles_root = site_dir / f"tiles_{self.appends}"
            tiles_dir = tiles_root / "tiles"
            predictions_json_path = tiles_root / "predictions"
            predictions_geojson_path = tiles_dir / "predictions_geo"
            crowns_out_file = site_dir / f"crowns_out_{self.configs.model}.gpkg"

            # step 0: Tiling (already done)
            # step 1: predicting
            model_path = find_final_model(self.model_dir)
            cfg = setup_cfg(update_model=model_path)
            if parallel:
                parallel_predict_on_data(tiles_dir, cfg=cfg)
            else:
                predict_on_data(tiles_dir, predictor=DefaultPredictor(cfg))
            
            project_to_geojson(tiles_dir,
                               predictions_json_path,
                               predictions_geojson_path)
            shutil.rmtree(predictions_json_path, ignore_errors=True) # Use this to save disk space 

            # step 2: stitching
            crowns = stitch_crowns(predictions_geojson_path, 2)
            # stitched_crowns = site_dir / f"stitched_crowns_{self.configs.model}.gpkg"
            # crowns.to_file(stitched_crowns) # Temperarily save stitched crowns
            shutil.rmtree(predictions_geojson_path, ignore_errors=True) # Use this to save disk space 

            # step 3: cleaning duplicates
            clean = clean_crowns(crowns, self.configs.intersection, confidence=self.configs.confidence, area_threshold=self.configs.min_area)
            clean = clean.set_geometry(clean.simplify(self.configs.simplify)) 
            # stitched_crowns.unlink(missing_ok=True)
            clean_2 = secondary_cleaning(clean)
            clean_2.to_file(crowns_out_file)

            self.logger.info(f"Done predicting. Results saved in {crowns_out_file}")

    def evaluate(self, test_site: Path | str = None):
        model_path = find_final_model(self.model_dir)
        cfg = setup_cfg(update_model=model_path)

        eval_results = {}

        if not test_site:
            # right now it works for testing the "test" site of training
            for site_dir in self.sites:
                tiles_root = site_dir / f"tiles_{self.appends}"
                prec, recall, f1 = self._evaluate_model(cfg, tiles_root)
                eval_results[site_dir.stem] = {'precision': prec, 'recall': recall, 'f1': f1}
        else:
            # TODO: support an independent test site
            # maybe just be the same -- if so edit the if else structure

            # need tiling before
            test_site = Path(test_site)
            self._tile_data(test_site)
            test_tiles_root = test_site / f'tiles_{self.appends}'
            prec, recall, f1 = self._evaluate_model(cfg, test_tiles_root)
            eval_results[test_site.stem] = {'precision': prec, 'recall': recall, 'f1': f1}

        return eval_results

    def _evaluate_model(self, cfg, tiles_root, dem=None):
        """Evaluate a model on tiled data.
        
        Args:
            cfg: Model configuration
            tiles_root (str): Directory containing tiles
            dem (str, optional): Path to DEM layer
            
        Returns:
            Tuple[float, float, float]: Precision, recall, and F1 score
        """
        pred_folder = tiles_root / "predictions"
        if pred_folder.exists():
            shutil.rmtree(pred_folder, True)
        if Path('./eval').exists():
            shutil.rmtree('./eval', True)

        predictions_on_data(tiles_root, DefaultPredictor(cfg))
        to_eval_geojson(pred_folder)

        prec, recall, f1 = site_f1_score2(
            tile_directory=tiles_root / "tiles", 
            test_directory=tiles_root / "test",
            pred_directory=pred_folder,
            lidar_img=dem,
            IoU_threshold= 0.5,
            border_filter=[False, 1],
            conf_threshold=0.2,
            area_threshold=16,
        )

        return prec, recall, f1

