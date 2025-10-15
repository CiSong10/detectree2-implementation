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
        - crowns # crowns_dir
            - crowns.shp
        - rgb # img_dir
            - rgb.tif
        - tiles_{self.appends}  # tiles_root
            - tiles # tiles_dir
                - t_1.tif
                - t_1.geojson
                - t_1.png
                - t_2. ...
            - train # train_dir
                - fold_1
                - ...
            - test # test_dir
            - predictions # predict_dir
            - predictions_geo # predict_geojson_dir
    - site_2

- models
    - pretrained
    - finetuned
        - configs.output_dir
            - xxx.pth
"""

class Site:
    def __init__(self, path: Path, appends: str, configs: Configs):
        self.path = path
        self.name = path.stem
        self.crowns_dir = path / "crowns"
        self.img_dir = path / "rgb"
        self.tiles_root = path / f"tiles_{appends}"
        self.tiles_dir = self.tiles_root / "tiles"
        self.train_dir = self.tiles_root / "train"
        self.test_dir = self.tiles_root / "test"
        self.predict_dir = self.tiles_root / "predictions"
        self.predict_geojson_dir = self.tiles_root / "predictions_geo"
        self.crowns_out_file = path / f"{self.name}_prediction_{configs.model}.gpkg"

        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

    def tile_data(self):
        """
        Tile data. Skip if tiles already exists, unless force_retile config is True.

        Args:
            sites (str | Path | list[str | Path] | None): 
                One or more site directories. If None, uses self.sites.
        """
        if self.tiles_dir.is_dir() and any(self.tiles_dir.iterdir()) and not self.configs.force_retile:
            self.logger.debug(f"Tiles already exist for site {self.name}, skipping tiling.")
        else:
            if self.tiles_root.exists():
                shutil.rmtree(self.tiles_root)

            # Allows a imgs_dir with multiple imgs & only one crown_path 
            # Saves tiles into one tiles_dir
            imgs_dir = list(self.img_dir.glob("*.tif"))
            
            try: 
                crown_path = next(self.crowns_dir.glob("*.shp")) # Allow crown_path not exist, just simply tile with crowns = None
                crowns = gpd.read_file(crown_path)
                # Check projection
                # with rasterio.open(imgs_dir[0]) as img:
                #     if crowns.crs != img.crs:
                #         self.logger.warning(f"CRS mismatch. Transforming crowns from {crowns.crs} to {img.crs.data}")
                #         crowns = crowns.to_crs(img.crs.data) # ensure CRS match
            except StopIteration:
                crowns = None

            for img_path in imgs_dir:
                tile_data(img_path, self.tiles_dir,
                          self.configs.buffer, self.configs.tile_size, self.configs.tile_size,
                          crowns, 
                          self.configs.threshold, 
                          mode=self.configs.mode,
                          tile_placement=self.configs.tile_placement,
                          multithreaded = True) 

    def train(self):
        pass

    def evaluate(self, cfg, dem=None):
        """Evaluate a model on tiled data.
        
        Args:
            cfg: Model configuration
            tiles_root (str): Directory containing tiles
            dem (str, optional): Path to DEM layer
            
        Returns:
            Tuple[float, float, float]: Precision, recall, and F1 score
        """

        if self.predict_dir.exists():
            shutil.rmtree(self.predict_dir, True)
        if Path('./eval').exists():
            shutil.rmtree('./eval', True)

        predictions_on_data(self.tiles_root, DefaultPredictor(cfg))
        to_eval_geojson(self.predict_dir)

        prec, recall, f1 = site_f1_score2(
            tile_directory=self.tiles_dir, 
            test_directory=self.test_dir,
            pred_directory=self.predict_dir,
            lidar_img=dem,
            IoU_threshold= 0.5,
            border_filter=[False, 1],
            conf_threshold=0.2,
            area_threshold=16,
        )

        return prec, recall, f1

    def predict(self):
        self.logger.info(f"[{self.name}] Predicting...")

        # step 0: Tiling (already done)
        # step 1: predicting
        model_path = find_final_model(f"models/finetuned/{self.configs.model}")
        cfg = setup_cfg(update_model=model_path)
        predict_on_data(self.tiles_dir, self.predict_dir, predictor=DefaultPredictor(cfg))
        
        project_to_geojson(self.tiles_dir,
                            self.predict_dir,
                            self.predict_geojson_dir)
        shutil.rmtree(self.predict_dir, ignore_errors=True) # Use this to save disk space 

        # step 2: stitching
        crowns = stitch_crowns(self.predict_geojson_dir, 2)
        shutil.rmtree(self.predict_geojson_dir, ignore_errors=True) # Use this to save disk space 

        # step 3: cleaning duplicates
        clean = clean_crowns(crowns, self.configs.intersection, confidence=self.configs.confidence, area_threshold=self.configs.min_area)
        clean = clean.set_geometry(clean.simplify(self.configs.simplify)) 
        clean_2 = secondary_cleaning(clean)
        clean_2.to_file(self.crowns_out_file)

        self.logger.info(f"[{self.name}] Done predicting. Saved: {self.crowns_out_file}")


class Pipeline:
    def __init__(self, configs: Configs):
        self.configs = configs

        self.appends = f"{self.configs.tile_size}_{self.configs.buffer}_{self.configs.threshold}"
        self.model_dir = Path("models/finetuned") / (
            configs.model or datetime.now().strftime("%y%m%d_%H")
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(configs.data, str):
            configs.data = [configs.data]
        self.sites = [
            Site(Path("data") / site, self.appends, configs)
            for site in configs.data
        ]
        
        for site in self.sites:
            site.tile_data()

        # random.seed(configs.seed)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self, sample_n_tiles=None):

        train_datasets, val_datasets = [], []

        for site in self.sites:
            to_traintest_folders(site.tiles_dir, site.tiles_root, test_frac=self.configs.test_frac,
                                 folds=self.configs.folds, strict=self.configs.strict)
            register_train_data(site.train_dir, site.name, val_fold=self.configs.val_fold)
            train_datasets.append(f"{site.name}_train")
            val_datasets.append(f"{site.name}_val")

        cfg = setup_cfg(
            self.configs.base_model, tuple(train_datasets), tuple(val_datasets),
            self.configs.pretrained_model, self.configs.workers, 
            eval_period=self.configs.eval_period, max_iter=self.configs.max_iter,
            out_dir=str(self.model_dir), resize=self.configs.resize
        )
        trainer = MyTrainer(cfg, self.configs.patience)
        trainer.resume_or_load(resume=self.configs.resume)
        trainer.train()

        self.logger.info(f"Fine-tuning completed. Results saved in {self.model_dir}")
        
        self._plot_metrics()
        
        return cfg
    
    def _plot_metrics(self):
        plots_dir = self.model_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Load metrics
        metrics_path = self.model_dir / "metrics.json"
        experiment_metrics = load_json_arr(metrics_path)

        # Plot training and validation loss
        plt.figure()
        plt.plot(
            [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
            [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], 
            label='Total Validation Loss', color='red'
            )
        plt.plot(
            [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
            [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], 
            label='Total Training Loss'
            )
        plt.legend(loc='upper right')
        plt.title('Comparison of the training and validation loss')
        plt.ylabel('Total Loss')
        plt.xlabel('Number of Iterations')
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_validation_loss.png', dpi=300)
        plt.close()

        # plot AP50 metrics
        colors = plt.cm.tab10.colors

        for i, site in enumerate(self.sites):
            site_name = site.name
            site_metrics = [x for x in experiment_metrics if f'{site_name}_val/segm/AP50' in x]
            iterations = [x['iteration'] for x in site_metrics]
            ap50_values = [x[f"{site_name}_val/segm/AP50"] for x in site_metrics]

            plt.plot(
                iterations, 
                ap50_values,
                label=f'Site {site_name}',
                color = colors[i % len(colors)],
                marker = 'o',
                linewidth = 2
                )
        
        plt.legend(loc="best")
        plt.title('Comparison of the training and validation loss of Mask R-CNN')
        plt.ylabel('Validation AP50')
        plt.xlabel('Number of Iterations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / 'val_fold_ap50.png', dpi=300)
        plt.close()
    
        self.logger.info(f"Saved plots to {plots_dir}")
        

        pass

    def predict(self, parallel=False):
        for site in self.sites:
            site.predict()

    def evaluate(self, test_site: Path | str = None):
        model_path = find_final_model(self.model_dir)
        cfg = setup_cfg(update_model=model_path)

        eval_results = {}

        if not test_site:
            # right now it works for testing the "test" site of training
            for site in self.sites:
                prec, recall, f1 = site.evaluate(cfg)
                eval_results[site.name] = {'precision': prec, 'recall': recall, 'f1': f1}
        else:
            test_site = Site(Path(test_site), self.appends, self.configs)
            test_site.tile_data()
            to_traintest_folders(test_site.tiles_dir, test_site.tiles_root, test_frac=1, # everyything goes to the test_dir
                                 folds=self.configs.folds, strict=self.configs.strict)
            prec, recall, f1 = test_site.evaluate(cfg)
            eval_results[test_site.stem] = {'precision': prec, 'recall': recall, 'f1': f1}

        return eval_results
