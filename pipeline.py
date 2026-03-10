import logging
import shutil
import re
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from detectree2.preprocessing.tiling import tile_data, to_traintest_folders, image_details, is_overlapping_box
from detectree2.models.train import (register_train_data, MyTrainer, setup_cfg, load_json_arr,
                                     predictions_on_data, get_latest_model_path, multiply_conv1_weights, 
                                     FlexibleDatasetMapper)
from detectree2.models.outputs import project_to_geojson, to_eval_geojson, stitch_crowns, clean_crowns, post_clean
from detectree2.models.predict import predict_on_data
from detectree2.models.evaluation import site_f1_score2
from detectron2.engine import DefaultPredictor
from configs import Configs

from utility import (secondary_cleaning, safe_register_train_data,
                     stitch_crowns_parallel, project_to_geojson_parallel, canopy_mask_filter, 
                     MultiBandPredictor)


""" data structure

- data # data_dir
    - site_1 # site_dir
        - crowns # crowns_dir
            - crowns.shp
        - rgb # img_dir
            - rgb.tif
        - mask
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
        self.img_dir = path / "rgb" if configs.mode=="rgb" else path / "ms"
        self.tiles_root = path / f"tiles_{appends}"
        self.tiles_dir = self.tiles_root / "tiles"
        self.train_dir = self.tiles_root / "train"
        self.test_dir = self.tiles_root / "test"
        self.predict_dir = self.tiles_root / "predictions"
        self.predict_geojson_dir = self.tiles_root / "predictions_geo"
        self.crowns_out_file = path / f"{self.name}_prediction.gpkg"

        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.configs.override_img_dir:
            self.img_dir = path / self.configs.override_img_dir

    def tile(self):
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
                crown_path = next(self.crowns_dir.glob("*.gpkg")) # Allow crown_path not exist, just simply tile with crowns = None
                crowns = gpd.read_file(crown_path)
                # Check projection
                # with rasterio.open(imgs_dir[0]) as img:
                #     if crowns.crs != img.crs:
                #         self.logger.warning(f"CRS mismatch. Transforming crowns from {crowns.crs} to {img.crs.data}")
                #         crowns = crowns.to_crs(img.crs.data) # ensure CRS match
            except StopIteration:
                crowns = None

            for img_path in imgs_dir:
                tile_data(img_path, 
                          self.tiles_dir,
                          self.configs.buffer, 
                          self.configs.tile_size, self.configs.tile_size,
                          crowns, 
                          threshold=self.configs.threshold, 
                          nan_threshold=self.configs.nan_threshold,
                          mode=self.configs.mode,
                          tile_placement=self.configs.tile_placement,
                          mask_path=None, # This can be added. No tiles will be created outside of mask
                          multithreaded = False,
                          additional_nodata = [255],
                          overlapping_tiles=True,
                          ignore_bands_indices = self.configs.ignore_bands_indices,
                          use_convex_mask=False,
                          enhance_rgb_contrast=True) 

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

        #######################
        # self.logger.info(f"Evaluating Site {self.name}")
        # if self.configs.mode == "rgb":
        #     predictor = DefaultPredictor(cfg)
        # elif self.configs.mode == "ms":
        #     predictor = MultiBandPredictor(cfg)
        # predictions_on_data(self.tiles_root, predictor)

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

    def predict(self, cfg):
        self.logger.info(f"[{self.name}] Predicting...")

        # step 0: Tiling (already done)
        # step 1: predicting

        if self.configs.mode == "rgb":
            predictor = DefaultPredictor(cfg)
        elif self.configs.mode == "ms":
            predictor = MultiBandPredictor(cfg)

        predict_on_data(self.tiles_dir, self.predict_dir, predictor=predictor)

        project_to_geojson_parallel(self.tiles_dir,
                                    self.predict_dir,
                                    self.predict_geojson_dir,
                                    max_workers=self.configs.workers)

        # step 2: stitching
        crowns = stitch_crowns_parallel(self.predict_geojson_dir, shift=1, max_workers=self.configs.workers)
        try: 
            canopy_mask_path = next((self.path / "mask").glob("*.tif"))
            crowns = canopy_mask_filter(crowns, canopy_mask_path)
        except StopIteration:
            self.logger.info(f'[{self.name}] Did not find Canopy Mask in {str(self.path / "mask")}')

        crowns.to_file(self.crowns_out_file, driver="GPKG", layer=f"{self.name}_{self.configs.model}_unclean")

        # step 3: cleaning duplicates
        clean = clean_crowns(crowns, self.configs.intersection, confidence=self.configs.confidence, area_threshold=self.configs.min_area)
        clean = clean.set_geometry(clean.simplify(self.configs.simplify))

        clean_2 = secondary_cleaning(clean)
        clean_2.to_file(self.crowns_out_file, driver="GPKG", layer=f"{self.name}_{self.configs.model}_final")

        # Fill in the gaps left by the cleaning
        clean_3 = post_clean(unclean_df=crowns, clean_df=clean_2)
        clean_3 = secondary_cleaning(clean_3)
        clean_3.to_file(self.crowns_out_file, driver="GPKG", layer=f"{self.name}_{self.configs.model}_clean3")

        self.logger.info(f"[{self.name}] Done predicting. Saved: {self.crowns_out_file}")

    def large_prediction(self, chunk_size=10000):
        import pandas as pd
        from shapely.geometry import box
        from tqdm import tqdm
        import fiona

        gpkg_path = self.path / "crowns_clean.gpkg"   # Single multi-layer geopackage
    
        # # Phase 1: Chunked Stitch + Clean
        if gpkg_path.exists(): gpkg_path.unlink() 
        geojson_files = list(self.predict_geojson_dir.glob("*.geojson"))
        canopy_mask_path = next((self.path / "mask").glob("*.tif"))
        print(f"Phase 1: Found {len(geojson_files)} files. Processing in chunks of {chunk_size}...")

        for i in range(0, len(geojson_files), chunk_size):
            chunk = geojson_files[i: i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{len(geojson_files)//chunk_size + 1}")
            stitched_crowns = stitch_crowns_parallel(files=chunk)
            crowns = canopy_mask_filter(stitched_crowns, canopy_mask_path)
            crowns = crowns.set_geometry(crowns.simplify(self.configs.simplify))
            clean = clean_crowns(crowns, self.configs.intersection, self.configs.confidence, self.configs.min_area)
            clean2 = secondary_cleaning(clean)
            clean2.to_file(gpkg_path, layer=f"chunk_{i}", driver="GPKG")
        
        # Phase 2: Handle Edge Cases Only
        # Compute tile boundaries automatically from GPKG layer extents
        print("Phase 2: Handling edge overlaps...")
        tile_bounds = []
        layers = fiona.listlayers(gpkg_path)
        for layer in layers:
            g = gpd.read_file(gpkg_path, layer=layer, rows=1)
            tile_bounds.append(box(*g.total_bounds))
        boundaries_gdf = gpd.GeoDataFrame(geometry=tile_bounds, crs=g.crs)
        buffers = boundaries_gdf.buffer(35)
        
        merged = []
        for buff in buffers:
            local = gpd.read_file(gpkg_path, bbox=buff.bounds)
            if len(local) == 0: 
                continue
            local = clean_crowns(local, self.configs.intersection, self.configs.confidence, self.configs.min_area)
            merged.append(local)
        
        if merged:
            merged = gpd.GeoDataFrame(pd.concat(merged, ignore_index=True), crs=g.crs)
        else:
            merged = gpd.GeoDataFrame(columns=g.columns, crs=g.crs)
        
        print(f"Edge features fixed: {len(merged)}")

        # Phase 3: Final Merge and output as gpkg
        print("Phase 3: Final merge.")
        buffer_union = gpd.GeoSeries(buffers.union_all(), crs=g.crs)
        final_parts = []
        for layer in tqdm(layers, desc="Final merge"):
            g = gpd.read_file(gpkg_path, layer=layer)
            mask = ~g.intersects(buffer_union.iloc[0])
            g = g.loc[mask]
            final_parts.append(g)
        final = gpd.GeoDataFrame(pd.concat(final_parts + [merged], ignore_index=True), crs=g.crs)
        final.to_file(self.crowns_out_file, driver="GPKG")
        print(f"Final GPKG saved: {self.crowns_out_file}")


class Pipeline:
    def __init__(self, configs: Configs):
        self.configs = configs

        self.appends = f"{configs.tile_size}_{configs.buffer}_{configs.threshold}"
        self.model_dir = configs.model_dir

        self.sites = [
            Site(Path("data") / site, self.appends, configs)
            for site in configs.data
        ]
        
        for site in self.sites:
            site.tile()

        # random.seed(configs.seed)
        self.logger = logging.getLogger(self.__class__.__name__)

        if configs.mode == "ms":
            self.num_bands = 4
            # sample_raster = next(self.sites[0].img_dir.glob("*.tif"))
            # with rasterio.open(sample_raster) as r:
            # self.num_bands = r.count - len(self.configs.ignore_bands_indices)
        else:
            self.num_bands = 3

    def train(self):
        
        if Path('./eval').exists():
            shutil.rmtree('./eval', True)

        train_datasets, val_datasets = [], []

        for site in self.sites:
            to_traintest_folders(site.tiles_dir, site.tiles_root, test_frac=self.configs.test_frac,
                                 folds=self.configs.folds, strict=self.configs.strict)
            safe_register_train_data(site.train_dir, site.name, val_fold=1)
            train_datasets.append(f"{site.name}_train")
            val_datasets.append(f"{site.name}_val")
        
        cfg = setup_cfg(
            self.configs.base_model, 
            tuple(train_datasets), 
            tuple(val_datasets),
            self.configs.pretrained_model, 
            self.configs.workers, 
            gamma=0.1,
            backbone_freeze=3,
            base_lr=0.0003389,
            max_iter=self.configs.max_iter,
            eval_period=self.configs.eval_period, 
            out_dir=str(self.model_dir), 
            resize=self.configs.resize,
            imgmode=self.configs.mode,
            num_bands=self.num_bands
        )
        trainer = MyTrainer(cfg, self.configs.patience)
        trainer.resume_or_load(resume=self.configs.resume)

        if self.configs.mode == "ms" and self.configs.pretrained_model:
            self.logger.info("Adjusting first conv layer weights for extra channels...")
            multiply_conv1_weights(trainer.model)

        if self.configs.freezing == True:
            self.logger.info("Applying custom layer freezing... ")

            # Freeze the initial convolutional stem
            # trainer.model.backbone.bottom_up.stem.freeze()
            for name, param in trainer.model.backbone.named_parameters():
                if "stem.conv1" not in name:
                    param.requires_grad = False

            # # Freeze the blocks within the first residual stage (res2)
            # for block in trainer.model.backbone.bottom_up.stages[0].children():
            #     block.freeze()            
        
        # if self.configs.mode == "ms" and self.configs.pretrained_model:
        #     self.logger.info("Adjusting first conv layer weights for extra channels...")
        #     multiply_conv1_weights(trainer.model)

        # if self.configs.selective_band_usage:
        #     FlexibleDatasetMapper = CustomBandMapper

        trainer.train()

        self.logger.info(f"Fine-tuning completed. Results saved in {self.model_dir}")
        
        self._plot_metrics()
        
        return cfg
    
    def _plot_metrics(self):
        metrics = load_json_arr(self.model_dir / "metrics.json")
        train_metrics = [row for row in metrics if 'total_loss' in row]
        val_metrics = [row for row in metrics if 'validation_loss' in row]
        
        last_train_loss = "{:.2f}".format(train_metrics[-1]['total_loss'])
        last_val_loss = "{:.2f}".format(val_metrics[-1]['validation_loss'])
        self.logger.info(f"[{self.configs.model}] Last Train Loss: {last_train_loss}")
        self.logger.info(f"[{self.configs.model}] Last Val loss: {last_val_loss}")

        # Plot training and validation loss
        plt.figure()
        plt.plot(
            [x['iteration'] for x in val_metrics],
            [x['validation_loss'] for x in val_metrics], 
            label='Total Validation Loss', color='red'
            )
        plt.plot(
            [x['iteration'] for x in train_metrics],
            [x['total_loss'] for x in train_metrics], 
            label='Total Training Loss'
            )
        plt.legend(loc='upper right')
        plt.title(f'Training and validation loss of model {self.configs.model}')
        plt.ylabel('Total Loss')
        plt.xlabel('Number of Iterations')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_validation_loss.png', dpi=300)
        plt.close()

        # plot AP50 metrics
        colors = plt.cm.tab10.colors

        if len(self.sites) > 1:
            for i, site in enumerate(self.sites):
                ap50_metrics = [row for row in metrics if f'{site.name}_val/segm/AP50' in row]
                iterations = [x['iteration'] for x in ap50_metrics]
                ap50_values = [x[f"{site.name}_val/segm/AP50"] for x in ap50_metrics]
                self.logger.info(f"[{self.configs.model}] Final AP50 Value of site {site.name}: {'{:.2f}'.format(ap50_values[-1])}")

                plt.plot(
                    iterations, 
                    ap50_values,
                    label=f'Site {site.name}',
                    color = colors[i % len(colors)],
                    marker = 'o',
                    linewidth = 2
                    )
        else:
            site = self.sites[0]
            ap50_metrics = [row for row in metrics if 'segm/AP50' in row]
            iterations = [x['iteration'] for x in ap50_metrics]
            ap50_values = [x["segm/AP50"] for x in ap50_metrics]
            self.logger.info(f"[{self.configs.model}] Final AP50 Value: {'{:.2f}'.format(ap50_values[-1])}")
            plt.plot(
                iterations, 
                ap50_values,
                label=f'Site {site.name}',
                marker = 'o',
                linewidth = 2
                )
        
        plt.legend(loc="best")
        plt.title(f'Validation AP50 of model {self.configs.model}')
        plt.ylabel('AP50')
        plt.xlabel('Number of Iterations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.model_dir / 'val_ap50.png', dpi=300)
        plt.close()

        return

    def predict(self):
        model_path = get_latest_model_path(self.model_dir)
        cfg = setup_cfg(
            update_model=model_path,
            imgmode=self.configs.mode,
            num_bands=self.num_bands,
            )
                
        for site in self.sites:
            site.predict(cfg)
            # site.large_prediction()

    def evaluate(self, test_site: Path | str = None, resplit:bool=False):

        if resplit:
            for site in self.sites:
                to_traintest_folders(site.tiles_dir, site.tiles_root, test_frac=self.configs.test_frac,
                                    folds=self.configs.folds, strict=self.configs.strict)

        model_path = get_latest_model_path(self.model_dir)

        cfg = setup_cfg(
            update_model=model_path,
            imgmode=self.configs.mode,
            num_bands=self.num_bands,
            )

        eval_results = {}

        if not test_site:
            # right now it works for testing the "test" site of training
            for site in self.sites:
                self.logger.info(f'[{site}] Evaluating...')
                prec, recall, f1 = site.evaluate(cfg)
                eval_results[site.name] = {'precision': prec, 'recall': recall, 'f1': f1}
        else:
            # TODO: Need to check this part
            test_site = Site(Path(test_site), self.appends, self.configs)
            test_site.tile()
            to_traintest_folders(test_site.tiles_dir, test_site.tiles_root, test_frac=1, # everyything goes to the test_dir
                                 folds=self.configs.folds, strict=self.configs.strict)
            prec, recall, f1 = test_site.evaluate(cfg)
            eval_results[test_site.stem] = {'precision': prec, 'recall': recall, 'f1': f1}

        return eval_results

    def clean_models(self):
        final_model = self.model_dir / "model_final.pth"
        if final_model.exists():
            keep = final_model
        else:
            # Otherwise, find the latest indexed model
            keep = Path(get_latest_model_path(self.model_dir))

        # Delete all other model_*.pth files
        for f in self.model_dir.glob("model_*.pth"):
            if f != keep:
                f.unlink()


# class CustomBandMapper(FlexibleDatasetMapper):
#     def __call__(self, dataset_dict):
#         try:
#             with rasterio.open(dataset_dict["file_name"]) as src:
#                 # Read only the specified bands
#                 img = src.read(indexes=)

#             # Transpose to (H, W, C)
#             img = np.transpose(img, (1, 2, 0)).astype("float32")

#             aug_input = T.AugInput(img)
#             transforms = self.augmentations(aug_input)
#             img = aug_input.image
#             dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

#             if "annotations" in dataset_dict:
#                 self._transform_annotations(dataset_dict, transforms, img.shape[:2])

#             return dataset_dict
#         except Exception as e:
#             print(f"Error processing {dataset_dict.get('file_name', 'unknown')}: {e}")
#             return None
