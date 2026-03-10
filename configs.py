from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

@dataclass
class Configs:
    """
    if train, the data_dir is the data to be trained on, model_dir is the output model dir name
    if predict or eval, the data_dir is the data to be predicted, model_dir is the finetuned model dir name
    """
    model: str = None
    data: str | list[str]  = None

    # mode
    mode: str = "rgb" # "rgb" or "ms"

    # tiling
    tile_size: int = 40
    buffer: int = 30
    threshold: float = 0 # Minimum proportion of the tile covered by crowns to be accepted [0,1]
    nan_threshold = 0.2
    force_retile: bool = False
    ignore_bands_indices: List[int] = field(default_factory=list)
    tile_placement: str = 'adaptive' # 'grid'

    # model
    base_model: str = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    pretrained_model: str = 'models/pretrained/250312_flexi.pth'

    # training
    test_frac: float = 0.15
    folds: int = 5
    max_iter: int = 3500
    eval_period: int = 100
    patience: int = 5
    workers: int = 8
    strict: bool = False
    seed: int = 0
    resize: str = "random" # 'fixed', 'random', or 'rand_fixed'
    resume: bool = False # Whether to resume training from the last checkpoint
    freezing: bool = True # Freeze backbone.stem and backbone.res2 -- TODO: enable more finegrain control later
    gamma: float = 0.1
    backbone_freeze: int = 3
    base_lr: float = 0.0003389
    # selective_band_usage = []

    # predict
    confidence: float = 0.15 # Confidence threshold for filtering crowns
    min_area: float = 1.5 # Minimum area of crowns to be retained (m^2)
    simplify: float = 0.3 # Tolerance for simplifying crown geometries
    intersection: float = 0.5 # Threshold for crown intersection


    def __post_init__(self):
        self.model_dir = Path("models/finetuned") / self.model
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.data, str):
            self.data = [self.data]

        # Write configs to configs.txt
        config_file = self.model_dir / "configs.txt"
        if not config_file.exists():
            with open(config_file, "w") as f:
                for key, value in asdict(self).items():
                    f.write(f"{key}: {value}\n")