from dataclasses import dataclass

@dataclass
class Configs:
    """
    if train, the data_dir is the data to be trained on, model_dir is the output model dir 
    if predict or eval, the data_dir is the data to be predicted, model_dir is the finetuned model dir
    """
    model: str = None
    data: str | list[str]  = None

    # mode
    mode: str = "rgb" # "rgb" or "ms"

    # tiling
    tile_size: int = 40
    buffer: int = 30
    threshold: float = 0 # Minimum proportion of the tile covered by crowns to be accepted [0,1]
    force_retile: bool = False
    tile_placement: str = "adaptive" # adaptive or grid

    # model
    base_model: str = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    pretrained_model: str = 'models/pretrained/250312_flexi.pth'

    # training
    train_dir: str = "./data/train"
    test_frac: float = 0.15
    folds: int = 5
    val_fold: int = 4
    max_iter: int = 3000
    eval_period: int = 100
    patience: int = 5
    workers: int = 8
    strict: bool = False
    seed: int = 0
    resize: str = "rand_fixed"
    resume: bool = False # Whether to resume training from the last checkpoint

    # predict
    confidence: float = 0.2 # Confidence threshold for filtering crowns
    min_area: float = 2.0 # Minimum area of crowns to be retained (m^2)
    simplify: float = 0.3 # Tolerance for simplifying crown geometries
    intersection: float = 0.5 # Threshold for crown intersection

