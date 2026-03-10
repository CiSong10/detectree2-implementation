from pipeline import Pipeline
from configs import Configs
from detectree2.models.train import get_latest_model_path

# pretrained_model = get_latest_model_path("models/finetuned/251111_akron_rgb")

configs = Configs(
    model=None,
    data=None,
    mode='rgb',
    threshold=0,
    # force_retile=True,
    workers=15,
    resize='rand_fixed',
    confidence=0.2,
    intersection=0.4,
    )
pipeline = Pipeline(configs)
pipeline.train()
# pipeline.predict()
