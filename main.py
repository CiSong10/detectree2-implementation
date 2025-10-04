from pipeline import Pipeline
from configs import Configs
import json

configs = Configs(model="flexi929", data="akron")
pipeline = Pipeline(configs)
pipeline.predict()
