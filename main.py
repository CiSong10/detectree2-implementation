from pipeline import Pipeline
from configs import Configs
import json

configs = Configs(model_name="flexi929", force_retile=True)
pipeline = Pipeline(configs)
pipeline.train()
eval_results = pipeline.evaluate()

print(eval_results)

result_path = pipeline.model_dir / "evaluation results.json"
with result_path.open('w') as f:
    json.dumps(eval_results)

