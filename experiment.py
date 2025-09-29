import random
import pandas as pd
import matplotlib.pyplot as plt
import logging

from configs import Configs
from pipeline import Pipeline
from pathlib import Path
import json

def setup_logging():
    logging.basicConfig(
        level=logging.WARNING, 
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Force all loggers except __main__ to WARNING
    for name in logging.root.manager.loggerDict:
        if name != "__main__":
            logging.getLogger(name).setLevel(logging.WARNING)

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


def experiment(tile_size, buffer, sample_tiles = [5, 10, 20, 40], reps=3, test_site=None):
    results = []
    experiment_count = 0

    for n in sample_tiles:
        for r in range(reps):
            experiment_count += 1
            logger.info(f"Experiment {experiment_count} out of {len(sample_tiles) * reps}: "
                        f"{n} sample tiles, rep {r}")
            
            models_dir = f"model_tiles_{n}_rep_{r}"
            
            configs = Configs(models_dir=models_dir,
                            train_dir="./data/train", tile_size=tile_size, buffer=buffer, seed=r)
            pipeline = Pipeline(configs)

            if not any(pipeline.models_dir.iterdir()):
                pipeline.train(sample_n_tiles=n)

            eval_results = pipeline.evaluate(test_site)
            # {"site_1": {"precision":..., "recall":..., "f1":...}, ...}

            for site_name, metrics in eval_results.items():
                results.append({
                    "sample_tiles": n,
                    "rep": r,
                    "site": site_name,
                    "f1": metrics["f1"]
                })

                logger.info(f'site {site_name}: F1 score: {metrics["f1"]}')
            
            result_path = pipeline.models_dir / "evaluation results.json"
            with result_path.open('a') as f:
                f.write(json.dumps(results) + "\n")

    return results

def visualize_learning_curve(results):
    experiment_path = Path("experiment")
    df = pd.DataFrame(results)
    df.to_csv(experiment_path / "learning_curve.csv", index=False)

    # summary stats per site
    summary = df.groupby(["site", "sample_tiles"])["f1"].agg(["mean", "std"]).reset_index()

    # Plot 
    plt.figure(figsize=(8, 5))
    for site, site_df in summary.groupby("site"):
        plt.errorbar(
            site_df["sample_tiles"],
            site_df["mean"],
            yerr=site_df["std"],
            fmt="o-",
            capsize=5,
            label=site
        )

    plt.xlabel("Number of Training Tiles")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve (per site)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(experiment_path / "learning_curve.png", dpi=300)
    plt.show()


def main():
    setup_logging()
    
    results = experiment(tile_size=90, buffer=30,)

    visualize_learning_curve(results)



if __name__=="__main__":
    main()