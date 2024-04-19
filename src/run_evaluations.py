import logging
import json
import yaml

from src.evaluate import run

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# load config.yml
with open('config.yml') as file:
    config = yaml.safe_load(file)

# Run evaluations
results = []
for model in config.get('models'):
    for dataset in config.get('datasets'):
        results.append(
            run(
                model_name=model,
                dataset_name=dataset,
                n_rows=config.get('n_samples_per_dataset'),
            )
        )

# Log evaluation results
logger.info('Evaluation results:')
for result in results:
    logger.info(json.dumps(result.dict(), indent=4))
