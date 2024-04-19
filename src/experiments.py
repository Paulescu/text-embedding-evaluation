import logging
import json

from src.evaluate import run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_TO_EVALUATE = [
    'sentence-transformers/all-mpnet-base-v2', # 109 params
    'Snowflake/snowflake-arctic-embed-l', # 334M params
    'Salesforce/SFR-Embedding-Mistral', # 7.11B params
]

DATASET_TO_EVALUATE = 'explodinggradients/ragas-wikiqa'
DATASET_ROWS = 10

results = []
for model_name in MODELS_TO_EVALUATE:
    results.append(run(model_name=model_name, dataset_name=DATASET_TO_EVALUATE, n_rows=DATASET_ROWS))

logger.info('Evaluation results:')
for result in results:
    logger.info(json.dumps(result.dict(), indent=4))