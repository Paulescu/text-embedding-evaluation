import logging
import os
from typing import List, Optional, Tuple
from time import time

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import find_dotenv, load_dotenv
from fire import Fire
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant, VectorStore
from pydantic import BaseModel
from ragas import RunConfig, evaluate
from ragas.metrics import context_precision, context_recall
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())

class EvaluationResult(BaseModel):
    model_name: str
    dataset_name: str
    top_k_to_retrieve: int
    context_precision: float
    context_recall: float
    seconds_taken_to_embed: float
    seconds_taken_to_retrieve: float


def split_texts(
    texts: List[str],
    chunk_size: Optional[int] = 200,
    chunk_overlap: Optional[int] = 30,
) -> List[str]:
    """
    Split each text into chunks of a fixed size and overlap

    Args:
        texts (List[str]): List of texts to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[str]: List of chunked texts
    """
    # Split text by tokens using the tiktoken tokenizer
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        keep_separator=False,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_texts = []
    for text in texts:
        chunks = text_splitter.create_documents([text])
        chunked_texts.extend([chunk.page_content for chunk in chunks])
    return chunked_texts


def load_dataset_from_hugging_face(
    hf_dataset_name: str,
    hf_dataset_split: Optional[str] = 'train',
    n_rows: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Download a dataset from Hugging Face and return it as a DataFrame

    Args:
        hf_dataset_name (str): Hugging Face dataset name
        hf_dataset_split (str): Split to load from the dataset

    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    logger.info('Loading dataset')
    data = load_dataset(hf_dataset_name, split=hf_dataset_split)
    data = pd.DataFrame(data)
    if n_rows:
        data = data.head(n_rows)

    return data['question'], data['context'], data['correct_answer']


def load_model_from_hugging_face(model_name: str) -> HuggingFaceEmbeddings:
    logger.info('Downloading embeddings model from Hugging Face')
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def embed(
    texts: List[str],
    model: HuggingFaceEmbeddings,
) -> Tuple[VectorStore, float]:
    """
    Steps:
    1. Splits the texts into chunks,
    2. embeds them into a vector store, and
    3. returns the vector store and the time taken to embed the texts

    Args:
        texts (List[str]): List of texts to embed
        model (HuggingFaceEmbeddings): Hugging Face model

    Returns:
        VectorStore: Vector store containing the embedded texts
        float: Seconds taken to embed the texts
    """
    logger.info('Splitting texts into chunks')
    chunks = [split_texts(text) for text in texts]

    logger.info('Aggregating list of chunks into a flat list')
    docs = [item for chunk in chunks for item in chunk]

    logger.info(f'Number of documents: {len(docs)}')
    logger.info('Example document:')
    logger.info(docs[0])

    logger.info('Embedding documents into Qdrant')
    start_time = time()
    qdrant = Qdrant.from_texts(
        docs,
        model,
        url=os.environ['QDRANT_URL'],
        prefer_grpc=True,
        api_key=os.environ['QDRANT_API_KEY'],
        collection_name='text_embeddings_eval',
        force_recreate=True,
    )
    seconds_taken = time() - start_time

    return qdrant, seconds_taken


def retrieve(
    questions: List[str], vector_store: VectorStore, top_k_to_retrive: int
) -> Tuple[List[List[str]], float]:
    """
    Retrieve relevant documents for each question from the vector store

    Args:
        questions (List[str]): List of questions
        vector_store (VectorStore): Vector store
        top_k_to_retrive (int): Number of top documents to retrieve for each question

    Returns:
        List[List[str]]: List of relevant documents for each question
        float: Seconds taken to retrieve the documents
    """
    logger.info('Creating retriever')
    retriever = vector_store.as_retriever(search_kwargs={'k': top_k_to_retrive})

    start_time = time()

    contexts = []
    for question in tqdm(questions):
        contexts.append(
            [doc.page_content for doc in retriever.get_relevant_documents(question)]
        )

    seconds_taken = time() - start_time

    return contexts, seconds_taken


def evaluate_retriever(
    questions: List[str],
    retrieved_documents: List[List[str]],
    correct_answers: List[str],
):
    data = {
        'question': questions,
        'ground_truth': correct_answers,
        'contexts': retrieved_documents,
    }

    # RAGAS expects a Dataset object with the following format
    dataset = Dataset.from_dict(data)

    # RAGAS runtime settings to avoid hitting OpenAI rate limits
    run_config = RunConfig(max_workers=4, max_wait=180)

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=run_config,
        raise_exceptions=False,
    )

    return result


def run(
    model_name: Optional[str] = 'sentence-transformers/all-mpnet-base-v2',
    dataset_name: Optional[str] = 'explodinggradients/ragas-wikiqa',
    top_k_to_retrive: Optional[int] = 2,
    n_rows: Optional[int] = None,
):
    """
    Steps:
    1. Load dataset and model from Hugging Face,
    2. embed the documents into a vector store,
    3. retrieve relevant documents for each question, and
    4. evaluate the performance of the retriever using RAGAS
    5. Log the evaluation results

    Args:
        model_name (str): Hugging Face model name
        dataset_name (str): Hugging Face dataset name
        top_k_to_retrive (int): Number of top documents to retrieve for each question
        n_rows (int): Number of rows to load from the dataset
    """
    # 1. Load a dataset and model from Hugging Face
    questions, contexts, correct_answers = load_dataset_from_hugging_face(dataset_name, n_rows=n_rows)
    model = load_model_from_hugging_face(model_name)

    # 2. Embed the documents into a vector store    
    vector_store, seconds_taken = embed(contexts, model)
    seconds_taken_to_embed = int(seconds_taken)

    # 3. Retrieve relevant documents for each question
    retrieved_documents, seconds_taken = retrieve(questions, vector_store, top_k_to_retrive)
    seconds_taken_to_retrieve = int(seconds_taken)

    # 4. Evaluate the performance of the retriever using RAGAS
    result = evaluate_retriever(questions, retrieved_documents, correct_answers)

    # 5. Log the evaluation results
    logger.info('Evaluation results:')
    logger.info(f'Model: {model_name}')
    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'Number of rows: {n_rows}')
    logger.info(f'Top K to retrieve: {top_k_to_retrive}')
    logger.info(f'Context Precision: {result["context_precision"]}')
    logger.info(f'Context Recall: {result["context_recall"]}')
    logger.info(f'Seconds taken to embed: {seconds_taken_to_embed}')
    logger.info(f'Seconds taken to retrieve: {seconds_taken_to_retrieve}')

    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        top_k_to_retrieve=top_k_to_retrive,
        context_precision=result['context_precision'],
        context_recall=result['context_recall'],
        seconds_taken_to_embed=seconds_taken_to_embed,
        seconds_taken_to_retrieve=seconds_taken_to_retrieve,
    )

if __name__ == '__main__':
    Fire(run)
