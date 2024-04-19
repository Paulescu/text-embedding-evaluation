<div align="center">
    <h1>Which embedding model should you use?</h1>    
</div>

<div align="center">
    <img src="assets/banner.gif" alt="Intro to the course" style="width:75%;">
  </a>
</div>


#### Table of contents
* [Problem](#the-problem)
* [Solution](#solution)
* [Video lecture](#video-lecture)
* [Wanna learn more real-time ML?](#wanna-learn-more-real-world-ml)


## The problem

Text embeddings are vector representations of raw text that you compute using an embedding model

<div align="center">
    <img src="./assets/embedding_model.gif" width='400' />
</div>

These vectors representations are then used for downstream tasks, like

* **Classification** → for example, to classify tweet sentiment as either positive or negative.

* **Clustering** → for example, to automatically group news into topics.

* **Retrieval** → for example, to find similar documents to a given query.

Retrieval (the “R” in RAG) is the task of finding the most relevant documents given an input query. This is one of the most popular usages of embeddings these days, and the one we focus on in this repository.

<div align="center">
    <img src="./assets/retrieval.gif" width='400' />
</div>


There are many embedding models, both open and proprietary, so the question is:

> What embedding model is best for you problem? 🤔


In this repository you can find an evaluation script that helps you find the right embedding model for your use case.


## Solution

To evaluate a model for retrieval using a particular dataset we will

1. Load the model and your dataset from HuggingFace, with
    - `questions`
    - `contexts`, and
    - `correct answers`
2. Embed the context into the Vector DB, in our case Qdrant.
3. For each question retrieve the top `K` relevant documents from the Vector DB, and
4. Compare the overlap in information between the retrieved documents and the `correct answers`. We will use ragas, an open-source framework for RAG evalution, to compute `context precision` and `context recall`

5. Finally log the results, so you know what worked best.

## Before running the code

You will need

- An [OpenAI](https://openai.com/blog/openai-api) API key, because ragas will be making calls to `GPT-3.5 Turbo` to evaluate the context precision and recall.

- A [Qdrant]((https://qdrant.to/cloud?utm_source=twitter&utm_medium=social&utm_campaign=pau-labarta-bajo-hybrid-cloud-launch)) Vector DB with its corresponding URL and API key, which you can get for FREE by [signing up here](https://qdrant.to/cloud?utm_source=twitter&utm_medium=social&utm_campaign=pau-labarta-bajo-hybrid-cloud-launch)

## Let's run the code

1. Create an `.env` file and paste your `OPENAI_API_KEY`, `QDRANT_URL` and `QDRANT_API_KEY`
    ```
    $ cp .env.example .env
    ```

2. Create the virtual environment with Python Poetry
    ```
    $ make install
    ```

3. Update the list of models and datasets you want to test in `config.yml`
    
4. Run the evaluations
    ```
    $ make run-evals
    ```

## Video lecture


## Wanna learn more Real World ML?
Join more than 15k subscribers to the Real-World ML Newsletter. Every Saturday morning.

→ [Subscribe for FREE 🤗](https://www.realworldml.net/subscribe)