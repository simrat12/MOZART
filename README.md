# MOZART: Modularized Orchestrator for Various AI Language Models

MOZART is an innovative FastAPI application designed to seamlessly interface with various large language models (LLMs). It selects the most relevant LLM for a given query based on the context and the specialties of available LLMs. This selection is done using cosine similarity, a mathematical measure used in text analysis.

## Key Features

- **Dynamic LLM Endpoint Registration**: Allows users to register new LLM APIs with their specific contexts.
- **Cosine Similarity-Based Selection**: Chooses the most suitable LLM for a query by comparing the query's context with the contexts of registered LLMs.
- **Modular Architecture**: Easily extendable to incorporate additional LLMs and functionalities.

## Components

### LLM Resolver (`llm_resolver.py`)

- **Purpose**: To select the most appropriate LLM for a given query.
- **Method**: Utilizes cosine similarity to compare the context of a query against the contexts of available LLMs.
- **Cosine Similarity**: A measure of similarity between two non-zero vectors. In this case, it's used to find the similarity between the text of the query and the contexts of different LLMs. The LLM with the highest similarity score is chosen.

### Main Application (`main.py`)

- **API Endpoints**:
  - `/register_llm_api`: Registers a new LLM API with an identifier, endpoint URL, and context.
  - `/mozart`: Main endpoint that receives a query and returns a response from the selected LLM.

### Configuration File (`llm_config.py`)

- **Function**: Stores the configuration of available LLMs and their contexts.

## Installation and Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `uvicorn main:app --reload`

## How to Use

1. **Register an LLM API**:
   - Send a POST request to `/register_llm_api` with the LLM's ID, endpoint, and context.
2. **Query an LLM**:
   - Send a POST request to `/mozart` with your query. The system will select the most suitable LLM and return its response.

## Understanding Cosine Similarity

Cosine similarity is central to MOZART's LLM selection mechanism. It measures the cosine of the angle between two vectors projected in a multi-dimensional space. In the context of text analysis, these vectors are derived from the text (using TF-IDF vectorization) of the query and the contexts of registered LLMs. The closer the cosine value to 1, the higher the similarity.

## Future Enhancements

- Implement caching for faster response times.
- Add more robust error handling and logging.
- Include support for more complex queries and contexts.
