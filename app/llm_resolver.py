import httpx
from typing import Optional, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM APIs you have set up
llm_apis = {
    "llama2:7b": {
        "endpoint": "http://localhost:8000/receive_question/",
        "context": "Specialty description of this LLM"
    },
    # ... other LLMs
}

def calculate_cosine_similarity(contexts: List[str], query_context: str) -> np.ndarray:
    """
    Calculate cosine similarity between the query context and a list of document contexts.
    """
    vectorizer = TfidfVectorizer()
    context_vectors = vectorizer.fit_transform(contexts + [query_context])
    cosine_similarities = cosine_similarity(context_vectors[-1], context_vectors[:-1]).flatten()
    return cosine_similarities

async def get_llm_status(llm_id: str) -> bool:
    """
    Check the status of an LLM by sending a health check request.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(llm_apis[llm_id]["endpoint"] + "status")
            return response.status_code == httpx.codes.OK
    except httpx.RequestError as e:
        logger.error(f"Error checking LLM status for {llm_id}: {e}")
        return False

async def llm_resolver(query: str, query_context: str) -> str:
    """
    Select an LLM based on the cosine similarity between the context of the query and the LLM's specialty.
    """
    llm_contexts = [llm['context'] for llm in llm_apis.values()]

    loop = asyncio.get_event_loop()
    cosine_similarities = await loop.run_in_executor(
        None, calculate_cosine_similarity, llm_contexts, query_context
    )

    most_similar_llm_index = np.argmax(cosine_similarities)
    most_similar_llm_id = list(llm_apis.keys())[most_similar_llm_index]

    logger.info(f"Selected LLM {most_similar_llm_id} for query: {query}")
    return most_similar_llm_id

async def call_llm(query: str, context: Optional[str], llm_id: str) -> Dict:
    """
    Call the chosen LLM API and return the response.
    """
    logger.info(f"Attempting to call LLM {llm_id} with query: '{query}' and context: '{context}'")

    llm_endpoint = llm_apis[llm_id]['endpoint']
    logger.info(f"llm endpoint: {llm_endpoint}")

    try:
        async with httpx.AsyncClient() as client:
            post_data = {"question": query, "context": context}
            logger.info(f"Sending POST request to {llm_endpoint} with data: {post_data}")
            response = await client.post(llm_endpoint, json=post_data)
            response.raise_for_status()
            logger.info(f"Response received from LLM API {llm_id}: {response.json()}")
            return response.json()
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM API {llm_id}: {e}, type: {type(e).__name__}")
        raise

