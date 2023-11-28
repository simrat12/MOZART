import httpx
from typing import Optional, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio

# LLM APIs you have set up, for example:
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
            # This assumes you have a status endpoint to verify the LLM is operational
            response = await client.get(llm_apis[llm_id] + "status")
            return response.status_code == httpx.codes.OK
    except httpx.RequestError:
        return False

async def llm_resolver(query: str, query_context: str) -> str:
    """
    Select an LLM based on the cosine similarity between the context of the query and the LLM's specialty.
    """
    llm_contexts = [llm['context'] for llm in llm_apis.values()]

    # Run the computationally intensive cosine similarity calculation in an executor
    loop = asyncio.get_event_loop()
    cosine_similarities = await loop.run_in_executor(
        None, calculate_cosine_similarity, llm_contexts, query_context
    )

    # Find the most suitable LLM
    most_similar_llm_index = np.argmax(cosine_similarities)
    most_similar_llm_id = list(llm_apis.keys())[most_similar_llm_index]

    # Return the endpoint of the most suitable LLM
    return llm_apis[most_similar_llm_id]['endpoint']


async def call_llm(query: str, context: Optional[str], llm_id: str) -> Dict:
    """
    Call the chosen LLM API and return the response.
    """
    # Make sure the LLM is available
    if not await get_llm_status(llm_id):
        raise ValueError(f"LLM API {llm_id} is not available.")

    # Call the LLM API
    async with httpx.AsyncClient() as client:
        response = await client.post(llm_apis[llm_id], json={"question": query, "context": context})
        response.raise_for_status()
        return response.json()
