from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .llm_resolver import llm_resolver, call_llm, get_llm_status

app = FastAPI()

# Pydantic model for receiving requests from Chainlink or other clients
class LLMRequestData(BaseModel):
    query: str
    context: Optional[str] = None

# Endpoint to receive requests from Chainlink
@app.post("/mozart")
async def mozart_endpoint(data: LLMRequestData):
    # Resolve which LLM to use based on the query and context
    # Note: Since llm_resolver is synchronous, you might want to make it async if it performs I/O
    llm_id = llm_resolver(data.query, data.context)

    # Call the resolved LLM and get the response
    response_data = await call_llm(data.query, data.context, llm_id)

    # Return the response data back to the caller
    return response_data


