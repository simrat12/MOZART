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
    try:
        # Ensure you're awaiting llm_resolver and getting the endpoint
        llm_endpoint = await llm_resolver(data.query, data.context)
        response_data = await call_llm(data.query, data.context, llm_endpoint)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


