import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Sonic AI API")

HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

MODEL = "Qwen/Qwen3-4B-Thinking-2507"
api_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False
)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    temperature: float | None = 0.7
    max_tokens: int | None = 2048

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Sonic AI API is running"
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat(
    request: ChatRequest,
    authorization: str | None = Security(api_key_header)
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured"
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return response.model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )