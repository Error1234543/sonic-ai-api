from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Sonic AI API")


class ChatRequest(BaseModel):
    message: str


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
                "id": "sonic-model",
                "object": "model"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    return {
        "id": "sonic-response",
        "object": "chat.completion",
        "model": "sonic-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sonic AI API is working!"
                },
                "finish_reason": "stop"
            }
        ]
    }
