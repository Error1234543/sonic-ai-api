import os

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# =========================================================
# SONIC AI API
# =========================================================

app = FastAPI(title="Sonic AI API")


# =========================================================
# CORS
# Allows your browser/Netlify frontend to call this API
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ENVIRONMENT VARIABLES
# Set these in Render Environment Variables
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")
SONIC_API_KEY = os.getenv("SONIC_API_KEY")


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


# =========================================================
# MODEL
# =========================================================

MODEL = "Qwen/Qwen3-4B-Thinking-2507"


# =========================================================
# AUTHORIZATION HEADER
# =========================================================

api_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    temperature: float | None = 0.7
    max_tokens: int | None = 2048


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Sonic AI API is running"
    }


# =========================================================
# MODELS
# =========================================================

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


# =========================================================
# CHAT COMPLETIONS
# =========================================================

@app.post("/v1/chat/completions")
def chat(
    request: ChatRequest,
    authorization: str | None = Security(api_key_header)
):

    # -----------------------------------------------------
    # Check Sonic API key configuration
    # -----------------------------------------------------

    if not SONIC_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="SONIC_API_KEY is not configured"
        )


    # -----------------------------------------------------
    # Check Authorization header
    # -----------------------------------------------------

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )


    # -----------------------------------------------------
    # Validate API key
    # -----------------------------------------------------

    expected_key = f"Bearer {SONIC_API_KEY}"

    if authorization != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


    # -----------------------------------------------------
    # Check Hugging Face token
    # -----------------------------------------------------

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured"
        )


    # -----------------------------------------------------
    # Send request to Hugging Face
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return response.model_dump()


    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )