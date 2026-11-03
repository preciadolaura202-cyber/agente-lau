import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

app = FastAPI(title="Agente IA Profesional", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_client = None


def get_client() -> Groq:
    global _client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar la variable de entorno GROQ_API_KEY.",
        )
    if _client is None:
        _client = Groq(api_key=api_key)
    return _client


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=FileResponse)
async def serve_home():
    """Servimos el HTML principal cuando el usuario entra a la URL base."""
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="El mensaje no puede estar vacío.")

    try:
        chat_completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente virtual amigable, preciso y profesional.",
                },
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error al consultar el modelo: {exc}",
        ) from exc

    content = chat_completion.choices[0].message.content
    if not content:
        raise HTTPException(
            status_code=502,
            detail="El modelo no devolvió una respuesta válida.",
        )

    return {"response": content}
