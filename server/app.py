import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from server.rag import load_index, search, format_context
from server.ollama_client import OllamaClient

# Load env
load_dotenv()
OWNER_NAME = os.getenv("BOT_OWNER_NAME", "Your Name")
BOT_TONE = os.getenv("BOT_TONE", "friendly")
GEN_MODEL = os.getenv("GEN_MODEL", "phi3:mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Init Ollama client
client = OllamaClient(base_url=OLLAMA_BASE_URL, gen_model=GEN_MODEL, embed_model=EMBED_MODEL)

# App
app = FastAPI(title="AI Resume Bot (Free Edition)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector index at startup
INDEX_CHUNKS = load_index()

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    reply: str
    sources: List[Dict[str, Any]] = []

def build_system_prompt(owner_name: str, tone: str = "friendly") -> str:
    tone = tone.lower().strip()
    return (
        f"You are ResumeGPT, a helpful, {tone} AI assistant for {owner_name}.\n"
        f"Answer questions about {owner_name}'s background, skills, and projects using provided context.\n\n"
        "Rules:\n"
        "- Be concise; use bullet points for lists.\n"
        "- If info not in context, avoid inventing details about the person.\n"
        "- Cite sources inline as [S#] when using context.\n"
        "- Ask for clarification if needed.\n"
        "- Professional, safe tone."
    )

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        retrieved = search(client, user_msg, INDEX_CHUNKS, top_k=5)
        context_str, _ = format_context(retrieved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    system_prompt = build_system_prompt(OWNER_NAME, BOT_TONE)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context for {OWNER_NAME}:\n{context_str}\n\nRemember to cite [S#]."},
    ]
    if req.history:
        for m in req.history[-8:]:
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_msg})

    try:
        reply = client.chat(messages, temperature=0.6)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    ui_sources = [{
        "tag": f"S{s['rank']}",
        "section_title": s["section_title"],
        "score": round(s["score"], 3),
        "source": s["source"],
    } for s in retrieved]

    return ChatResponse(reply=reply, sources=ui_sources)

# Serve the static frontend
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
WEB_DIR = os.path.join(ROOT_DIR, "web")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")