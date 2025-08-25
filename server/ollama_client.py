from typing import List, Dict, Any
import requests
import numpy as np

class OllamaClient:
    def __init__(self, base_url: str, gen_model: str, embed_model: str):
        self.base_url = base_url.rstrip("/")
        self.gen_model = gen_model
        self.embed_model = embed_model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, d) embeddings via Ollama's /api/embeddings."""
        vecs = []
        for t in texts:
            payload = {"model": self.embed_model, "input": t, "prompt": t}
            r = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=120)
            r.raise_for_status()
            data: Dict[str, Any] = r.json()
            if "embedding" in data:
                emb = data["embedding"]
            elif "data" in data and data["data"] and "embedding" in data["data"][0]:
                emb = data["data"][0]["embedding"]
            else:
                raise RuntimeError(f"Unexpected embeddings response: {data}")
            vecs.append(emb)
        return np.array(vecs, dtype=np.float32)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
        payload = {
            "model": self.gen_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "content" in data:
            return data["content"]
        raise RuntimeError(f"Unexpected chat response: {data}")