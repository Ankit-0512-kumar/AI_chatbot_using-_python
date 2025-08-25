import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "index.json")

@dataclass
class Chunk:
    id: str
    text: str
    section_title: str
    source: str
    embedding: np.ndarray

def load_index(index_path: str = DEFAULT_INDEX_PATH) -> List[Chunk]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}. Run build_index.py first.")
    with open(index_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks: List[Chunk] = []
    for r in raw:
        chunks.append(Chunk(
            id=r["id"],
            text=r["text"],
            section_title=r.get("section_title", ""),
            source=r.get("source", "knowledge.md"),
            embedding=np.array(r["embedding"], dtype=np.float32),
        ))
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def search(client, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    q_vec = client.embed([query])  # (1, d)
    mat = np.vstack([c.embedding for c in chunks])  # (n, d)
    sims = cosine_sim(q_vec, mat).flatten()  # (n,)
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(idxs, start=1):
        c = chunks[i]
        results.append({
            "id": c.id,
            "score": float(sims[i]),
            "text": c.text,
            "section_title": c.section_title,
            "source": c.source,
            "rank": rank,
        })
    return results

def format_context(sources: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    lines = []
    source_map: Dict[str, Dict[str, Any]] = {}
    for s in sources:
        tag = f"S{s['rank']}"
        header = s["section_title"] or "Context"
        lines.append(f"[{tag}] {header} — {s['text']}".strip())
        source_map[tag] = {
            "id": s["id"],
            "section_title": s["section_title"],
            "score": s["score"],
            "source": s["source"],
        }
    return "\n".join(lines), source_map





