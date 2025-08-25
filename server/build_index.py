import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import re
from server.ollama_client import OllamaClient

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
KB_PATH = os.path.join(DATA_DIR, "knowledge.md")
INDEX_PATH = os.path.join(DATA_DIR, "index.json")

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_markdown(md: str, target_chars: int = 800, overlap: int = 100) -> List[Dict[str, str]]:
    """
    Robust: split by headings, then chunk by characters with overlap.
    """
    # Find all headings with their start positions
    pattern = re.compile(r"(?m)^#{1,6}\s+.*$")
    matches = list(pattern.finditer(md))
    sections = []
    if not matches:
        sections = [("Content", md)]
    else:
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
            heading_line = md[m.start():m.end()].strip()
            title = heading_line.lstrip("#").strip()
            content = md[m.end():end].strip()
            sections.append((title or "Section", content))

    chunks = []
    chunk_id = 1
    for title, content in sections:
        if not content:
            continue
        start = 0
        while start < len(content):
            end = min(len(content), start + target_chars)
            piece = content[start:end].strip()
            if piece:
                chunks.append({
                    "id": f"C{chunk_id}",
                    "section_title": title,
                    "text": piece,
                    "source": "knowledge.md",
                })
                chunk_id += 1
            # Overlap
            start = max(end - overlap, end)
    return chunks

def main():
    load_dotenv()
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    gen_model = os.getenv("GEN_MODEL", "phi3:mini")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"Knowledge file not found at {KB_PATH}")

    client = OllamaClient(base_url=base, gen_model=gen_model, embed_model=embed_model)

    md = read_file(KB_PATH)
    chunks = split_markdown(md)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks locally via Ollama ({embed_model})...")
    vecs = client.embed(texts)
    if vecs.shape[0] != len(texts):
        raise RuntimeError("Embedding count mismatch")

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        out.append({
            "id": c["id"],
            "text": c["text"],
            "section_title": c["section_title"],
            "source": c["source"],
            "embedding": [float(x) for x in vecs[i].tolist()],
        })
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} chunks to {INDEX_PATH}")

if __name__ == "__main__":
    main()