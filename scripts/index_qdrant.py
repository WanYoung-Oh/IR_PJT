#!/usr/bin/env python3
"""Index documents.jsonl into Qdrant with HuggingFace embedding (GPU 대용량)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.embeddings import build_huggingface_embedding
from ir_rag.io_util import iter_jsonl
from ir_rag.preprocess import preprocess_science_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 컬렉션을 삭제하고 재생성. 미지정 시 컬렉션이 이미 존재하면 중단.",
    )
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = Path(args.input or cfg["paths"]["documents"])
    if not doc_path.is_absolute():
        doc_path = root / doc_path

    embed = build_huggingface_embedding(cfg["embedding"])
    client = QdrantClient(url=cfg["qdrant"]["url"])
    coll = cfg["qdrant"]["collection"]
    vsize = int(cfg["qdrant"]["vector_size"])

    existing = [c.name for c in client.get_collections().collections]
    if coll in existing:
        if not args.force:
            raise SystemExit(
                f"컬렉션 '{coll}'이 이미 존재합니다. "
                "덮어쓰려면 --force 플래그를 사용하세요."
            )
        client.delete_collection(coll)

    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=vsize, distance=Distance.COSINE),
    )

    documents = [
        Document(
            text=preprocess_science_doc(d["content"]),
            doc_id=d["docid"],
            metadata={"src": d["src"], "docid": d["docid"]},
        )
        for d in iter_jsonl(doc_path)
    ]

    vs = QdrantVectorStore(client=client, collection_name=coll)
    VectorStoreIndex.from_documents(
        documents,
        vector_store=vs,
        embed_model=embed,
        show_progress=True,
    )
    print(f"Qdrant indexed {len(documents)} docs → {coll}")


if __name__ == "__main__":
    main()
