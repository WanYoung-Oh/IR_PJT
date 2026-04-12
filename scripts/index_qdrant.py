#!/usr/bin/env python3
"""Index documents.jsonl into Qdrant with HuggingFace embedding (GPU 대용량)."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.embeddings import build_huggingface_embedding
from ir_rag.io_util import iter_jsonl
from ir_rag.preprocess import preprocess_science_doc


def _batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="임베딩 배치 크기 (기본 32)")
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

    embed_model = build_huggingface_embedding(cfg["embedding"])
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
        print(f"기존 컬렉션 '{coll}' 삭제 완료")

    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=vsize, distance=Distance.COSINE),
    )
    print(f"컬렉션 '{coll}' 생성 완료 (size={vsize})")

    documents = list(iter_jsonl(doc_path))
    print(f"문서 로드: {len(documents)}건")

    total = len(documents)
    indexed = 0

    for batch_docs in _batch(documents, args.batch_size):
        texts = [preprocess_science_doc(d["content"]) for d in batch_docs]

        # 텍스트 임베딩 생성 (text_instruction 적용)
        vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d["docid"])),
                vector=vec,
                payload={"docid": d["docid"], "src": d.get("src", "")},
            )
            for d, vec in zip(batch_docs, vectors)
        ]

        client.upsert(collection_name=coll, points=points)
        indexed += len(points)
        print(f"  [{indexed}/{total}] upserted", end="\r", flush=True)

    print(f"\nQdrant indexed {indexed} docs → {coll}")
    # 최종 확인
    count = client.count(coll)
    print(f"검증: Qdrant points_count = {count.count}")


if __name__ == "__main__":
    main()
