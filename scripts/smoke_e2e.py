#!/usr/bin/env python3
"""
Smoke: small ES subset + BM25 MAP. Requires Elasticsearch (직접 설치 후 systemctl start elasticsearch).
Does not load Qwen embedding (GPU). Exit 1 if ES up but MAP logic fails.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:
    print("SKIP smoke: pip install elasticsearch (requirements-core.txt)")
    sys.exit(0)

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.es_util import ensure_index
from ir_rag.eval_map import build_relevance_bm25, evaluate_map
from ir_rag.io_util import iter_jsonl, write_jsonl
from ir_rag.preprocess import preprocess_science_doc
from ir_rag.retrieval import es_bm25_doc_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--limit-docs", type=int, default=80)
    parser.add_argument("--limit-eval", type=int, default=15)
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    es_url = cfg["elasticsearch"]["url"]
    base_index = cfg["elasticsearch"]["index"]
    smoke_index = f"{base_index}_smoke"
    doc_src = root / cfg["paths"]["documents"]
    eval_src = root / cfg["paths"]["eval"]

    try:
        es = Elasticsearch(es_url)
        if not es.ping():
            raise ConnectionError("ping failed")
    except Exception as e:
        print(f"SKIP smoke: Elasticsearch unavailable ({es_url}): {e}")
        print("Start Elasticsearch: sudo systemctl start elasticsearch")
        sys.exit(0)

    tmp = Path(tempfile.mkdtemp(prefix="ir_rag_smoke_"))
    try:
        docs_subset = tmp / "docs.jsonl"
        eval_subset = tmp / "eval.jsonl"

        docs_rows = [doc for doc, _ in zip(iter_jsonl(doc_src), range(args.limit_docs))]
        eval_rows = [row for row, _ in zip(iter_jsonl(eval_src), range(args.limit_eval))]
        write_jsonl(docs_subset, docs_rows)
        write_jsonl(eval_subset, eval_rows)

        ensure_index(es, smoke_index, recreate=True)

        def actions():
            for doc in iter_jsonl(docs_subset):
                yield {
                    "_index": smoke_index,
                    "_id": doc["docid"],
                    "_source": {
                        "docid": doc["docid"],
                        "src": doc["src"],
                        "content": preprocess_science_doc(doc["content"]),
                    },
                }

        bulk(es, actions())
        es.indices.refresh(index=smoke_index)

        rel = build_relevance_bm25(eval_subset, es, smoke_index, llm=None, top_k=3)

        def retriever(q: str, top_k: int) -> list[str]:
            return es_bm25_doc_ids(es, smoke_index, q, top_k)

        score = evaluate_map(eval_subset, retriever, rel, top_k=10, llm=None)
        print(f"SMOKE OK MAP@10≈{score:.4f} (subset, pseudo-relevance)")
        assert rel, "empty relevance"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if es.indices.exists(index=smoke_index):
            es.indices.delete(index=smoke_index)


if __name__ == "__main__":
    main()
