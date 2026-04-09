#!/usr/bin/env python3
"""Offline MAP@k with BM25 pseudo-relevance (Elasticsearch).

리더보드 변형 MAP은 GT가 필요하므로 scripts/run_competition_map.py 를 사용한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.eval_map import build_relevance_bm25, evaluate_map
from ir_rag.retrieval import es_bm25_doc_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rel-top", type=int, default=3)
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    es = Elasticsearch(cfg["elasticsearch"]["url"])
    index = cfg["elasticsearch"]["index"]
    eval_path = root / cfg["paths"]["eval"]
    if not eval_path.exists():
        raise SystemExit(f"Missing {eval_path}")

    rel = build_relevance_bm25(
        eval_path, es, index, llm=None, top_k=args.rel_top
    )

    def retriever(q: str, top_k: int) -> list[str]:
        return es_bm25_doc_ids(es, index, q, top_k)

    score = evaluate_map(
        eval_path, retriever, rel, top_k=args.top_k, llm=None
    )
    print(f"MAP@{args.top_k} (BM25 pseudo relevance, top {args.rel_top}): {score:.4f}")


if __name__ == "__main__":
    main()
