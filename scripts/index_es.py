#!/usr/bin/env python3
"""Index documents.jsonl into Elasticsearch (Nori analyzer)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.es_util import ensure_index
from ir_rag.io_util import iter_jsonl
from ir_rag.preprocess import preprocess_science_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None, help="Override documents jsonl path")
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    es_url = cfg["elasticsearch"]["url"]
    index = cfg["elasticsearch"]["index"]
    doc_path = Path(args.input or cfg["paths"]["documents"])
    if not doc_path.is_absolute():
        doc_path = root / doc_path

    es = Elasticsearch(es_url)
    ensure_index(es, index)

    def actions():
        for doc in iter_jsonl(doc_path):
            yield {
                "_index": index,
                "_id": doc["docid"],
                "_source": {
                    "docid": doc["docid"],
                    "src": doc["src"],
                    "content": preprocess_science_doc(doc["content"]),
                },
            }

    bulk(es, actions())
    es.indices.refresh(index=index)
    print(f"Indexed into {index} from {doc_path}")


if __name__ == "__main__":
    main()
