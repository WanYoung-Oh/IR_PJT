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
    parser.add_argument("--recreate", action="store_true",
                        help="인덱스를 삭제 후 재생성 (설정 변경 시 사용)")
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    es_url = cfg["elasticsearch"]["url"]
    index = cfg["elasticsearch"]["index"]
    doc_path = Path(args.input or cfg["paths"]["documents"])
    if not doc_path.is_absolute():
        doc_path = root / doc_path

    es = Elasticsearch(es_url)

    # 사용자 사전 로드 (artifacts/user_dict.txt)
    user_dict_rules: list[str] | None = None
    user_dict_path = root / cfg["paths"].get("user_dict_out", "artifacts/user_dict.txt")
    if user_dict_path.exists():
        # user_dictionary_rules 인라인 포맷: 단어만 (품사 태그 제외)
        # "단어\tNNG" → "단어"  (NNG 등 품사를 넣으면 분해로 오인해 BadRequestError 발생)
        rules = [
            line.strip().split("\t")[0]
            for line in user_dict_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        user_dict_rules = rules or None
        print(f"사용자 사전 로드: {len(rules)}개 → {user_dict_path}")
    else:
        print(f"사용자 사전 없음 — 기본 Nori 사용 ({user_dict_path})")

    ensure_index(es, index, recreate=args.recreate, user_dict_rules=user_dict_rules)

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
