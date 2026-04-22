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
from ir_rag.es_util import ensure_index, load_synonyms_file
from ir_rag.io_util import iter_jsonl
from ir_rag.preprocess import preprocess_science_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None, help="Override documents jsonl path")
    parser.add_argument("--recreate", action="store_true",
                        help="인덱스를 삭제 후 재생성 (설정 변경 시 사용)")
    # F-2 옵션
    parser.add_argument("--metadata", default=None, metavar="JSONL",
                        help="doc_metadata.jsonl 경로. 지정 시 title/keywords/summary/category 필드도 색인")
    parser.add_argument("--synonyms", default=None, metavar="TXT",
                        help="science_synonyms.txt 경로. 지정 시 ES 동의어 필터에 추가")
    parser.add_argument("--lm-jelinek-mercer", action="store_true",
                        help="LMJelinekMercer 유사도 적용 (기본: BM25)")
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
        rules = [
            line.strip().split("\t")[0]
            for line in user_dict_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        user_dict_rules = rules or None
        print(f"사용자 사전 로드: {len(rules)}개 → {user_dict_path}")
    else:
        print(f"사용자 사전 없음 — 기본 Nori 사용 ({user_dict_path})")

    # F-2b: 과학 동의어 사전 로드
    extra_synonyms: list[str] | None = None
    if args.synonyms:
        syn_path = Path(args.synonyms)
        if not syn_path.is_absolute():
            syn_path = root / syn_path
        extra_synonyms = load_synonyms_file(syn_path) or None
        print(f"동의어 사전 로드: {len(extra_synonyms or [])}개 규칙 → {syn_path}")

    # F-1: 메타 정보 로드
    meta_map: dict[str, dict] = {}
    use_meta = bool(args.metadata)
    if use_meta:
        meta_path = Path(args.metadata)
        if not meta_path.is_absolute():
            meta_path = root / meta_path
        for row in iter_jsonl(meta_path):
            meta_map[row["docid"]] = row
        print(f"메타 정보 로드: {len(meta_map)}건 → {meta_path}")

    if args.lm_jelinek_mercer:
        print("유사도 알고리즘: LMJelinekMercer (λ=0.7)")
    else:
        print("유사도 알고리즘: BM25 (기본)")

    ensure_index(
        es, index,
        recreate=args.recreate,
        user_dict_rules=user_dict_rules,
        extra_synonyms=extra_synonyms,
        use_lm_jelinek_mercer=args.lm_jelinek_mercer,
        use_meta_fields=use_meta,
    )

    def actions():
        for doc in iter_jsonl(doc_path):
            did = doc["docid"]
            source: dict = {
                "docid": did,
                "src": doc["src"],
                "content": preprocess_science_doc(doc["content"]),
            }
            if use_meta and did in meta_map:
                m = meta_map[did]
                source["title"] = m.get("title", "")
                # keywords가 중첩 리스트인 경우 flatten 후 공백 구분 저장
                kws = m.get("keywords", [])
                flat_kws = [str(x) for k in kws for x in (k if isinstance(k, list) else [k])]
                source["keywords"] = " ".join(flat_kws)
                source["summary"] = m.get("summary", "")
                source["category"] = m.get("category", "")
            yield {
                "_index": index,
                "_id": did,
                "_source": source,
            }

    bulk(es, actions())
    es.indices.refresh(index=index)
    mode = "메타+멀티필드" if use_meta else "기본"
    print(f"Indexed into {index} from {doc_path} [{mode}]")


if __name__ == "__main__":
    main()
