#!/usr/bin/env python3
"""Elasticsearch·Qdrant 색인이 documents.jsonl과 일치하는지 점검한다.

사용 예:
  python scripts/verify_indices.py --config config/default.yaml
  python scripts/verify_indices.py --config config/default.yaml --probe-dense  # GPU, 임베딩 1회
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl
from ir_rag.retrieval import es_bm25_doc_ids, qdrant_dense_doc_ids


def _count_jsonl_docs(path: Path) -> int:
    n = 0
    for _ in iter_jsonl(path):
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-es", action="store_true", help="Elasticsearch 점검 생략")
    parser.add_argument("--skip-qdrant", action="store_true", help="Qdrant 점검 생략")
    parser.add_argument(
        "--probe-dense",
        action="store_true",
        help="Qwen 임베딩을 로드해 벡터 검색 1회 (GPU·시간 소요)",
    )
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = root / cfg["paths"]["documents"]
    expected = _count_jsonl_docs(doc_path)

    print(f"기준 문서 수 (documents.jsonl): {expected}  ← {doc_path}")
    ok = True

    if not args.skip_es:
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            print(
                "[Elasticsearch] elasticsearch 패키지 미설치 — pip install elasticsearch",
            )
            return 1

        es_url = cfg["elasticsearch"]["url"]
        es_index = cfg["elasticsearch"]["index"]
        print(f"\n[Elasticsearch] {es_url}  index={es_index}")
        try:
            es = Elasticsearch(es_url)
            if not es.ping():
                print("  실패: ping 불가")
                ok = False
            elif not es.indices.exists(index=es_index):
                print(f"  실패: 인덱스 '{es_index}' 없음")
                ok = False
            else:
                cnt = es.count(index=es_index)["count"]
                print(f"  문서 수: {cnt}")
                if cnt != expected:
                    print(f"  경고: 기준({expected})과 불일치")
                    ok = False
                # 간단 검색
                try:
                    hits = es_bm25_doc_ids(es, es_index, "과학", min(5, expected))
                    print(f"  샘플 BM25 검색('과학', top5): {len(hits)}건 반환")
                    if not hits:
                        print("  경고: 검색 결과 0건 — 색인 필드·분석기 확인")
                        ok = False
                except Exception as e:
                    print(f"  실패: 검색 오류 — {e}")
                    ok = False
        except Exception as e:
            print(f"  실패: 연결 오류 — {e}")
            ok = False

    if not args.skip_qdrant:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            print(
                "\n[Qdrant] qdrant_client 미설치 — 프로젝트 venv에서 실행하세요:\n"
                "  source py310/bin/activate   # 또는 pip install qdrant-client",
            )
            return 1

        q_url = cfg["qdrant"]["url"]
        coll = cfg["qdrant"]["collection"]
        vsize_cfg = int(cfg["qdrant"]["vector_size"])
        print(f"\n[Qdrant] {q_url}  collection={coll}  (설정 vector_size={vsize_cfg})")
        try:
            client = QdrantClient(url=q_url)
            names = {c.name for c in client.get_collections().collections}
            if coll not in names:
                print(f"  실패: 컬렉션 '{coll}' 없음 (존재: {sorted(names)[:5]}{'…' if len(names) > 5 else ''})")
                ok = False
            else:
                info = client.get_collection(coll)
                vecs = info.config.params.vectors
                if vecs is not None:
                    # 단일 벡터 이름 없을 수 있음
                    sz = getattr(vecs, "size", None)
                    if sz is None and hasattr(vecs, "values"):
                        first = next(iter(vecs.values()), None)
                        sz = getattr(first, "size", None) if first else None
                else:
                    sz = None
                if sz is not None and sz != vsize_cfg:
                    print(f"  경고: 컬렉션 벡터 차원={sz}, 설정={vsize_cfg}")
                    ok = False
                elif sz is not None:
                    print(f"  벡터 차원: {sz} (설정과 일치)")

                nq = client.count(collection_name=coll, exact=True)
                n = nq.count
                print(f"  포인트 수: {n}")
                if n != expected:
                    print(f"  경고: 기준({expected})과 불일치")
                    ok = False

                if args.probe_dense:
                    from ir_rag.embeddings import build_huggingface_embedding

                    print("  임베딩 모델 로드 중 (probe-dense)…")
                    embed = build_huggingface_embedding(cfg["embedding"])

                    def embed_fn(q: str) -> list[float]:
                        return embed.get_query_embedding(q)

                    doc_path_str = str(doc_path)
                    import uuid as uuid_mod

                    uuid_to_docid = {}
                    with open(doc_path_str, encoding="utf-8") as f:
                        import json as json_mod

                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            doc = json_mod.loads(line)
                            did = doc.get("docid", "")
                            if did:
                                uid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, did))
                                uuid_to_docid[uid] = did

                    ids = qdrant_dense_doc_ids(
                        client,
                        coll,
                        embed_fn,
                        "machine learning",
                        3,
                        uuid_to_docid=uuid_to_docid,
                    )
                    print(f"  샘플 dense 검색('machine learning', top3): {ids}")
                    if len(ids) < 1:
                        print("  경고: 벡터 검색 결과 없음")
                        ok = False
        except Exception as e:
            print(f"  실패: {e}")
            ok = False

    if ok:
        print("\n결과: 색인 점검 통과 (문서 수·연결·샘플 검색)")
        return 0
    print("\n결과: 일부 실패 또는 경고 — 위 메시지 확인")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
