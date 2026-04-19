#!/usr/bin/env python3
"""B-3a: sft_doc_qa.jsonl에서 Reranker Fine-tuning용 트리플렛 생성.

A-1 데이터(sft_doc_qa.jsonl)를 파싱하여 (query, positive_docid, negative_docids)
트리플렛을 생성한다.

처리 흐름:
  sft_doc_qa.jsonl 각 행 →
    user 메시지에서 질문 파싱
    user 메시지 첫 번째 문서 내용 파싱 → documents.jsonl과 매칭 → positive docid
    BM25 검색 → negative docid 목록 (positive 제외)
    → artifacts/reranker_triplets.jsonl

사용 예시:
    python scripts/build_reranker_triplets.py \\
      --config config/default.yaml \\
      --sft-data artifacts/sft_doc_qa.jsonl \\
      --output artifacts/reranker_triplets.jsonl
VERSION: v1
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DOC_PATTERN = re.compile(r"\[문서 \d+\]\s*(.*?)(?=\n\n\[문서 |\n\n질문: |$)", re.DOTALL)


def _parse_user_message(user_content: str) -> tuple[str, str]:
    """user 메시지에서 (질문, 첫 번째 문서 본문)을 파싱한다."""
    question = ""
    if "\n\n질문: " in user_content:
        question = user_content.split("\n\n질문: ", 1)[-1].strip()

    first_doc = ""
    matches = _DOC_PATTERN.findall(user_content)
    if matches:
        first_doc = matches[0].strip()

    return question, first_doc


def _build_content_index(doc_path: Path, prefix_len: int = 50) -> dict[str, str]:
    """documents.jsonl 본문 앞 prefix_len자 → docid 역매핑을 반환한다."""
    index: dict[str, str] = {}
    for doc in iter_jsonl(doc_path):
        content = doc.get("content", "").strip()
        key = content[:prefix_len]
        if key:
            index[key] = doc["docid"]
    return index


def _match_docid(first_doc: str, content_index: dict[str, str], prefix_len: int = 50) -> str | None:
    """first_doc 앞 prefix_len자로 docid를 찾는다. 없으면 None."""
    key = first_doc[:prefix_len]
    return content_index.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sft-data", default="artifacts/sft_doc_qa.jsonl")
    parser.add_argument("--output", default="artifacts/reranker_triplets.jsonl")
    parser.add_argument("--top-k-neg", type=int, default=5, help="negative 후보 수")
    parser.add_argument("--use-multi-field", action="store_true",
                        help="BM25 검색 시 멀티필드 쿼리 사용 (F-2 색인 완료 후)")
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = root / cfg["paths"]["documents"]
    sft_path = root / args.sft_data
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from elasticsearch import Elasticsearch
    from ir_rag.retrieval import es_bm25_doc_ids

    es = Elasticsearch(cfg["elasticsearch"]["url"])
    es_index = cfg["elasticsearch"]["index"]

    logger.info("documents.jsonl 로드 및 content 역매핑 구축 중…")
    content_index = _build_content_index(doc_path)
    logger.info("역매핑 구축 완료: %d건", len(content_index))

    # 이어받기: 기존 출력 행 수 확인
    existing = 0
    if out_path.exists():
        existing = sum(1 for _ in iter_jsonl(out_path))
        logger.info("이어받기: 기존 %d건 건너뜀", existing)

    total = sum(1 for _ in iter_jsonl(sft_path))
    logger.info("SFT 데이터 총 %d건 → 트리플렛 변환 시작", total)

    ok = 0
    skip_no_q = 0
    skip_no_pos = 0
    skip_no_neg = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for idx, row in enumerate(iter_jsonl(sft_path)):
            if idx < existing:
                continue

            msgs = row.get("messages", [])
            user_content = next(
                (m["content"] for m in msgs if m["role"] == "user"), ""
            )
            question, first_doc = _parse_user_message(user_content)

            if not question:
                skip_no_q += 1
                continue

            positive_id = _match_docid(first_doc, content_index)
            if not positive_id:
                skip_no_pos += 1
                logger.debug("positive 매칭 실패: %s…", first_doc[:40])
                continue

            # BM25로 negative 후보 검색
            retrieved = es_bm25_doc_ids(
                es, es_index, question,
                top_k=args.top_k_neg + 5,
                use_multi_field=args.use_multi_field,
            )
            negatives = [d for d in retrieved if d != positive_id][: args.top_k_neg]
            if not negatives:
                skip_no_neg += 1
                continue

            triplet = {
                "query": question,
                "positive": positive_id,
                "negatives": negatives,
            }
            f_out.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            ok += 1

            if ok % 100 == 0:
                logger.info("진행: %d건 완료 / 실패(질문없음=%d, positive없음=%d, negative없음=%d)",
                            ok, skip_no_q, skip_no_pos, skip_no_neg)

    logger.info("완료: %d건 트리플렛 생성 → %s", ok, out_path)
    logger.info("실패 통계 — 질문없음: %d, positive없음: %d, negative없음: %d",
                skip_no_q, skip_no_pos, skip_no_neg)


if __name__ == "__main__":
    main()
