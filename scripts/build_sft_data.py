#!/usr/bin/env python3
"""SFT 학습 데이터 생성.

eval.jsonl + documents.jsonl + retriever → Unsloth SFT 포맷(messages JSONL) 변환.
설계 문서 §⑤ LLM 응답 생성 「SFT 데이터 변환」 참조.

사용 예시:
    python scripts/build_sft_data.py --config config/default.yaml --output artifacts/sft_data.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl, write_jsonl
from ir_rag.query_rewrite import build_search_query


def build_sft_dataset(
    eval_path: Path,
    doc_path: Path,
    retriever,
    out_path: Path,
    llm=None,
    top_k: int = 5,
) -> None:
    """SFT 학습 데이터를 생성하여 JSONL 파일로 저장한다.

    각 레코드는 ``{"messages": [...]}`` 형태로,
    system / user(참고 문서 + 질문) 메시지를 포함한다.
    assistant 답변은 실제 추론 결과가 없으므로 placeholder로 채워진다.
    (실제 학습 시 고품질 답변으로 교체 필요)

    Parameters
    ----------
    eval_path:
        ``eval.jsonl`` 경로.
    doc_path:
        ``documents.jsonl`` 경로.
    retriever:
        ``retriever(query: str, top_k: int) -> list[str]`` 인터페이스 함수.
    out_path:
        출력 JSONL 경로.
    llm:
        쿼리 재작성용 LLM (없으면 원본 쿼리 사용).
    top_k:
        컨텍스트에 포함할 문서 수.
    """
    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}

    records = []
    for sample in iter_jsonl(eval_path):
        msg = sample["msg"]
        query = build_search_query(msg, llm=llm)
        top_docids = retriever(query, top_k=top_k)

        context = "\n\n".join(
            f"[문서 {i + 1}] {doc_map[did]}"
            for i, did in enumerate(top_docids)
            if did in doc_map
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 과학 전문가입니다. "
                    "제공된 참고 문서를 근거로 질문에 답하세요. "
                    "문서에 없는 내용은 절대 추측하지 마세요."
                ),
            }
        ]
        for m in msg[:-1]:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({
            "role": "user",
            "content": f"참고 문서:\n{context}\n\n질문: {msg[-1]['content']}",
        })

        # assistant 답변은 고품질 데이터로 교체 후 사용
        messages.append({
            "role": "assistant",
            "content": "[TODO: 고품질 답변으로 교체하세요]",
        })

        records.append({"messages": messages})

    write_jsonl(out_path, records)
    print(f"SFT 데이터 생성 완료: {len(records)}건 → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="artifacts/sft_data.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    eval_path = root / cfg["paths"]["eval"]
    doc_path = root / cfg["paths"]["documents"]
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # BM25 retriever (ES 연결 없이도 동작하는 더미 retriever 제공)
    try:
        from elasticsearch import Elasticsearch
        from ir_rag.retrieval import es_bm25_doc_ids

        es = Elasticsearch(cfg["elasticsearch"]["url"])
        index = cfg["elasticsearch"]["index"]

        def retriever(query: str, top_k: int) -> list[str]:
            return es_bm25_doc_ids(es, index, query, top_k)

    except Exception as e:
        print(f"[경고] Elasticsearch 연결 실패 ({e}) — 빈 컨텍스트로 진행")

        def retriever(query: str, top_k: int) -> list[str]:
            return []

    build_sft_dataset(
        eval_path=eval_path,
        doc_path=doc_path,
        retriever=retriever,
        out_path=out_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
