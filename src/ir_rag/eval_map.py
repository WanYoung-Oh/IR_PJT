from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ir_rag.io_util import iter_jsonl
from ir_rag.query_rewrite import build_search_query


def build_relevance_bm25(
    eval_path: Path,
    es: Any,
    index: str,
    *,
    llm: Any | None = None,
    top_k: int = 3,
) -> dict[int, set[str]]:
    relevance: dict[int, set[str]] = {}
    for sample in iter_jsonl(eval_path):
        eval_id = int(sample["eval_id"])
        query = build_search_query(sample["msg"], llm=llm)
        resp = es.search(
            index=index,
            query={"match": {"content": query}},
            size=top_k,
        )
        relevance[eval_id] = {str(h["_id"]) for h in resp["hits"]["hits"]}
    return relevance


def evaluate_map(
    eval_path: Path,
    retriever: Callable[..., list[str]],
    relevance_map: dict[int, set[str]],
    *,
    top_k: int = 20,
    llm: Any | None = None,
) -> float:
    """competition 방식 MAP: AP = sum_precision / hit_count (hit이 없으면 0).

    빈 relevant 집합(검색 불필요 케이스)은 topk가 비어 있으면 1.0, 아니면 0.0.
    """
    ap_scores: list[float] = []
    for sample in iter_jsonl(eval_path):
        eval_id = int(sample["eval_id"])
        relevant = relevance_map.get(eval_id, set())
        query = build_search_query(sample["msg"], llm=llm)
        ranked_docs = retriever(query, top_k=top_k)

        if relevant:
            hits = 0
            precision_sum = 0.0
            for rank, doc_id in enumerate(ranked_docs, start=1):
                if str(doc_id) in relevant:
                    hits += 1
                    precision_sum += hits / rank
            ap = precision_sum / hits if hits > 0 else 0.0
        else:
            # 검색 불필요 케이스: topk 비어 있으면 정답
            ap = 0.0 if ranked_docs else 1.0

        ap_scores.append(ap)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
