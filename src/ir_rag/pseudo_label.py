"""Pseudo Labeling — 임베딩 + LLM ideal answer 블렌딩으로 relevance 구축.

설계 문서 §⑥ 평가 「방법 ② Pseudo Labeling (권장)」 참조.
BM25 heuristic(build_relevance_bm25)보다 높은 정확도로 로컬 MAP을 측정할 때 사용.

주의:
    로컬 MAP의 한계: 이 relevance 는 대회 주최 측 비공개 GT와 수치가 다를 수 있다.
    개발 중 상대 비교용으로만 사용하고 최종 순위는 제출 후 공식 결과를 기준으로 한다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ir_rag.io_util import iter_jsonl
from ir_rag.query_rewrite import build_search_query

logger = logging.getLogger(__name__)


def build_relevance_pseudo(
    eval_path: Path,
    doc_path: Path,
    llm: Any,
    embed_model: Any,
    top_k: int = 3,
) -> dict[int, set[str]]:
    """임베딩 유사도 + LLM ideal answer 블렌딩으로 pseudo relevance를 구축한다.

    알고리즘:
    1. 모든 문서 임베딩 행렬(doc_mat)을 배치로 계산.
    2. 각 eval 샘플에 대해:
       a. 쿼리 임베딩(qv) 계산.
       b. LLM으로 ideal answer 생성 후 임베딩(av) 계산.
       c. ``blend = normalize(qv + av)`` 로 두 벡터를 블렌딩.
       d. doc_mat @ blend 코사인 유사도에서 top_k 추출.

    Parameters
    ----------
    eval_path:
        ``eval.jsonl`` 경로.
    doc_path:
        ``documents.jsonl`` 경로.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체.
    embed_model:
        ``embed_model.get_text_embedding_batch(texts) -> list[list[float]]``
        또는 ``embed_model.get_text_embedding(text) -> list[float]``
        인터페이스를 가진 임베딩 모델 (LlamaIndex HuggingFaceEmbedding).
    top_k:
        각 쿼리에 대해 반환할 관련 문서 수.

    Returns
    -------
    dict[int, set[str]]
        eval_id → 관련 docid 집합.

    Notes
    -----
    문서 수(4,272건) × 임베딩 차원(4096)의 행렬을 메모리에 올리므로
    FP32 기준 약 67MB.
    """
    # 1. 문서 전체 로드
    logger.info("문서 임베딩 계산 중 (%s) …", doc_path)
    docs: dict[str, str] = {}
    for d in iter_jsonl(doc_path):
        docs[d["docid"]] = d["content"]

    doc_ids = list(docs.keys())
    texts = [docs[did] for did in doc_ids]

    # 배치 임베딩 (지원 여부에 따라 분기)
    if hasattr(embed_model, "get_text_embedding_batch"):
        embeddings = embed_model.get_text_embedding_batch(texts)
        doc_mat = np.stack([np.array(e) for e in embeddings])
    else:
        doc_mat = np.stack([np.array(embed_model.get_text_embedding(t)) for t in texts])

    norms = np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12
    doc_mat = doc_mat / norms
    logger.info("문서 임베딩 완료: %d건", len(doc_ids))

    # 2. eval 쿼리별 pseudo relevance 구축
    relevance: dict[int, set[str]] = {}
    for sample in iter_jsonl(eval_path):
        eval_id = int(sample["eval_id"])
        query = build_search_query(sample["msg"], llm=llm)

        ideal_answer = llm.complete(
            f"다음 과학 질문에 대한 정확한 답변을 작성하세요.\n질문: {query}\n답변:"
        ).text.strip()

        qv = np.array(embed_model.get_query_embedding(query))
        av = np.array(embed_model.get_query_embedding(ideal_answer))
        blend = qv + av
        norm = np.linalg.norm(blend) + 1e-12
        blend = blend / norm

        sims = doc_mat @ blend  # (N,)
        top_idxs = sims.argsort()[-top_k:][::-1]
        relevance[eval_id] = {doc_ids[i] for i in top_idxs}

    logger.info("Pseudo relevance 구축 완료: %d건", len(relevance))
    return relevance
