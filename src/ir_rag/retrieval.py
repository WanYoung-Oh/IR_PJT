from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import NotFoundError


def _bm25_query_body(query: str, use_multi_field: bool) -> dict[str, Any]:
    """BM25 검색용 ES `query` 절 (multi_field 시 필드 boost 일원화)."""
    if use_multi_field:
        return {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "keywords^1.5", "summary^1.2", "content^1.2"],
                "type": "best_fields",
            }
        }
    return {"match": {"content": query}}


def build_uuid_to_docid(doc_path: str) -> dict[str, str]:
    """documents.jsonl을 읽어 UUID5(docid) → 원본 docid 역매핑을 생성한다."""
    import json
    mapping: dict[str, str] = {}
    with open(doc_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            did = doc.get("docid", "")
            if did:
                uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, did))
                mapping[uid] = did
    return mapping


def es_bm25_top_score(
    es: Elasticsearch,
    index: str,
    query: str,
    use_multi_field: bool = False,
) -> float:
    es_query = _bm25_query_body(query, use_multi_field)
    try:
        resp = es.search(
            index=index,
            query=es_query,
            size=1,
            source=False,
        )
    except (ESConnectionError, NotFoundError) as e:
        raise RuntimeError(f"Elasticsearch search failed on index '{index}': {e}") from e
    hits = resp["hits"]["hits"]
    return float(hits[0]["_score"]) if hits else 0.0


def es_bm25_doc_ids(
    es: Elasticsearch,
    index: str,
    query: str,
    top_k: int,
    use_multi_field: bool = False,
) -> list[str]:
    """BM25 검색으로 docid 목록을 반환한다.

    Parameters
    ----------
    use_multi_field:
        ``True`` 이면 title(^2)·keywords(^1.5)·summary·content 멀티필드 검색.
        F-2 메타 색인 이후에 활성화한다.
    """
    es_query = _bm25_query_body(query, use_multi_field)

    try:
        resp = es.search(
            index=index,
            query=es_query,
            size=top_k,
            source=["docid"],
        )
    except (ESConnectionError, NotFoundError) as e:
        raise RuntimeError(f"Elasticsearch search failed on index '{index}': {e}") from e
    out: list[str] = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source") or {}
        did = src.get("docid") or h.get("_id")
        if did:
            out.append(str(did))
    return out


# ---------------------------------------------------------------------------
# Dense retrieval (Qdrant)
# ---------------------------------------------------------------------------

def qdrant_dense_doc_ids(
    client: Any,
    collection: str,
    embed_fn: Callable[[str], list[float]],
    query: str,
    top_k: int,
    uuid_to_docid: dict[str, str] | None = None,
) -> list[str]:
    """Qdrant 컬렉션에서 벡터 검색을 수행하고 docid 목록을 반환한다.

    Parameters
    ----------
    client:
        ``qdrant_client.QdrantClient`` 인스턴스.
    collection:
        검색 대상 컬렉션 이름.
    embed_fn:
        쿼리 문자열을 벡터(``list[float]``)로 변환하는 함수.
    query:
        검색 쿼리 문자열.
    top_k:
        반환할 최대 결과 수.
    uuid_to_docid:
        UUID5 → 원본 docid 역매핑. payload에 docid가 없을 때 폴백으로 사용.

    Returns
    -------
    list[str]
        점수 내림차순으로 정렬된 docid 목록.
    """
    vector = embed_fn(query)
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    doc_ids: list[str] = []
    for hit in response.points:
        payload = hit.payload or {}
        did = payload.get("docid")
        if not did:
            hit_id = str(hit.id)
            did = (uuid_to_docid or {}).get(hit_id, hit_id)
        doc_ids.append(str(did))
    return doc_ids


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf_score(
    rankings: list[list[str]],
    k: int = 30,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """여러 검색 결과 랭킹을 RRF 공식으로 통합하여 점수 딕셔너리를 반환한다.

    RRF 공식: score(d) = sum_r( w_r / (k + rank(d, r) + 1) )
    rank 는 0-indexed.

    Parameters
    ----------
    rankings:
        각 검색 시스템이 반환한 docid 리스트의 목록.
        순서가 앞일수록 순위가 높다.
    k:
        RRF 상수. 기본값 20.
    weights:
        각 랭킹의 가중치. None 이면 균등(1.0) 적용.
        예: BM25 0.7, Dense 0.3 → weights=[0.7, 0.3]

    Returns
    -------
    dict[str, float]
        docid → RRF 점수. 점수 내림차순으로 정렬됨.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights 길이({len(weights)})가 rankings 길이({len(rankings)})와 다릅니다.")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    # 동점 시 docid 로 안정 정렬 (실행 간 순서 흔들림 완화)
    return dict(sorted(scores.items(), key=lambda t: (-t[1], t[0])))


# ---------------------------------------------------------------------------
# HyDE (Hypothetical Document Embeddings) — 조건부 실행
# ---------------------------------------------------------------------------

def generate_hyde_doc(query: str, llm: Any) -> str:
    """LLM 을 사용해 쿼리에 대한 가상 문서(HyDE)를 생성한다.

    Parameters
    ----------
    query:
        원본 검색 쿼리.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체
        (예: LlamaIndex ``LLM`` 서브클래스).

    Returns
    -------
    str
        생성된 가상 문서 텍스트 (<think> 블록 제거 완료).
    """
    import re
    prompt = (
        "다음 과학 질문에 대한 정확한 설명 문단을 100자 내외로 작성하세요.\n"
        f"질문: {query}\n설명:"
    )
    raw = llm.complete(prompt).text
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def hybrid_search_with_hyde(
    query: str,
    llm: Any,
    retriever_fn: Callable[[str], list[str]],
    es: Elasticsearch,
    index: str,
    hyde_threshold: float = 5.0,
    axis_weights: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """BM25 top score 기반으로 HyDE를 조건부 실행하는 하이브리드 검색.

    BM25 최고 점수가 ``hyde_threshold`` 이상이면 원본 쿼리 결과만 RRF 로 반환한다.
    임계값 미만이면 HyDE 문서와 쿼리 재작성을 추가하여 3축 RRF 를 수행한다.

    ``axis_weights`` 는 ``export_submission.run_pipeline`` 의 다축 가중
    (standalone, HyDE, alt_query) 와 동일한 의미다. ``None`` 이면 균등(1,1,1).

    Parameters
    ----------
    query:
        원본 검색 쿼리.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체.
    retriever_fn:
        쿼리 문자열을 받아 docid ``list[str]`` 을 반환하는 검색 함수.
    es:
        Elasticsearch 클라이언트.
    index:
        BM25 점수를 조회할 ES 인덱스 이름.
    hyde_threshold:
        HyDE 활성화 임계값.
        높게(10+) → 거의 항상 HyDE 실행 /
        낮게(2~3) → 검색 미흡 시만 실행 /
        기본값 5.0 → 균형.
    axis_weights:
        3축 RRF 가중치 ``(standalone, HyDE, alt_query)``. ``None`` 이면 균등.

    Returns
    -------
    dict[str, float]
        docid → RRF 점수. 점수 내림차순으로 정렬됨.
    """
    results_original = retriever_fn(query)
    bm25_score = es_bm25_top_score(es, index, query)

    if bm25_score >= hyde_threshold:
        # BM25 검색이 이미 충분히 높은 점수 → HyDE 불필요
        return rrf_score([results_original])

    hyde_doc = generate_hyde_doc(query, llm)
    results_hyde = retriever_fn(hyde_doc)

    fusion_prompt = (
        "다음 과학 질문을 다른 표현으로 재작성하세요 (1개만).\n"
        f"질문: {query}\n재작성:"
    )
    alt_query = llm.complete(fusion_prompt).text.strip()
    results_alt = retriever_fn(alt_query)

    w0, w1, w2 = axis_weights if axis_weights is not None else (1.0, 1.0, 1.0)
    return rrf_score(
        [results_original, results_hyde, results_alt],
        weights=[w0, w1, w2],
    )
