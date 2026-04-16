"""재순위화(Reranking) 유틸: Qwen3-Reranker Cross-Encoder + Soft Voting.

설계 문서 §③ 재순위화 참조.
- 로컬: Qwen3-Reranker-4B (실험) / Qwen3-Reranker-8B (최종)  VRAM ~18GB
- API 대안: Cohere Rerank 3 (GPU 불필요)
- Soft Voting: Reranker 0.7 : RRF 0.3 MinMax 정규화 가중 결합
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------

def load_reranker(
    model_name: str = "Qwen/Qwen3-Reranker-8B",
    device: str = "cuda",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Qwen3-Reranker 모델과 토크나이저를 로드한다.

    Parameters
    ----------
    model_name:
        HuggingFace 모델 ID.
        ``"Qwen/Qwen3-Reranker-4B"`` (실험용, ~10GB) 또는
        ``"Qwen/Qwen3-Reranker-8B"`` (최종 제출, ~18GB).
    device:
        ``"cuda"`` 또는 ``"cpu"``.
    torch_dtype:
        ``"float16"`` 또는 ``"bfloat16"``.
    trust_remote_code:
        HuggingFace 모델의 커스텀 코드를 신뢰할지 여부.
        Qwen 계열은 ``True`` 필요. config YAML ``reranker.trust_remote_code`` 로 제어.

    Returns
    -------
    tuple[model, tokenizer]
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device).eval()
    logger.info("Reranker 로드 완료: %s (device=%s)", model_name, device)
    return model, tokenizer


def rerank_with_crossencoder(
    query: str,
    doc_ids: list[str],
    doc_texts: dict[str, str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> dict[str, float]:
    """Cross-Encoder로 문서를 재순위화하여 docid → 점수 딕셔너리를 반환한다.

    Parameters
    ----------
    query:
        검색 쿼리.
    doc_ids:
        재순위화할 docid 목록 (RRF 결과 등).
    doc_texts:
        docid → 문서 본문 매핑.
    model:
        ``load_reranker()`` 로 로드된 Cross-Encoder 모델.
    tokenizer:
        ``load_reranker()`` 로 로드된 토크나이저.
    batch_size:
        한 번에 처리할 문서 수.
    max_length:
        입력 최대 토큰 수.
    device:
        ``"cuda"`` 또는 ``"cpu"``.

    Returns
    -------
    dict[str, float]
        docid → relevance score, 점수 내림차순 정렬.
    """
    import torch

    scores: dict[str, float] = {}
    valid_ids = [d for d in doc_ids if d in doc_texts]

    for i in range(0, len(valid_ids), batch_size):
        batch_ids = valid_ids[i : i + batch_size]
        pairs = [[query, doc_texts[did]] for did in batch_ids]
        enc = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            # 이진 분류(관련/비관련) 모델: logit[:, 1] 또는 sigmoid(logit).squeeze()
            if logits.shape[-1] == 2:
                batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            else:
                batch_scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
        for did, score in zip(batch_ids, batch_scores):
            scores[did] = float(score)

    return dict(sorted(scores.items(), key=lambda t: (-t[1], t[0])))


# ---------------------------------------------------------------------------
# Soft Voting (Reranker + RRF 가중 결합)
# ---------------------------------------------------------------------------

def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """MinMax 정규화. min == max 인 경우(단일 값 또는 동점) NaN 없이 0을 반환한다."""
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def soft_voting_rerank(
    rrf_scores: dict[str, float],
    reranker_scores: dict[str, float],
    w_reranker: float = 0.65,
) -> dict[str, float]:
    """MinMax 정규화 후 Reranker/RRF 점수를 가중 결합한다.

    설계 문서 §③ Soft Voting 참조.
    ``w_reranker`` 튜닝 권장 범위: 0.6 ~ 0.8.

    Parameters
    ----------
    rrf_scores:
        ``retrieval.rrf_score()`` 결과 (docid → RRF 점수).
    reranker_scores:
        ``rerank_with_crossencoder()`` 결과 (docid → Reranker 점수).
    w_reranker:
        Reranker 가중치. RRF 가중치는 ``1 - w_reranker``.

    Returns
    -------
    dict[str, float]
        docid → 결합 점수, 내림차순 정렬.
    """
    doc_ids = sorted(set(rrf_scores) | set(reranker_scores))
    if not doc_ids:
        return {}

    rrf_vals = np.array([rrf_scores.get(d, 0.0) for d in doc_ids])
    rer_vals = np.array([reranker_scores.get(d, 0.0) for d in doc_ids])

    rrf_norm = _minmax_normalize(rrf_vals)
    rer_norm = _minmax_normalize(rer_vals)

    combined = w_reranker * rer_norm + (1.0 - w_reranker) * rrf_norm
    return dict(sorted(zip(doc_ids, combined), key=lambda x: (-x[1], x[0])))
