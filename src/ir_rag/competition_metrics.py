"""대회 공개 베이스라인과 동일한 변형 MAP (리더보드 채점 로직 정합).

GT(ground truth)는 공개 eval.jsonl에 포함되지 않음. 주최 측 제공 파일 또는
자체 pseudo 라벨(`eval_gt.jsonl` 등)과 제출본을 맞춰 로컬에서만 검증한다.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any

from ir_rag.io_util import iter_jsonl

# eval_id -> 관련 docid 집합. **빈 집합**이면 「검색 불필요」GT로 간주 (베이스라인 README else 분기).
GtType = dict[int, set[str]]


def load_gt_jsonl(path: str | Path) -> GtType:
    """JSONL: 각 줄 ``{"eval_id": int, "relevant_docids": ["uuid", ...]}``.

    ``relevant_docids``가 빈 리스트이면 해당 eval은 검색 불필요 케이스.
    """
    gt: GtType = {}
    for obj in iter_jsonl(Path(path)):
        eid = int(obj["eval_id"])
        raw = obj.get("relevant_docids") or []
        gt[eid] = {str(x) for x in raw}
    return gt


def load_submission_rows(path: str | Path) -> list[dict[str, Any]]:
    """`.csv` 확장자여도 내용은 JSON 한 줄당 한 오브젝트 (베이스라인 관례)."""
    return list(iter_jsonl(Path(path)))


def calc_map(
    gt: Mapping[int, Set[str] | frozenset[str]],
    pred: Sequence[Mapping[str, Any]],
) -> float:
    """베이스라인 README ``calc_map`` 와 동일한 스칼라 MAP.

    - ``gt[eid]``가 비어 있지 않으면: ``topk`` 상위 3개에 대해 AP.
      AP = sum_precision / hit_count (hit이 없으면 0)
    - ``gt[eid]``가 비어 있으면: ``topk``가 비어 있으면 1, 아니면 0.

    ``pred`` 항목은 ``eval_id``, ``topk`` (docid 리스트) 필요.
    """
    sum_average_precision = 0.0
    for j in pred:
        eid = int(j["eval_id"])
        relevant = frozenset(gt.get(eid, frozenset()))
        topk = list(j.get("topk") or [])

        if relevant:
            hit_count = 0
            sum_precision = 0.0
            for i, docid in enumerate(topk[:3]):
                if str(docid) in relevant:
                    hit_count += 1
                    sum_precision += hit_count / (i + 1)
            average_precision = (
                sum_precision / hit_count if hit_count > 0 else 0.0
            )
        else:
            average_precision = 0.0 if topk else 1.0

        sum_average_precision += average_precision

    return sum_average_precision / len(pred) if pred else 0.0
