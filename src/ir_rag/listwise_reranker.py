"""Phase 2.5 Listwise Reranker — Solar API 기반.

Reranker 출력 순서를 LLM이 listwise로 재정렬한다.
llm.complete(prompt) 인터페이스를 사용하며, 실패 시 원본 순서를 유지한다.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Few-shot examples: D10-S1 0.9371 발견 당시 실제 Up 패턴 기반
_FEWSHOT_BLOCK = """예시 1 (명사 혼동 구분):
질문: 아세틸 콜린의 역할이 뭐야?
후보:
[1] 신경근 접합부에서 아세틸콜린이 방출되어 근육 수축 유발
[2] 피루브산이 아세틸 CoA로 변환되는 탄수화물 대사 (주의: "아세틸 CoA"는 "아세틸콜린"과 다른 물질)
[3] DFP는 아세틸콜린에스테라아제의 활성 부위에 결합하여 분해를 억제
답변: 1, 3, 2

예시 2 (직접 답 > 간접 답):
질문: 나무가 생태계에서 하는 역할에 대해 설명해줘
후보:
[1] 나무는 광합성으로 CO2를 흡수하고 O2를 방출 (나무의 직접 역할)
[2] 이차 천이: 손상된 서식지가 복원되는 과정 (간접적 개념)
[3] 나무는 토양의 물을 뿌리로 흡수해 잎으로 대기에 방출 (나무의 직접 역할)
답변: 1, 3, 2

"""


def _make_prompt(query: str, docs: list[dict], n: int, use_fewshot: bool) -> str:
    doc_lines = []
    for i, d in enumerate(docs, 1):
        content = (d.get("content") or "")[:300]
        doc_lines.append(f"[{i}] {content}")
    doc_block = "\n\n".join(doc_lines)

    if use_fewshot:
        intro = (
            f'다음 질문에 "가장 직접적으로 답하는 문서"부터 순서대로 1~{n}번을 재배열하세요.\n'
            "동일/유사 키워드만으로 판단하지 말고 질문의 진짜 의도에 직접 답하는지 보세요.\n\n"
            f"{_FEWSHOT_BLOCK}위 예시처럼:\n\n"
        )
    else:
        intro = f'다음 질문에 "가장 직접적으로 답하는 문서"부터 순서대로 1~{n}번을 재배열하세요.\n\n'

    return (
        f"{intro}"
        f"질문: {query}\n\n"
        f"후보 문서 {n}개:\n{doc_block}\n\n"
        f"규칙:\n"
        f"- 가장 직접적인 답을 주는 문서일수록 앞에 배치\n"
        f"- {n}개 모두 포함하고 중복 없이\n"
        f"- 답변은 숫자 {n}개만, 쉼표로 구분, 설명 금지\n\n"
        f"답변:"
    )


def _parse_order(text: str, n: int) -> list[int]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    nums = re.findall(r"\b(\d+)\b", text)
    seen: set[int] = set()
    order: list[int] = []
    for s in nums:
        v = int(s)
        if v not in seen and 1 <= v <= n:
            seen.add(v)
            order.append(v)
    return order


def listwise_rerank(
    query: str,
    docs: list[dict],
    llm: Any,
    preserve_top1: bool = True,
    use_fewshot: bool = False,
) -> list[dict]:
    """Reranker 순서 docs를 LLM listwise로 재정렬한다.

    Parameters
    ----------
    query:
        standalone query text.
    docs:
        Reranker 정렬 순서의 dict 목록. 각 dict에 'content' 키 필요.
    llm:
        llm.complete(prompt) 인터페이스 객체 (OpenAIChatCompletionLLM).
    preserve_top1:
        True이면 Reranker top-1 문서를 항상 1위로 고정 (안전 모드).
    use_fewshot:
        True이면 few-shot 예시 2개를 프롬프트에 포함.

    Returns
    -------
    재정렬된 docs 리스트 (동일 아이템, 다른 순서).
    파싱 실패 / LLM 오류 시 원본 순서 그대로 반환.
    """
    n = len(docs)
    if n < 2:
        return list(docs)

    prompt = _make_prompt(query, docs, n, use_fewshot)
    try:
        raw = llm.complete(prompt).text or ""
    except Exception as e:
        logger.warning("Listwise LLM 호출 실패, 원본 순서 유지: %s", e)
        return list(docs)

    order = _parse_order(raw, n)
    if len(order) < n:
        logger.warning(
            "Listwise 파싱 불완전 (%d/%d 파싱됨), 원본 순서 유지. raw=%r",
            len(order), n, raw[:120],
        )
        return list(docs)

    if preserve_top1:
        if 1 in order:
            order.remove(1)
        order.insert(0, 1)

    return [docs[i - 1] for i in order]
