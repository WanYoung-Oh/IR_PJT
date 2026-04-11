"""LLM 응답 생성: CRAG/CoT 프롬프트 + Faithfulness 기반 Self-check.

설계 문서 §⑤ LLM 응답 생성 참조.
- CRAG_PROMPT: 문서 관련성 평가 후 답변
- SCIENCE_QA_PROMPT: CoT + 출처 명시
- generate_with_selfcheck: RAGAS Faithfulness 기반 재생성 (max 2회)
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Qwen3.5 <think>...</think> 블록을 제거한다."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ---------------------------------------------------------------------------
# 프롬프트 템플릿
# ---------------------------------------------------------------------------

CRAG_PROMPT = """아래 참고 문서를 검토하고, 질문에 답하기 전 각 문서의 관련성을 평가하세요.

관련성 없는 문서가 있다면 [무관련]으로 표시하고 해당 문서는 무시하세요.
모든 문서가 관련 없다면 "검색 결과 불충분"이라고 응답하세요.

참고 문서:
{context}

질문: {question}

답변 형식:
- 사용한 문서: [번호 목록]
- 답변: [답변 내용]
- 확신도: [높음/중간/낮음]"""

CHITCHAT_PROMPT = """당신은 친절한 AI 어시스턴트입니다. 사용자의 말에 공감하며 자연스럽게 대화하세요.
과학적 설명이나 문서 검색 없이 일상적인 대화로 응답하세요.

사용자: {question}
어시스턴트:"""

SCIENCE_QA_PROMPT = """당신은 과학 전문가입니다. 아래 문서를 근거로 질문에 답하세요.

규칙:
1. 반드시 제공된 문서 내용만을 근거로 답하세요
2. 추론 과정을 단계별로 서술하세요
3. 각 주장에 [문서 N] 형식으로 출처를 명시하세요
4. 문서에 없는 내용은 "문서에 근거 없음"으로 표기하세요

문서: {context}
질문: {question}"""


# ---------------------------------------------------------------------------
# 컨텍스트 포맷팅
# ---------------------------------------------------------------------------

def format_context(
    doc_ids: list[str],
    doc_map: dict[str, str],
    top_k: int = 5,
) -> str:
    """상위 docid 목록으로 프롬프트에 삽입할 컨텍스트 문자열을 만든다.

    Parameters
    ----------
    doc_ids:
        reranking 결과 등 순위순 docid 목록.
    doc_map:
        docid → 문서 본문 매핑.
    top_k:
        사용할 최대 문서 수.

    Returns
    -------
    str
        ``[문서 1] ...\n\n[문서 2] ...`` 형태 문자열.
    """
    parts: list[str] = []
    for i, did in enumerate(doc_ids[:top_k]):
        text = doc_map.get(str(did), "")
        if text:
            parts.append(f"[문서 {i + 1}] {text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 단순 생성
# ---------------------------------------------------------------------------

def generate_chitchat(question: str, llm: Any) -> str:
    """치챗/일상 대화에 대해 검색 없이 자연스러운 응답을 생성한다."""
    prompt = CHITCHAT_PROMPT.format(question=question)
    return _strip_think(llm.complete(prompt).text)


def generate_answer(
    question: str,
    context: str,
    llm: Any,
    use_crag: bool = True,
) -> str:
    """CRAG 또는 SCIENCE_QA 프롬프트로 LLM 응답을 생성한다.

    Parameters
    ----------
    question:
        사용자 질문 (standalone query).
    context:
        ``format_context()`` 로 만든 문서 컨텍스트 문자열.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체.
    use_crag:
        ``True`` 이면 CRAG_PROMPT 사용, ``False`` 이면 SCIENCE_QA_PROMPT 사용.

    Returns
    -------
    str
        생성된 답변 텍스트.
    """
    template = CRAG_PROMPT if use_crag else SCIENCE_QA_PROMPT
    prompt = template.format(context=context, question=question)
    return _strip_think(llm.complete(prompt).text)


# ---------------------------------------------------------------------------
# Self-check (RAGAS Faithfulness 기반 재생성)
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """환경변수에 따라 RAGAS용 LLM 래퍼를 반환한다.

    우선순위: SOLAR_API_KEY > GOOGLE_API_KEY > OPENAI_API_KEY (기본값).
    """
    import os
    if os.environ.get("SOLAR_API_KEY"):
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(ChatOpenAI(
            model="solar-pro",
            api_key=os.environ["SOLAR_API_KEY"],
            base_url="https://api.upstage.ai/v1",
        ))
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
    return None  # ragas 기본값(OpenAI) 사용


def _eval_faithfulness(question: str, answer: str, context: str) -> float:
    """RAGAS Faithfulness 점수를 계산한다.

    API 타임아웃·오류 발생 또는 결과가 nan이면 1.0을 반환해 self-check를 통과시킨다.
    (평가자 장애로 인한 불필요한 재생성 방지)
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [[context]],
        })

        from ragas import RunConfig

        eval_kwargs = {}
        llm = _build_ragas_llm()
        if llm is not None:
            eval_kwargs["llm"] = llm

        run_config = RunConfig(max_retries=3, max_wait=30, timeout=60)
        raw = evaluate(dataset=data, metrics=[faithfulness],
                       run_config=run_config, **eval_kwargs)

        if hasattr(raw, "to_pandas"):
            score = float(raw.to_pandas()["faithfulness"].iloc[0])
        elif isinstance(raw, dict):
            v = raw["faithfulness"]
            score = float(v[0] if isinstance(v, (list, tuple)) else v)
        else:
            score = float(getattr(raw, "faithfulness", [1.0])[0])

        if math.isnan(score):
            logger.warning("Faithfulness 평가 결과 nan (API 타임아웃 추정) — 통과 처리")
            return 1.0
        return score

    except Exception as e:
        logger.warning("Faithfulness 평가 실패 (%s) — 통과 처리", e)
        return 1.0


def generate_with_selfcheck(
    question: str,
    context: str,
    llm: Any,
    threshold: float = 0.7,
    max_retries: int = 2,
) -> str:
    """RAGAS Faithfulness 점수를 검사하며 threshold 미달 시 재생성한다.

    설계 문서 §⑤ Self-check 참조.

    Parameters
    ----------
    question:
        사용자 질문.
    context:
        문서 컨텍스트 문자열.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체.
    threshold:
        Faithfulness 최소 기준값 (0~1). 기본 0.7.
    max_retries:
        재생성 최대 횟수. 기본 2회.

    Returns
    -------
    str
        Faithfulness >= threshold 인 답변, 또는 max_retries 소진 후 마지막 답변.
    """
    import os
    has_api_key = (
        os.environ.get("SOLAR_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not has_api_key or os.environ.get("DISABLE_SELFCHECK"):
        logger.warning("Self-check 비활성화 — 단순 생성")
        return generate_answer(question, context, llm)

    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
    except ImportError as e:
        logger.warning("RAGAS 미설치 — Self-check 없이 단순 생성: %s", e)
        return generate_answer(question, context, llm)

    answer = generate_answer(question, context, llm)
    retries_left = max_retries

    while True:
        score = _eval_faithfulness(question, answer, context)
        if score >= threshold:
            logger.info("Self-check 통과: faithfulness=%.2f", score)
            return answer

        logger.warning(
            "Self-check 실패: faithfulness=%.2f < %.2f — 재생성 (%d회 남음)",
            score, threshold, retries_left,
        )
        if retries_left == 0:
            break

        retry_prompt = (
            "이전 답변이 문서 근거를 충분히 활용하지 못했습니다. "
            "반드시 아래 문서 내용만을 근거로 답하세요.\n\n"
            + SCIENCE_QA_PROMPT.format(context=context, question=question)
        )
        answer = _strip_think(llm.complete(retry_prompt).text)
        retries_left -= 1

    return answer
