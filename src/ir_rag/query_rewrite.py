from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _LLMComplete(Protocol):
    def complete(self, prompt: str) -> Any: ...


FEW_SHOT_EXAMPLES = """예시 1)
대화:
[user] 광합성은 어디서 일어나?
[assistant] 엽록체에서 일어납니다.
[user] 그 과정을 더 자세히 설명해줘.
검색 쿼리: 광합성 과정 엽록체 명반응 암반응

예시 2)
대화:
[user] 뉴턴의 법칙에는 뭐가 있어?
[assistant] 운동 제1, 2, 3 법칙이 있습니다.
[user] 제2 법칙이 뭔지 알려줘.
검색 쿼리: 뉴턴 운동 제2 법칙 F=ma 가속도"""


def _is_valid_query(query: str, original: str, min_len: int = 4) -> bool:
    if len(query) < min_len:
        return False
    if len(query) > len(original) * 3:
        return False
    stopwords = {"네", "아", "음", "그렇군요", "맞습니다", "안녕"}
    if query.strip() in stopwords:
        return False
    return True


def is_science_question(msg: list[dict], llm: _LLMComplete) -> bool:
    """LLM을 이용해 과학 관련 질문인지 판별한다.

    치챗·감정 표현·일상 대화는 False, 과학·기술·학문 질문은 True 반환.
    판별 실패 시 True(과학 질문)로 간주해 파이프라인을 정상 실행한다.
    """
    import re

    last = next((m["content"] for m in reversed(msg) if m["role"] == "user"), "")
    prompt = (
        "다음 메시지가 과학·기술·학문 관련 질문이면 'yes', "
        "일상 대화·감정 표현·치챗이면 'no'로만 답하세요. "
        "반드시 yes 또는 no 한 단어만 출력하세요.\n\n"
        f"메시지: {last}\n답:"
    )
    try:
        out = llm.complete(prompt)
        raw = getattr(out, "text", str(out))
        # <think>...</think> 블록 제거 후 마지막 비어있지 않은 줄만 확인
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        answer = clean.strip().splitlines()[-1].strip().lower() if clean.strip() else ""
        if "yes" in answer:
            return True
        if "no" in answer:
            return False
        # 판별 불가 → 과학 질문으로 간주
        logger.warning("과학 질문 분류 불명확, 과학 질문으로 간주: '%s' (응답: '%s')", last, answer)
        return True
    except Exception:
        logger.warning("과학 질문 분류 실패, 과학 질문으로 간주: %s", last)
        return True


def build_search_query(msg: list[dict], llm: _LLMComplete | None = None) -> str:
    user_msgs = [m for m in msg if m["role"] == "user"]
    if not user_msgs:
        return ""
    original = user_msgs[-1]["content"]

    if len(user_msgs) == 1:
        return original

    if llm is not None:
        dialogue = "\n".join(f'[{m["role"]}] {m["content"]}' for m in msg)
        prompt = f"""{FEW_SHOT_EXAMPLES}

다음 대화에서 마지막 질문의 의도를 파악해 독립적인 검색 쿼리를 한 줄로만 생성하세요.
쿼리는 핵심 과학 용어 위주로 간결하게 작성하고, 불필요한 감정 표현은 제거하세요.

대화:
{dialogue}
검색 쿼리:"""
        out = llm.complete(prompt)
        text = getattr(out, "text", str(out))
        rewritten = text.strip().split("\n")[0]

        if _is_valid_query(rewritten, original):
            return rewritten
        logger.warning(
            "쿼리 재작성 실패, 원본 사용: '%s' (재작성: '%s')", original, rewritten
        )

    context = " ".join(m["content"] for m in msg[:-1])
    return f"{context} {original}"
