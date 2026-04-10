from __future__ import annotations

import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _LLMComplete(Protocol):
    def complete(self, prompt: str) -> Any: ...


# ---------------------------------------------------------------------------
# 규칙 기반 과학 질문 pre-filter
# 해당 패턴이 마지막 사용자 메시지에 있으면 LLM 호출 없이 is_science=True 반환.
# 4B 모델이 구어체·생활 접목 과학 질문을 치챗으로 오판하는 문제를 보완한다.
# ---------------------------------------------------------------------------
_SCIENCE_PREFILTER_RE = re.compile(
    r"(?:"
    r"뭐야|뭔지|뭔가|무엇|"
    r"어떻게|왜\b|이유|원인|"
    r"차이[는가]?|"
    r"알려줘|설명해|노하우|방법|과정|원리|특징|종류|역할|효과|장단점|"
    r"이로운|해로운|유익|유해|나열|"        # "이로운 점을 나열해줘" 등
    r"어떤\s*편|어느\s*정도|"              # "어떤 편이야", "어느 정도야"
    r"높[다은]|낮[다은]|많[다은]|적[다은]|"  # 비교·수량 질문
    r"어디야|어디에|어디서|어디[가이]|"
    r"누구야|누구인|"
    r"얼마나|몇\s"
    r")",
    re.IGNORECASE,
)

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

# is_science_question() 프롬프트용 few-shot 예시
_SCIENCE_FEW_SHOT = """판별 예시 (yes — 과학·지식 질문):
- "정상파가 뭐야?" → yes (물리학)
- "복숭아 키우는 노하우좀?" → yes (농업·원예)
- "잠을 잘 잤을 때 이로운 점을 나열해줘." → yes (의학·건강)
- "이태리에서 언론의 자유는 어떤 편이야?" → yes (사회과학)
- "건설 현장에서 망치로 벽을 치는 이유는?" → yes (건축·물리)
- "금성에서 달이 어떻게 보일까?" → yes (천문학)
- "담수가 가장 많이 있는 곳이 어디야?" → yes (지구과학)
- "폭풍을 쫓는 사람들이 누구야?" → yes (기상학)
- "식물과 동물 중 번식력이 높은 건 뭐야?" → yes (생물학)

판별 예시 (no — 순수 일상·감정·인사):
- "오늘 기분이 너무 좋아!" → no
- "안녕하세요 반갑습니다" → no
- "ㅋㅋㅋ 맞아 그렇지" → no
- "오늘 뭐 먹을까?" → no"""


def _is_valid_query(query: str, original: str, min_len: int = 4) -> bool:
    if len(query) < min_len:
        return False
    if len(query) > len(original) * 3:
        return False
    stopwords = {"네", "아", "음", "그렇군요", "맞습니다", "안녕"}
    if query.strip() in stopwords:
        return False
    return True


def _strip_think(text: str) -> str:
    """<think>...</think> 블록 제거. 닫힘 태그 없이 truncate된 경우도 처리."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)  # 미닫힘 블록
    return text.strip()


def is_science_question(msg: list[dict], llm: _LLMComplete) -> bool:
    """LLM을 이용해 과학/지식 관련 질문인지 판별한다.

    판별 순서:
      1. 규칙 기반 pre-filter — 지식 탐색 패턴이 명확하면 LLM 호출 없이 True 반환.
      2. LLM 판별 — 모호한 케이스만 LLM 호출. 실패 시 True(과학 질문)로 간주.

    순수 일상 대화·감정 표현·인사·치챗은 False,
    과학·기술·학문·사회·경제·역사·문화·교육·의학·농업·IT 관련 질문은 True 반환.
    """
    last = next((m["content"] for m in reversed(msg) if m["role"] == "user"), "")
    user_msgs = [m for m in msg if m["role"] == "user"]

    # ── Step 1: 규칙 기반 pre-filter ───────────────────────────────────────
    if _SCIENCE_PREFILTER_RE.search(last):
        logger.debug("과학 질문 pre-filter 통과 (LLM 호출 생략): '%s'", last)
        return True

    # ── Step 2: LLM 판별 ────────────────────────────────────────────────────
    if len(user_msgs) > 1:
        # 멀티턴: 전체 대화 맥락을 포함해 판별
        dialogue = "\n".join(f'[{m["role"]}] {m["content"]}' for m in msg)
        prompt = (
            "다음 대화의 마지막 질문이 지식·학문·과학·기술·사회·경제·역사·문화·"
            "교육·의학·농업·원예·정보통신(IT)·코딩 관련 질문이면 'yes', "
            "순수한 일상 대화·감정 표현·인사·치챗이면 'no'로만 답하세요. "
            "구체적인 사실·원리·원인을 묻는 질문은 yes로 판별하세요. "
            "반드시 yes 또는 no 한 단어만 출력하세요.\n\n"
            f"{_SCIENCE_FEW_SHOT}\n\n"
            f"대화:\n{dialogue}\n<think>\n</think>\n답:"
        )
    else:
        # 단일턴: 마지막 메시지만 판별
        prompt = (
            "다음 메시지가 지식·학문·과학·기술·사회·경제·역사·문화·"
            "교육·의학·농업·원예·정보통신(IT)·코딩 관련 질문이면 'yes', "
            "순수한 일상 대화·감정 표현·인사·치챗이면 'no'로만 답하세요. "
            "구체적인 사실·원리·원인을 묻는 질문은 yes로 판별하세요. "
            "반드시 yes 또는 no 한 단어만 출력하세요.\n\n"
            f"{_SCIENCE_FEW_SHOT}\n\n"
            f"메시지: {last}\n<think>\n</think>\n답:"
        )
    try:
        out = llm.complete(prompt)
        raw = getattr(out, "text", str(out))
        clean = _strip_think(raw)
        answer = clean.splitlines()[-1].strip().lower() if clean else ""
        if "yes" in answer:
            return True
        if "no" in answer:
            return False
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
<think>
</think>
검색 쿼리:"""
        out = llm.complete(prompt)
        text = getattr(out, "text", str(out))
        rewritten = _strip_think(text).split("\n")[0].strip()

        if _is_valid_query(rewritten, original):
            return rewritten
        logger.warning(
            "쿼리 재작성 실패, 원본 사용: '%s' (재작성: '%s')", original, rewritten
        )

    context = " ".join(m["content"] for m in msg[:-1])
    return f"{context} {original}"
