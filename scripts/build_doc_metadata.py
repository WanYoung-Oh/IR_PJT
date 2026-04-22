#!/usr/bin/env python3
"""F-1: documents.jsonl 기반 문서 메타 정보 생성.

각 문서에 대해 LLM API로 title, keywords, summary, category를 생성하여
artifacts/doc_metadata.jsonl에 저장한다. 중단 후 재실행 시 자동 이어받기.

사용 예시:
    python scripts/build_doc_metadata.py \\
      --config config/default.yaml \\
      --api solar \\
      --output artifacts/doc_metadata.jsonl

    # 이어받기 — 동일 명령어 재실행 (기존 출력 파일의 docid를 건너뜀)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CATEGORIES = [
    "물리", "화학", "생물", "지구과학",
    "의학", "영양학", "노화의학",
    "천문학", "컴퓨터", "전기공학",
    "사회과학", "과학일반", "기타",
]

# src 필드 도메인 → category 힌트 매핑
_SRC_DOMAIN_HINT: dict[str, str] = {
    "electrical_engineering":       "전기공학",
    "high_school_physics":          "물리",
    "college_physics":              "물리",
    "conceptual_physics":           "물리",
    "high_school_chemistry":        "화학",
    "college_chemistry":            "화학",
    "high_school_biology":          "생물",
    "college_biology":              "생물",
    "medical_genetics":             "생물",
    "virology":                     "생물 또는 의학",
    "anatomy":                      "의학",
    "college_medicine":             "의학",
    "human_sexuality":              "의학",
    "nutrition":                    "영양학",
    "human_aging":                  "노화의학",
    "astronomy":                    "천문학",
    "computer_security":            "컴퓨터",
    "college_computer_science":     "컴퓨터",
    "high_school_computer_science": "컴퓨터",
    "global_facts":                 "사회과학",
    "ARC_Challenge":                "과학일반(물리/화학/생물/지구과학 중 내용에 맞게 판단)",
}


def _extract_src_domain(src: str) -> str:
    """src 문자열에서 도메인 키를 추출한다.

    예: 'ko_mmlu__nutrition__test' → 'nutrition'
        'ko_ai2_arc__ARC_Challenge__test' → 'ARC_Challenge'
    """
    parts = src.split("__")
    return parts[1] if len(parts) >= 2 else src


_META_SYSTEM = (
    "당신은 과학 문서 분석 전문가입니다. "
    "주어진 문서를 분석하여 JSON 형식으로만 응답하세요. "
    "다른 설명이나 마크다운 코드 블록 없이 순수 JSON만 출력하세요."
)

_META_USER_TEMPLATE = """다음 과학 문서를 분석하여 JSON 형식으로 메타 정보를 생성하세요.

문서 출처(카테고리 분류 참고용): {domain_hint}
문서:
{content}

다음 형식으로 정확히 출력하세요:
{{
  "title": "30자 이내의 핵심 제목 (검색 쿼리처럼 간결하게)",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "summary": "문서에 명시된 내용만으로 2~3문장 요약 (추론·배경지식 추가 금지)",
  "category": "물리|화학|생물|지구과학|의학|영양학|노화의학|천문학|컴퓨터|전기공학|사회과학|과학일반|기타 중 하나"
}}

keywords 작성 기준:
- 이 문서를 검색할 때 사용할 핵심 용어 위주
- 문서에 직접 등장하거나 밀접한 전문 용어 우선
- 한글 표기와 영어 원어가 다른 경우 둘 다 포함 (예: "광합성 photosynthesis")"""


def _build_api_client(api: str):
    from openai import OpenAI
    configs = {
        "solar":  ("SOLAR_API_KEY",  "https://api.upstage.ai/v1"),
        "openai": ("OPENAI_API_KEY", None),
        "google": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    }
    if api not in configs:
        raise ValueError(f"지원하지 않는 API: {api}. 선택 가능: {list(configs)}")
    env_key, base_url = configs[api]
    key = os.environ.get(env_key, "")
    if not key:
        raise RuntimeError(f"{env_key}가 설정되지 않았습니다. .env에 추가하세요.")
    kwargs: dict = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _default_model(api: str) -> str:
    return {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}[api]


def _generate_metadata(client, model: str, content: str, src: str = "", retries: int = 2) -> dict | None:
    """문서 하나에 대한 메타 정보를 생성한다. 실패 시 None 반환."""
    domain = _extract_src_domain(src)
    domain_hint = _SRC_DOMAIN_HINT.get(domain, "알 수 없음 (내용으로 판단)")
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _META_SYSTEM},
                    {"role": "user", "content": _META_USER_TEMPLATE.format(
                        content=content[:800],
                        domain_hint=domain_hint,
                    )},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            # JSON 파싱
            meta = json.loads(raw)
            # 필수 필드 검증
            if not all(k in meta for k in ("title", "keywords", "summary", "category")):
                raise ValueError(f"필수 필드 누락: {list(meta.keys())}")
            # keywords 정규화: list가 아니면 변환, 중첩 리스트 flatten
            if not isinstance(meta["keywords"], list):
                meta["keywords"] = [str(meta["keywords"])]
            meta["keywords"] = [
                str(x)
                for k in meta["keywords"]
                for x in (k if isinstance(k, list) else [k])
            ][:8]  # 최대 8개
            # category 정규화
            if meta["category"] not in _CATEGORIES:
                meta["category"] = "기타"
            return meta
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패 (시도 %d/%d): %s", attempt + 1, retries + 1, e)
        except Exception as e:
            logger.warning("메타 생성 실패 (시도 %d/%d): %s", attempt + 1, retries + 1, e)
        if attempt < retries:
            time.sleep(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--api", default="solar", choices=["solar", "openai", "google"])
    parser.add_argument("--model", default=None, help="API 모델명 (미지정 시 api별 기본값)")
    parser.add_argument("--output", default="artifacts/doc_metadata.jsonl")
    parser.add_argument("--max-docs", type=int, default=None, help="처리할 최대 문서 수")
    parser.add_argument("--delay", type=float, default=0.1, help="요청 간 딜레이(초)")
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = root / cfg["paths"]["documents"]
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = args.model or _default_model(args.api)
    client = _build_api_client(args.api)
    logger.info("API: %s | 모델: %s", args.api, model)

    # 이어받기: 이미 처리된 docid 로드
    done: set[str] = set()
    if out_path.exists():
        for row in iter_jsonl(out_path):
            done.add(row["docid"])
        logger.info("이어받기: 기존 %d건 건너뜀", len(done))

    total = sum(1 for _ in iter_jsonl(doc_path))
    limit = min(args.max_docs, total) if args.max_docs else total
    logger.info("처리 대상: %d / %d건", limit - len(done), limit)

    processed = 0
    failed = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for doc in iter_jsonl(doc_path):
            if len(done) + processed >= limit:
                break
            docid = doc["docid"]
            if docid in done:
                continue

            content = doc.get("content", "")
            src = doc.get("src", "")
            meta = _generate_metadata(client, model, content, src=src)

            if meta is None:
                failed += 1
                logger.warning("[%s] 메타 생성 실패 — 건너뜀", docid)
            else:
                row = {"docid": docid, **meta}
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed += 1
                if processed % 50 == 0:
                    logger.info("진행: %d건 완료 / %d건 실패", processed, failed)

            if args.delay > 0:
                time.sleep(args.delay)

    logger.info("완료: %d건 생성 / %d건 실패 → %s", processed, failed, out_path)


if __name__ == "__main__":
    main()
