#!/usr/bin/env python3
"""F-2b: doc_metadata.jsonl의 키워드를 바탕으로 과학 동의어 사전 생성.

처리 흐름:
  1. doc_metadata.jsonl에서 빈출 키워드 수집
  2. LLM API로 키워드 배치별 동의어·이표기 쌍 생성
  3. ES synonym filter 형식(term1, term2 / term1 => term2)으로 출력

사용 예시:
    python scripts/build_synonyms.py \\
      --metadata artifacts/doc_metadata.jsonl \\
      --api solar \\
      --output artifacts/science_synonyms.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.io_util import iter_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SYN_SYSTEM = (
    "당신은 한국어 과학 용어 전문가입니다. "
    "주어진 과학 키워드 목록에서 동의어·이표기·약어 관계가 있는 쌍을 찾아 JSON으로 출력하세요. "
    "순수 JSON 배열만 출력하고 다른 설명은 하지 마세요."
)

_SYN_USER_TEMPLATE = """다음 과학 키워드 목록에서 동의어, 이표기(한국어↔영어), 약어 관계가 있는 쌍을 찾으세요.

키워드 목록:
{keywords}

규칙:
- 의미가 동등한 용어: ["광합성", "photosynthesis"] 형식으로 나열
- 한자어↔순우리말, 학술어↔일반어 포함
- 완전히 다른 의미는 제외
- 관계 없으면 빈 배열 반환

다음 형식으로 출력하세요 (최대 20쌍):
[
  ["용어1", "동의어1"],
  ["용어2", "동의어2a", "동의어2b"],
  ...
]"""


def _build_api_client(api: str):
    from openai import OpenAI
    configs = {
        "solar":  ("SOLAR_API_KEY",  "https://api.upstage.ai/v1"),
        "openai": ("OPENAI_API_KEY", None),
        "google": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    }
    env_key, base_url = configs[api]
    key = os.environ.get(env_key, "")
    if not key:
        raise RuntimeError(f"{env_key}가 설정되지 않았습니다.")
    kwargs: dict = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _default_model(api: str) -> str:
    return {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}[api]


def _collect_keywords(metadata_path: Path, top_n: int) -> list[str]:
    """doc_metadata.jsonl에서 빈출 키워드 top_n개를 반환한다."""
    counter: Counter = Counter()
    for row in iter_jsonl(metadata_path):
        for kw in row.get("keywords", []):
            kw = kw.strip()
            if kw:
                counter[kw] += 1
    keywords = [kw for kw, _ in counter.most_common(top_n)]
    logger.info("키워드 수집: 전체 %d종 → top %d 사용", len(counter), len(keywords))
    return keywords


def _generate_synonyms_batch(
    client, model: str, keywords: list[str]
) -> list[list[str]]:
    """키워드 배치에 대한 동의어 그룹을 생성한다."""
    kw_str = "\n".join(f"- {kw}" for kw in keywords)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYN_SYSTEM},
                {"role": "user", "content": _SYN_USER_TEMPLATE.format(keywords=kw_str)},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        groups = json.loads(raw)
        # 검증: list of list
        if not isinstance(groups, list):
            return []
        valid = []
        for g in groups:
            if isinstance(g, list) and len(g) >= 2:
                terms = [str(t).strip() for t in g if str(t).strip()]
                if len(terms) >= 2:
                    valid.append(terms)
        return valid
    except Exception as e:
        logger.warning("동의어 생성 실패: %s", e)
        return []


def _to_es_synonym_line(group: list[str]) -> str:
    """ES synonym filter 형식으로 변환. 양방향 동등: 'a, b, c'"""
    return ", ".join(group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="artifacts/doc_metadata.jsonl")
    parser.add_argument("--api", default="solar", choices=["solar", "openai", "google"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="artifacts/science_synonyms.txt")
    parser.add_argument("--top-keywords", type=int, default=300, help="사용할 상위 키워드 수")
    parser.add_argument("--batch-size", type=int, default=30, help="LLM 호출당 키워드 수")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    metadata_path = root / args.metadata
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise SystemExit(f"메타 파일 없음: {metadata_path}\n먼저 build_doc_metadata.py를 실행하세요.")

    model = args.model or _default_model(args.api)
    client = _build_api_client(args.api)
    logger.info("API: %s | 모델: %s", args.api, model)

    keywords = _collect_keywords(metadata_path, args.top_keywords)

    all_groups: list[list[str]] = []
    seen: set[frozenset] = set()

    for i in range(0, len(keywords), args.batch_size):
        batch = keywords[i:i + args.batch_size]
        logger.info("배치 %d/%d 처리 중 (%d키워드)…",
                    i // args.batch_size + 1,
                    (len(keywords) + args.batch_size - 1) // args.batch_size,
                    len(batch))
        groups = _generate_synonyms_batch(client, model, batch)
        for g in groups:
            key = frozenset(g)
            if key not in seen:
                seen.add(key)
                all_groups.append(g)
        if args.delay > 0:
            time.sleep(args.delay)

    # 출력
    lines = [_to_es_synonym_line(g) for g in all_groups]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("동의어 %d쌍 생성 완료 → %s", len(lines), out_path)
    logger.info("샘플 (상위 5개):\n%s", "\n".join(lines[:5]))


if __name__ == "__main__":
    main()
