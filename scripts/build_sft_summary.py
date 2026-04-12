#!/usr/bin/env python3
"""Phase A-2: 문서 요약 SFT 데이터 생성 (보조).

목적: 모델이 문서 핵심을 추출하는 능력을 키워 RAG 답변 품질을 간접 강화한다.
QA 데이터(A-1)와 권장 혼합 비율: QA 80% + 요약 20%.

처리 흐름:
  문서 d → Solar API로 3~5문장 요약 생성
  → (문서, 요약) 쌍을 SFT 데이터로 추가
  → 시스템 프롬프트: "제공된 문서를 핵심만 간결하게 요약하세요"

사용 예시:
    python scripts/build_sft_summary.py \\
      --config config/default.yaml \\
      --api solar \\
      --max-docs 500 \\
      --output artifacts/sft_summary_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
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

_SYSTEM_PROMPT = "제공된 문서를 핵심만 간결하게 요약하세요."


# ---------------------------------------------------------------------------
# API 클라이언트
# ---------------------------------------------------------------------------

def _build_api_client(api: str):
    from openai import OpenAI
    configs = {
        "solar": ("SOLAR_API_KEY", "https://api.upstage.ai/v1"),
        "openai": ("OPENAI_API_KEY", None),
        "google": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    }
    if api not in configs:
        raise ValueError(f"지원하지 않는 API: {api}")
    env_key, base_url = configs[api]
    key = os.environ.get(env_key, "")
    if not key:
        raise RuntimeError(f"{env_key}가 설정되지 않았습니다. .env에 추가하세요.")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _default_model(api: str) -> str:
    return {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}[api]


# ---------------------------------------------------------------------------
# 요약 생성
# ---------------------------------------------------------------------------

def _generate_summary(client, model: str, doc_content: str) -> str:
    """문서를 3~5문장으로 요약한다."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "다음 문서를 핵심 내용 중심으로 3~5문장으로 요약하세요. "
                    "마크다운이나 특수 기호 없이 자연스러운 한국어 문장으로만 작성하세요.\n\n"
                    f"문서:\n{doc_content}"
                ),
            },
        ],
        max_tokens=256,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        return {"processed": [], "stats": {"included": 0, "error": 0}}
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed": [], "stats": {"included": 0, "error": 0}}


def _save_checkpoint(ckpt_path: Path, ckpt: dict) -> None:
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def build_summary_sft(
    doc_path: Path,
    out_path: Path,
    client,
    model: str,
    max_docs: int,
    seed: int,
) -> None:
    # 체크포인트 로드
    ckpt_path = out_path.with_suffix(".ckpt.json")
    ckpt = _load_checkpoint(ckpt_path)
    done_docids: set[str] = set(ckpt["processed"])
    stats = ckpt["stats"]

    if done_docids:
        logger.info("이어받기: 이미 처리된 문서 %d건 건너뜀", len(done_docids))

    # 문서 목록 준비
    all_docs = list(iter_jsonl(doc_path))
    rng = random.Random(seed)
    rng.shuffle(all_docs)

    todo_docs = [d for d in all_docs if d["docid"] not in done_docids]
    if max_docs > 0:
        remaining_quota = max_docs - len(done_docids)
        todo_docs = todo_docs[:max(0, remaining_quota)]

    logger.info("처리 대상: %d건 (전체 %d건 중)", len(todo_docs), len(all_docs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("a", encoding="utf-8")

    try:
        for i, doc in enumerate(todo_docs):
            docid = doc["docid"]
            content = doc.get("content", "")

            if not content.strip():
                logger.warning("[%d/%d] docid=%s 빈 문서 — 건너뜀", i + 1, len(todo_docs), docid)
                done_docids.add(docid)
                ckpt["processed"].append(docid)
                _save_checkpoint(ckpt_path, ckpt)
                continue

            logger.info("[%d/%d] docid=%s 요약 생성 중...", i + 1, len(todo_docs), docid)

            try:
                summary = _generate_summary(client, model, content)
                if not summary or len(summary) < 10:
                    raise ValueError(f"생성된 요약이 너무 짧음: '{summary}'")
            except Exception as e:
                logger.warning("  요약 생성 실패 (%s) — 건너뜀", e)
                stats["error"] = stats.get("error", 0) + 1
                done_docids.add(docid)
                ckpt["processed"].append(docid)
                ckpt["stats"] = stats
                _save_checkpoint(ckpt_path, ckpt)
                continue

            record = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"문서:\n{content}"},
                    {"role": "assistant", "content": summary},
                ],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            stats["included"] = stats.get("included", 0) + 1
            done_docids.add(docid)
            ckpt["processed"].append(docid)
            ckpt["stats"] = stats
            _save_checkpoint(ckpt_path, ckpt)

            logger.info("  저장 완료 | 누적: %d건", stats["included"])

    finally:
        out_file.close()

    print(
        f"\n=== Phase A-2 완료 ===\n"
        f"포함: {stats.get('included', 0)}건 | 오류: {stats.get('error', 0)}건\n"
        f"출력: {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="artifacts/sft_summary_data.jsonl")
    parser.add_argument(
        "--api", choices=["solar", "openai", "google"], default="solar",
        help="요약 생성에 사용할 API (기본: solar)",
    )
    parser.add_argument("--model", default=None, help="모델명 (기본: API별 기본값)")
    parser.add_argument("--max-docs", type=int, default=500,
                        help="처리할 최대 문서 수 (기본 500, 0=전체)")
    parser.add_argument("--seed", type=int, default=42, help="문서 샘플링 랜덤 시드")
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = root / cfg["paths"]["documents"]
    out_path = root / args.output

    model = args.model or _default_model(args.api)
    client = _build_api_client(args.api)
    logger.info("API: %s / %s", args.api, model)

    build_summary_sft(
        doc_path=doc_path,
        out_path=out_path,
        client=client,
        model=model,
        max_docs=args.max_docs if args.max_docs > 0 else 0,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
