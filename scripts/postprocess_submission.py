#!/usr/bin/env python3
"""제출 파일 answer 필드 후처리 + Solar API 재생성.

현재 파이프라인 결과물에 남아있는 노이즈를 제거하고,
truncate된 <think> 케이스는 외부 API로 재생성한다.

패턴별 처리:
  [A] <think> 앞에 내용 있음 → 앞 내용만 사용 (18건)
  [B] <think>로 시작, 잘림   → Solar/OpenAI/Google API 재생성 (85건)
  [C] CRAG 대시 포맷         → "- 답변:" 본문만 추출 (111건)
  [D] 이미 깨끗함            → 그대로 유지 (6건)

사용 예시:
    # 패턴 미리보기 (저장 안 함)
    python scripts/postprocess_submission.py --dry-run

    # 후처리만 (Solar API 재생성 없음)
    python scripts/postprocess_submission.py \\
        --input  artifacts/sample_submission.csv \\
        --output artifacts/sample_submission_clean.csv

    # 후처리 + Solar API 재생성 (--regen-api)
    python scripts/postprocess_submission.py \\
        --input  artifacts/sample_submission.csv \\
        --output artifacts/sample_submission_clean.csv \\
        --regen-api solar \\
        --eval   data/eval.jsonl \\
        --docs   data/documents.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.io_util import iter_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# 패턴 분류
# ---------------------------------------------------------------------------

def _classify(answer: str) -> str:
    """answer 패턴을 분류한다."""
    if "<think>" not in answer:
        if answer.strip().startswith("-") and "사용한 문서" in answer:
            return "C"   # CRAG 대시 포맷
        return "D"       # 이미 깨끗함

    before = answer.split("<think>")[0].strip()
    if before:
        return "A"       # <think> 앞에 내용 있음
    return "B"           # <think>로 시작 (truncate)


# ---------------------------------------------------------------------------
# 각 패턴별 정제 함수
# ---------------------------------------------------------------------------

def _clean_A(answer: str) -> str:
    """<think> 앞 내용만 추출."""
    text = answer.split("<think>")[0].strip()
    # 뒤에 붙은 다이얼로그 형식("사용자: ... 어시스턴트: ...") 제거
    text = re.sub(r"\n+(사용자|유저|User)[:：].*", "", text, flags=re.DOTALL).strip()
    return text


def _clean_C(answer: str) -> str:
    """CRAG 대시 포맷에서 답변 본문만 추출."""
    text = answer.strip()

    # "- 답변: <본문>" 추출 (확신도 줄 직전까지)
    m = re.search(r"-\s*답변:\s*(.*?)(?=\n-\s*확신도:|\Z)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # 답변 라벨 없이 메타 줄만 제거
        lines = [
            l for l in text.split("\n")
            if not re.match(r"^-?\s*(사용한\s*문서|확신도|이유)[:：]", l)
        ]
        text = "\n".join(lines).strip()

    # 뒷부분 메타 제거 (확신도, **이유:** 등)
    text = re.sub(r"\n-\s*확신도:.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\n\*\*이유:\*\*.*", "", text, flags=re.DOTALL).strip()

    if not text or text in ("검색 결과 불충분", "검색 불충분"):
        text = "제공된 문서에서 관련 정보를 찾을 수 없습니다."

    return text


# ---------------------------------------------------------------------------
# API 재생성 (패턴 B용)
# ---------------------------------------------------------------------------

def _build_api_client(api: str):
    """OpenAI 호환 클라이언트 반환."""
    from openai import OpenAI
    if api == "solar":
        key = os.environ.get("SOLAR_API_KEY", "")
        if not key:
            raise RuntimeError("SOLAR_API_KEY가 .env에 설정되지 않았습니다.")
        return OpenAI(api_key=key, base_url="https://api.upstage.ai/v1")
    elif api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 .env에 설정되지 않았습니다.")
        return OpenAI(api_key=key)
    elif api == "google":
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY가 .env에 설정되지 않았습니다.")
        return OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    raise ValueError(f"지원하지 않는 API: {api}")


def _regen_answer(
    client,
    model: str,
    question: str,
    context: str,
    is_science: bool,
) -> str:
    """Solar/OpenAI/Google API로 답변 재생성."""
    if not is_science:
        messages = [
            {"role": "system", "content":
             "당신은 친절한 AI 어시스턴트입니다. 자연스러운 한국어로 대화하세요."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content":
             "당신은 과학 전문가입니다. 제공된 참고 문서를 근거로 질문에 답하세요. "
             "문서에 없는 내용은 절대 추측하지 마세요."},
            {"role": "user", "content":
             f"참고 문서:\n{context}\n\n질문: {question}"},
        ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  default="artifacts/sample_submission.csv")
    parser.add_argument("--output", default=None,
                        help="출력 경로 (기본: 원본 백업 후 덮어쓰기)")
    parser.add_argument("--dry-run", action="store_true",
                        help="변경 내용만 출력, 저장 안 함")
    parser.add_argument("--regen-api", choices=["solar", "openai", "google"],
                        default=None,
                        help="패턴 B (<think> truncate) 재생성에 사용할 API")
    parser.add_argument("--regen-model", default=None,
                        help="재생성 모델명 (기본: solar→solar-pro, openai→gpt-4o-mini, google→gemini-2.0-flash)")
    parser.add_argument("--eval",  default="data/eval.jsonl",
                        help="eval.jsonl 경로 (재생성 시 질문 조회)")
    parser.add_argument("--docs",  default="data/documents.jsonl",
                        help="documents.jsonl 경로 (재생성 시 컨텍스트 구성)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path

    # ── 보조 데이터 로드 (재생성 모드) ────────────────────────────────────
    api_client = None
    api_model  = "solar-pro"
    eval_map: dict[int, dict] = {}
    doc_map:  dict[str, str]  = {}

    if args.regen_api:
        api_client = _build_api_client(args.regen_api)
        default_models = {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}
        api_model = args.regen_model or default_models[args.regen_api]
        print(f"재생성 API: {args.regen_api} / {api_model}")

        eval_path = ROOT / args.eval if not Path(args.eval).is_absolute() else Path(args.eval)
        doc_path  = ROOT / args.docs  if not Path(args.docs).is_absolute()  else Path(args.docs)
        eval_map  = {int(d["eval_id"]): d for d in iter_jsonl(eval_path)}
        doc_map   = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}
        print(f"eval {len(eval_map)}건, docs {len(doc_map)}건 로드 완료")

    # ── 후처리 ───────────────────────────────────────────────────────────
    rows = list(iter_jsonl(in_path))
    stats = {"A": 0, "B_regen": 0, "B_skip": 0, "C": 0, "D": 0}
    out_rows = []

    for row in rows:
        original = row.get("answer", "")
        pat = _classify(original)
        fixed = original

        if pat == "A":
            fixed = _clean_A(original)
            stats["A"] += 1
            if args.dry_run:
                print(f"[{row['eval_id']}][A] → {repr(fixed[:100])}\n")

        elif pat == "B":
            if api_client is not None:
                eid  = int(row["eval_id"])
                ev   = eval_map.get(eid, {})
                question = ev.get("msg", [{}])[-1].get("content", "") if ev else ""
                topk = row.get("topk", [])
                # references에서 content 우선 사용, 없으면 doc_map 조회
                refs = row.get("references", [])
                if refs and isinstance(refs[0], dict):
                    ctx_parts = [r.get("content", "") for r in refs if r.get("content")]
                elif topk and doc_map:
                    ctx_parts = [doc_map[d] for d in topk if d in doc_map]
                else:
                    ctx_parts = []
                context = "\n\n".join(
                    f"[문서 {i+1}] {c}" for i, c in enumerate(ctx_parts)
                ) if ctx_parts else ""

                try:
                    fixed = _regen_answer(
                        api_client, api_model, question, context,
                        is_science=bool(topk),
                    )
                    stats["B_regen"] += 1
                    if args.dry_run:
                        print(f"[{row['eval_id']}][B→재생성] {repr(fixed[:100])}\n")
                except Exception as e:
                    print(f"[{row['eval_id']}] 재생성 실패 ({e}) — placeholder 사용")
                    fixed = "제공된 문서에서 관련 정보를 찾을 수 없습니다."
                    stats["B_skip"] += 1
            else:
                # API 없으면 placeholder
                fixed = "제공된 문서에서 관련 정보를 찾을 수 없습니다."
                stats["B_skip"] += 1
                if args.dry_run:
                    print(f"[{row['eval_id']}][B→placeholder] API 미지정\n")

        elif pat == "C":
            fixed = _clean_C(original)
            stats["C"] += 1
            if args.dry_run:
                print(f"[{row['eval_id']}][C] → {repr(fixed[:100])}\n")

        else:  # D
            stats["D"] += 1

        out_rows.append({**row, "answer": fixed})

    # ── 결과 출력 ────────────────────────────────────────────────────────
    total_changed = stats["A"] + stats["B_regen"] + stats["B_skip"] + stats["C"]
    print(
        f"\n총 {len(rows)}건 처리:"
        f"  [A] think앞내용 {stats['A']}건"
        f"  [B] 재생성 {stats['B_regen']}건 / placeholder {stats['B_skip']}건"
        f"  [C] CRAG추출 {stats['C']}건"
        f"  [D] 유지 {stats['D']}건"
        f"  → 변경 {total_changed}건"
    )

    if args.dry_run:
        print("(dry-run: 저장 생략)")
        return

    out_path = Path(args.output) if args.output else None
    if out_path and not out_path.is_absolute():
        out_path = ROOT / out_path

    if out_path is None:
        bak = in_path.with_suffix(".csv.bak")
        in_path.rename(bak)
        print(f"원본 백업 → {bak}")
        out_path = in_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out_rows)
    print(f"저장 완료 → {out_path}")


if __name__ == "__main__":
    main()
