#!/usr/bin/env python3
"""SFT 학습 데이터 생성.

eval.jsonl + documents.jsonl + retriever → Unsloth SFT 포맷(messages JSONL) 변환.

사용 예시:
    # 기본 (BM25 검색, placeholder 답변)
    python scripts/build_sft_data.py --config config/default.yaml

    # Phase 0 CSV 활용 (치챗 제외, standalone 쿼리 재활용)
    python scripts/build_sft_data.py --phase0-csv artifacts/phase0_queries.csv

    # Solar API로 답변 생성
    python scripts/build_sft_data.py --phase0-csv artifacts/phase0_queries.csv \\
        --answer-api solar --answer-model solar-pro

    # Google Gemini로 답변 생성
    python scripts/build_sft_data.py --phase0-csv artifacts/phase0_queries.csv \\
        --answer-api google
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl, write_jsonl
from ir_rag.query_rewrite import build_search_query


def _load_phase0_csv(csv_path: Path) -> dict[int, dict]:
    """Phase 0 중간 저장 CSV를 로드한다.

    Returns
    -------
    dict[int, dict]
        eval_id → {is_science, standalone, hyde_doc, alt_query}
    """
    result: dict[int, dict] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = int(row["eval_id"])
            result[eid] = {
                "is_science": row["is_science"].strip().lower() in ("true", "1", "yes"),
                "standalone": row["standalone"],
                "hyde_doc": row.get("hyde_doc", ""),
                "alt_query": row.get("alt_query", ""),
            }
    return result


def _build_api_client(api: str):
    """외부 API 클라이언트를 반환한다 (OpenAI 호환 인터페이스)."""
    from openai import OpenAI
    if api == "solar":
        key = os.environ.get("SOLAR_API_KEY", "")
        if not key:
            raise RuntimeError("SOLAR_API_KEY가 설정되지 않았습니다. .env에 추가하세요.")
        return OpenAI(api_key=key, base_url="https://api.upstage.ai/v1")
    elif api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env에 추가하세요.")
        return OpenAI(api_key=key)
    elif api == "google":
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다. .env에 추가하세요.")
        return OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        raise ValueError(f"지원하지 않는 API: {api}")


def _clean_answer(text: str) -> str:
    """API 답변에서 LaTeX·마크다운 포맷을 제거해 자연어 텍스트로 정제한다."""
    import re

    # LaTeX 블록 수식 \[ ... \] — 내용은 유지, 구분자만 제거
    text = re.sub(r"\\\[\s*", "", text)
    text = re.sub(r"\s*\\\]", "", text)
    # LaTeX 인라인 수식 \( ... \)
    text = re.sub(r"\\\(\s*", "", text)
    text = re.sub(r"\s*\\\)", "", text)
    # $$ ... $$ / $ ... $
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)

    # LaTeX 명령어 → 자연어 치환
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\left\(", "(", text)
    text = re.sub(r"\\right\)", ")", text)
    text = re.sub(r"\\left\[", "[", text)
    text = re.sub(r"\\right\]", "]", text)
    text = text.replace(r"\times", "×")
    text = text.replace(r"\cdot", "·")
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\pm", "±")
    text = text.replace(r"\%", "%")
    text = re.sub(r"\\,", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # 나머지 \cmd{...}
    text = re.sub(r"\\[a-zA-Z]+", "", text)                 # 나머지 \cmd

    # 마크다운 헤더 (## 제목 → 제목)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 마크다운 볼드/이탤릭 (**text** / *text* → text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # 마크다운 인라인 코드 `code` → code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 과도한 개행 정리 (3줄 이상 → 2줄)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 줄 앞뒤 공백 정리
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def _generate_answer_via_api(client, model: str, messages: list[dict]) -> str:
    """외부 API로 assistant 답변을 생성한다."""
    send_msgs = [m for m in messages if m["role"] != "assistant"]
    resp = client.chat.completions.create(
        model=model,
        messages=send_msgs,
        max_tokens=512,
        temperature=0.3,
    )
    return _clean_answer(resp.choices[0].message.content.strip())


def _paraphrase_doc(client, model: str, doc_content: str) -> str:
    """문서를 다른 표현으로 의역해 데이터 증강용 문서를 생성한다."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "주어진 문서의 핵심 내용을 유지하면서 다른 문장 구조와 표현으로 의역하세요. 마크다운이나 특수 기호 없이 자연스러운 한국어 문장으로만 작성하세요."},
            {"role": "user", "content": f"다음 문서를 의역하세요:\n\n{doc_content}"},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


_SYSTEM_PROMPT = (
    "당신은 과학 전문가입니다. "
    "제공된 참고 문서를 근거로 질문에 답하세요. "
    "문서에 없는 내용은 절대 추측하지 마세요. "
    "답변은 자연스러운 한국어 문장으로 작성하고, "
    "마크다운 헤더·불릿·LaTeX 수식 기호는 사용하지 마세요."
)


def build_sft_dataset(
    eval_path: Path,
    doc_path: Path,
    retriever,
    out_path: Path,
    llm=None,
    top_k: int = 5,
    min_bm25_score: float = 0.0,
    rel_score_ratio: float = 0.7,
    min_docs: int = 3,
    phase0_map: dict[int, dict] | None = None,
    api_client=None,
    api_model: str = "solar-pro",
) -> None:
    """SFT 학습 데이터를 생성하여 JSONL 파일로 저장한다.

    Parameters
    ----------
    eval_path:
        ``eval.jsonl`` 경로.
    doc_path:
        ``documents.jsonl`` 경로.
    retriever:
        ``retriever(query, top_k, min_score) -> list[tuple[str, float]]`` 인터페이스 함수.
        (docid, score) 튜플 리스트 반환.
    out_path:
        출력 JSONL 경로.
    llm:
        쿼리 재작성용 LLM (없으면 원본 쿼리 사용).
    top_k:
        검색할 최대 문서 수.
    min_bm25_score:
        BM25 최소 점수 임계값. 0이면 비활성화.
    rel_score_ratio:
        top-1 점수 대비 유지할 최소 비율 (기본 0.7 = 70%).
        이 비율 미만인 문서는 노이즈로 간주해 제거.
    min_docs:
        컨텍스트에 포함할 최소 문서 수 (기본 3).
        필터 후 부족하면 api_client로 paraphrase 생성해 보충.
    phase0_map:
        Phase 0 CSV 로드 결과. 치챗 제외 및 standalone 쿼리 재활용에 사용.
    api_client:
        외부 API 클라이언트 (None이면 placeholder 답변).
    api_model:
        외부 API 모델명.
    """
    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}

    skipped_chitchat = 0
    records = []

    for sample in iter_jsonl(eval_path):
        eid = int(sample["eval_id"])
        msg = sample["msg"]

        # Phase 0 CSV에서 치챗 여부 확인 및 standalone 쿼리 재활용
        if phase0_map is not None:
            p0 = phase0_map.get(eid)
            if p0 is not None:
                if not p0["is_science"]:
                    skipped_chitchat += 1
                    print(f"  [{eid}] 치챗 — 건너뜀")
                    continue
                query = p0["standalone"]
            else:
                query = build_search_query(msg, llm=llm)
        else:
            query = build_search_query(msg, llm=llm)

        hits = retriever(query, top_k=top_k, min_score=min_bm25_score)

        # ── 상대 점수 필터 (top-1 대비 rel_score_ratio 미만 제거) ──────────
        if hits and rel_score_ratio > 0:
            top_score = hits[0][1]
            threshold = top_score * rel_score_ratio
            hits = [(did, score) for did, score in hits if score >= threshold]

        doc_contents: list[str] = [doc_map[did] for did, _ in hits if did in doc_map]

        # ── 최소 문서 수 보충: paraphrase 생성 ────────────────────────────
        paraphrase_count = 0
        if api_client is not None and len(doc_contents) < min_docs and doc_contents:
            needed = min_docs - len(doc_contents)
            for i in range(needed):
                src = doc_contents[i % len(doc_contents)]
                try:
                    paraphrased = _paraphrase_doc(api_client, api_model, src)
                    doc_contents.append(paraphrased)
                    paraphrase_count += 1
                except Exception as e:
                    print(f"  [{eid}] paraphrase 생성 실패 ({e})")

        context = "\n\n".join(
            f"[문서 {i + 1}] {content}"
            for i, content in enumerate(doc_contents)
        )

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for m in msg[:-1]:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({
            "role": "user",
            "content": f"참고 문서:\n{context}\n\n질문: {msg[-1]['content']}",
        })

        # 답변 생성
        if api_client is not None:
            try:
                answer = _generate_answer_via_api(api_client, api_model, messages)
                answer_src = "API"
            except Exception as e:
                print(f"  [{eid}] API 답변 생성 실패 ({e}) — placeholder 사용")
                answer = "[TODO: 고품질 답변으로 교체하세요]"
                answer_src = "placeholder"
        else:
            answer = "[TODO: 고품질 답변으로 교체하세요]"
            answer_src = "placeholder"

        messages.append({"role": "assistant", "content": answer})
        records.append({"messages": messages})
        extra = f" +paraphrase {paraphrase_count}개" if paraphrase_count else ""
        print(f"  [{eid}] 완료 — 문서 {len(doc_contents)}개{extra} / {answer_src}")

    write_jsonl(out_path, records)
    summary = f"SFT 데이터 생성 완료: {len(records)}건 → {out_path}"
    if skipped_chitchat:
        summary += f" (치챗 제외 {skipped_chitchat}건)"
    print(f"\n{summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="artifacts/sft_data.jsonl")
    parser.add_argument("--top-k", type=int, default=5,
                        help="BM25 검색 최대 문서 수 (기본 5)")
    parser.add_argument(
        "--min-bm25-score", type=float, default=1.0,
        help="BM25 최소 점수 임계값 (기본 1.0). 미달 문서 제외, 전부 미달 시 top-1 유지. 0이면 비활성화.",
    )
    parser.add_argument(
        "--rel-score-ratio", type=float, default=0.7,
        help="top-1 점수 대비 유지할 최소 비율 (기본 0.7). 미달 문서는 노이즈로 제거. 0이면 비활성화.",
    )
    parser.add_argument(
        "--min-docs", type=int, default=3,
        help="컨텍스트 최소 문서 수 (기본 3). 필터 후 부족하면 paraphrase로 보충.",
    )
    parser.add_argument(
        "--phase0-csv", default=None,
        help="Phase 0 중간 저장 CSV 경로. 지정 시 치챗 질문 제외 및 standalone 쿼리 재활용.",
    )
    parser.add_argument(
        "--answer-api", choices=["solar", "openai", "google"], default=None,
        help="외부 API로 assistant 답변 생성 (기본: placeholder 사용).",
    )
    parser.add_argument(
        "--answer-model", default=None,
        help="외부 API 모델명 (기본: solar→solar-pro, openai→gpt-4o-mini, google→gemini-2.0-flash).",
    )
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    eval_path = root / cfg["paths"]["eval"]
    doc_path = root / cfg["paths"]["documents"]
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 0 CSV 로드
    phase0_map: dict[int, dict] | None = None
    if args.phase0_csv:
        p0_path = Path(args.phase0_csv)
        if not p0_path.is_absolute():
            p0_path = root / p0_path
        phase0_map = _load_phase0_csv(p0_path)
        print(f"Phase 0 CSV 로드: {len(phase0_map)}건 → {p0_path}")

    # 외부 API 클라이언트
    api_client = None
    api_model: str = "solar-pro"
    if args.answer_api:
        api_client = _build_api_client(args.answer_api)
        default_models = {"solar": "solar-pro", "openai": "gpt-4o-mini", "google": "gemini-2.0-flash"}
        api_model = args.answer_model or default_models[args.answer_api]
        print(f"외부 API 답변 생성: {args.answer_api} / {api_model}")

    # BM25 retriever (min_score 지원)
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(cfg["elasticsearch"]["url"])
        index = cfg["elasticsearch"]["index"]

        def retriever(query: str, top_k: int, min_score: float = 0.0) -> list[tuple[str, float]]:
            try:
                search_kwargs: dict = dict(
                    index=index,
                    query={"match": {"content": query}},
                    size=top_k,
                    source=["docid"],
                )
                if min_score > 0:
                    search_kwargs["min_score"] = min_score
                resp = es.search(**search_kwargs)
                hits = resp["hits"]["hits"]

                # 최소 1개 보장: min_score로 전부 필터됐으면 top-1 폴백
                if not hits and min_score > 0:
                    resp2 = es.search(
                        index=index,
                        query={"match": {"content": query}},
                        size=1,
                        source=["docid"],
                    )
                    hits = resp2["hits"]["hits"]

                out: list[tuple[str, float]] = []
                for h in hits:
                    src = h.get("_source") or {}
                    did = src.get("docid") or h.get("_id")
                    if did:
                        out.append((str(did), float(h.get("_score", 0.0))))
                return out
            except Exception as e:
                print(f"    [경고] ES 검색 실패: {e}")
                return []

    except Exception as e:
        print(f"[경고] Elasticsearch 연결 실패 ({e}) — 빈 컨텍스트로 진행")

        def retriever(query: str, top_k: int, min_score: float = 0.0) -> list[tuple[str, float]]:
            return []

    build_sft_dataset(
        eval_path=eval_path,
        doc_path=doc_path,
        retriever=retriever,
        out_path=out_path,
        top_k=args.top_k,
        min_bm25_score=args.min_bm25_score,
        rel_score_ratio=args.rel_score_ratio,
        min_docs=args.min_docs,
        phase0_map=phase0_map,
        api_client=api_client,
        api_model=api_model,
    )


if __name__ == "__main__":
    main()
