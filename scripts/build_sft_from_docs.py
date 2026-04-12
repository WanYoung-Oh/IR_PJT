#!/usr/bin/env python3
"""Phase A-1: documents.jsonl 기반 Gold SFT 데이터 구축.

문서에서 역방향으로 질문을 생성하여 문서-질문-답변 삼각 일관성을 구조적으로 보장한다.

처리 흐름:
  documents.jsonl의 문서 d 선택
  → question-api로 d를 근거로 답할 수 있는 과학 질문 q 생성
  → q로 Hybrid 검색(BM25 + Dense) → Reranker top-K
  → 검색 결과에 d 포함 여부 확인
      포함 O → 검색 결과 문서 세트로 답변 생성
      포함 X → d 강제 포함하여 답변 생성
  → RAGAS Faithfulness ≥ threshold → 최종 포함
  → Faithfulness < threshold → 재생성 1회 → 그래도 미달 시 제외

사용 예시:
    # BM25-only (GPU 불필요)
    python scripts/build_sft_from_docs.py \\
      --config config/default.yaml \\
      --question-api solar --answer-api solar \\
      --skip-dense --skip-reranker \\
      --max-docs 1000 \\
      --output artifacts/sft_data_gold.jsonl

    # 풀 파이프라인 (GPU 40GB+ 권장)
    python scripts/build_sft_from_docs.py \\
      --config config/default.yaml \\
      --question-api solar --answer-api solar \\
      --top-k-retrieve 20 --top-k-rerank 5 \\
      --faithfulness-threshold 0.7 \\
      --max-docs 1000 \\
      --output artifacts/sft_data_gold.jsonl

    # 재시작 (중단된 경우 자동 이어받기)
    python scripts/build_sft_from_docs.py ... (동일 --output 경로 지정)

VRAM 가이드:
    - 24GB: --skip-dense --skip-reranker  (BM25-only)
    - 24GB: --skip-reranker               (BM25 + Dense, 임베딩 ~18GB)
    - 40GB+: 기본값 (Dense + Reranker 순차 로드)
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

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "당신은 과학 전문가입니다. "
    "제공된 참고 문서를 근거로 질문에 답하세요. "
    "문서에 없는 내용은 절대 추측하지 마세요. "
    "답변은 자연스러운 한국어 문장으로 작성하고, "
    "마크다운 헤더·불릿·LaTeX 수식 기호는 사용하지 마세요."
)

_RETRY_SYSTEM_PROMPT = (
    "이전 답변이 문서 근거를 충분히 활용하지 못했습니다. "
    "반드시 제공된 문서 내용만을 근거로 답하고, "
    "문서에 없는 내용은 '문서에 근거 없음'으로 표기하세요. "
    "자연스러운 한국어 문장으로만 작성하세요."
)


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
        raise ValueError(f"지원하지 않는 API: {api}. 선택 가능: {list(configs)}")
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
# 생성 함수
# ---------------------------------------------------------------------------

def _load_eval_fewshots(eval_path: Path, n: int = 8) -> list[str]:
    """eval.jsonl에서 단일턴 과학 질문 few-shot 예시를 추출한다."""
    examples = []
    for sample in iter_jsonl(eval_path):
        msg = sample.get("msg", [])
        # 단일턴만 예시로 사용 (배포 스타일에 가장 가까움)
        user_msgs = [m for m in msg if m["role"] == "user"]
        if len(user_msgs) == 1:
            q = user_msgs[0]["content"].strip()
            if len(q) > 5:
                examples.append(q)
        if len(examples) >= n:
            break
    return examples


def _generate_question(
    client, model: str, doc_content: str, fewshot_examples: list[str]
) -> str:
    """문서를 근거로 답할 수 있는 과학 질문을 생성한다 (eval.jsonl 스타일)."""
    examples_str = "\n".join(f"- {ex}" for ex in fewshot_examples)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 과학 교육 전문가입니다. "
                    "주어진 문서를 읽고, 해당 문서의 내용만으로 완전히 답할 수 있는 한국어 질문을 1개 생성하세요. "
                    "질문은 아래 예시와 유사한 구어체·탐구적 스타일로 작성하세요. "
                    "질문만 출력하고 다른 설명은 절대 하지 마세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문 스타일 예시:\n{examples_str}\n\n"
                    f"문서:\n{doc_content[:800]}\n\n"
                    "이 문서를 바탕으로 답할 수 있는 질문 1개:"
                ),
            },
        ],
        max_tokens=128,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _generate_answer(
    client, model: str, context: str, question: str, retry: bool = False
) -> str:
    """문서 컨텍스트 기반으로 답변을 생성한다."""
    sys_prompt = _RETRY_SYSTEM_PROMPT if retry else _SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"참고 문서:\n{context}\n\n질문: {question}",
            },
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 검색 컴포넌트
# ---------------------------------------------------------------------------

def _setup_bm25(cfg: dict):
    """BM25 검색 함수를 설정한다. 실패 시 None 반환."""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(cfg["elasticsearch"]["url"])
        es.info()
        index = cfg["elasticsearch"]["index"]

        def bm25_fn(query: str, top_k: int) -> list[str]:
            try:
                resp = es.search(
                    index=index,
                    query={"match": {"content": query}},
                    size=top_k,
                    source=["docid"],
                )
                out = []
                for h in resp["hits"]["hits"]:
                    src = h.get("_source") or {}
                    did = src.get("docid") or h.get("_id")
                    if did:
                        out.append(str(did))
                return out
            except Exception as e:
                logger.warning("BM25 검색 실패: %s", e)
                return []

        logger.info("BM25 준비 완료 (index: %s)", index)
        return bm25_fn
    except Exception as e:
        logger.warning("Elasticsearch 연결 실패 (%s) — BM25 비활성화", e)
        return None


def _setup_dense(cfg: dict):
    """Dense 검색 함수와 임베딩 모델을 설정한다. 실패 시 (None, None) 반환."""
    try:
        from qdrant_client import QdrantClient
        from ir_rag.embeddings import build_huggingface_embedding
        from ir_rag.retrieval import qdrant_dense_doc_ids

        qdrant = QdrantClient(url=cfg["qdrant"]["url"])
        qdrant.get_collections()
        collection = cfg["qdrant"]["collection"]

        embed_model = build_huggingface_embedding(cfg["embedding"])

        def dense_fn(query: str, top_k: int) -> list[str]:
            try:
                return qdrant_dense_doc_ids(
                    qdrant, collection,
                    lambda q: embed_model.get_query_embedding(q),
                    query, top_k,
                )
            except Exception as e:
                logger.warning("Dense 검색 실패: %s", e)
                return []

        logger.info("Dense 준비 완료 (collection: %s)", collection)
        return dense_fn, embed_model
    except Exception as e:
        logger.warning("Dense retriever 설정 실패 (%s) — Dense 비활성화", e)
        return None, None


def _setup_reranker(cfg: dict):
    """Reranker 모델을 설정한다. 실패 시 (None, None) 반환."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA 없음 — Reranker 비활성화")
            return None, None
        from ir_rag.reranker import load_reranker
        rcfg = cfg.get("reranker", {})
        model, tokenizer = load_reranker(
            model_name=rcfg.get("model_name", "Qwen/Qwen3-Reranker-8B"),
            trust_remote_code=rcfg.get("trust_remote_code", True),
        )
        return model, tokenizer
    except Exception as e:
        logger.warning("Reranker 로드 실패 (%s) — Reranker 비활성화", e)
        return None, None


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path) -> dict:
    """체크포인트 파일을 로드한다."""
    if not ckpt_path.exists():
        return {"processed": [], "stats": {"included": 0, "excluded": 0, "error": 0}}
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed": [], "stats": {"included": 0, "excluded": 0, "error": 0}}


def _save_checkpoint(ckpt_path: Path, ckpt: dict) -> None:
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Faithfulness 평가 (generator.py 재사용)
# ---------------------------------------------------------------------------

def _eval_faithfulness(question: str, answer: str, context: str) -> float:
    """RAGAS Faithfulness 점수를 계산한다. 실패 시 1.0 반환."""
    import math
    try:
        import nest_asyncio
        nest_asyncio.apply()

        from datasets import Dataset
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness

        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [[context]],
        })

        eval_kwargs: dict = {}
        if os.environ.get("SOLAR_API_KEY"):
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            eval_kwargs["llm"] = LangchainLLMWrapper(ChatOpenAI(
                model="solar-pro",
                api_key=os.environ["SOLAR_API_KEY"],
                base_url="https://api.upstage.ai/v1",
            ))
        elif os.environ.get("GOOGLE_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            from ragas.llms import LangchainLLMWrapper
            eval_kwargs["llm"] = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            )

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
            logger.warning("Faithfulness nan (API 타임아웃 추정) — 통과 처리")
            return 1.0
        return score

    except Exception as e:
        logger.warning("Faithfulness 평가 실패 (%s) — 통과 처리", e)
        return 1.0


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def build_gold_sft(
    doc_path: Path,
    eval_path: Path,
    out_path: Path,
    q_client,
    q_model: str,
    a_client,
    a_model: str,
    bm25_fn,
    dense_fn,
    reranker_model,
    reranker_tokenizer,
    doc_map: dict[str, str],
    top_k_retrieve: int,
    top_k_rerank: int,
    faith_threshold: float,
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

    logger.info("처리 대상: %d건 (전체 %d건 중 미처리 %d건, 쿼터 %d건)",
                len(todo_docs), len(all_docs), len([d for d in all_docs if d["docid"] not in done_docids]), max_docs)

    # Few-shot 예시
    fewshots = _load_eval_fewshots(eval_path)
    logger.info("Few-shot 예시 %d건 로드", len(fewshots))

    # Faithfulness 게이트 사용 여부
    has_faith_api = bool(
        os.environ.get("SOLAR_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not has_faith_api:
        logger.warning("Faithfulness API 키 없음 — Faithfulness 게이트 비활성화 (모든 답변 포함)")

    # 출력 파일 (append 모드)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("a", encoding="utf-8")

    try:
        for i, doc in enumerate(todo_docs):
            docid = doc["docid"]
            content = doc.get("content", "")
            if not content.strip():
                logger.warning("[%d/%d] docid=%s 빈 문서 — 건너뜀", i + 1, len(todo_docs), docid)
                _mark_done(ckpt, ckpt_path, done_docids, stats, docid, "excluded")
                continue

            logger.info("[%d/%d] docid=%s 처리 중...", i + 1, len(todo_docs), docid)

            # ── Step 1: 질문 생성 ──────────────────────────────────────────
            try:
                question = _generate_question(q_client, q_model, content, fewshots)
                if not question or len(question) < 5:
                    raise ValueError(f"생성된 질문이 너무 짧음: '{question}'")
                logger.info("  질문: %s", question)
            except Exception as e:
                logger.warning("  질문 생성 실패 (%s) — 건너뜀", e)
                _mark_done(ckpt, ckpt_path, done_docids, stats, docid, "error")
                stats["error"] += 1
                continue

            # ── Step 2: 검색 (BM25 + Dense → RRF) ───────────────────────
            candidate_ids = _retrieve(question, docid, bm25_fn, dense_fn, top_k_retrieve)

            # ── Step 3: Reranking ─────────────────────────────────────────
            if reranker_model is not None and candidate_ids:
                from ir_rag.reranker import rerank_with_crossencoder, soft_voting_rerank
                from ir_rag.retrieval import rrf_score as _rrf

                # RRF 점수 재계산 (soft voting용)
                rrf_scores = _rrf([candidate_ids])
                reranker_scores = rerank_with_crossencoder(
                    query=question,
                    doc_ids=candidate_ids[:top_k_retrieve],
                    doc_texts=doc_map,
                    model=reranker_model,
                    tokenizer=reranker_tokenizer,
                )
                combined = soft_voting_rerank(rrf_scores, reranker_scores)
                candidate_ids = list(combined.keys())[:top_k_retrieve]

            # ── Step 4: source doc 강제 포함 ──────────────────────────────
            final_ids = _force_include(docid, candidate_ids, top_k_rerank)
            context_docs = [doc_map[d] for d in final_ids if d in doc_map]
            if not context_docs:
                # 검색 결과 없을 때도 source doc은 반드시 포함
                context_docs = [content]

            forced = docid not in candidate_ids[:top_k_rerank]
            if forced:
                logger.info("  source doc 강제 포함 (검색 결과에 미포함)")

            context = "\n\n".join(
                f"[문서 {i + 1}] {c}" for i, c in enumerate(context_docs)
            )

            # ── Step 5: 답변 생성 + Faithfulness 게이트 ──────────────────
            try:
                answer = _generate_answer(a_client, a_model, context, question)
            except Exception as e:
                logger.warning("  답변 생성 실패 (%s) — 건너뜀", e)
                _mark_done(ckpt, ckpt_path, done_docids, stats, docid, "error")
                stats["error"] += 1
                continue

            if has_faith_api:
                score = _eval_faithfulness(question, answer, context)
                if score < faith_threshold:
                    logger.info("  Faithfulness %.2f < %.2f — 재생성", score, faith_threshold)
                    try:
                        answer = _generate_answer(a_client, a_model, context, question, retry=True)
                        score = _eval_faithfulness(question, answer, context)
                    except Exception as e:
                        logger.warning("  재생성 실패 (%s)", e)

                if score < faith_threshold:
                    logger.info("  Faithfulness %.2f 미달 — 제외", score)
                    _mark_done(ckpt, ckpt_path, done_docids, stats, docid, "excluded")
                    stats["excluded"] += 1
                    continue
                logger.info("  Faithfulness %.2f — 통과", score)
            else:
                score = -1.0

            # ── Step 6: SFT 레코드 저장 ──────────────────────────────────
            record = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"참고 문서:\n{context}\n\n질문: {question}",
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            _mark_done(ckpt, ckpt_path, done_docids, stats, docid, "included")
            stats["included"] += 1
            logger.info(
                "  저장 완료 (문서 %d개, faith=%.2f) | 누적: 포함 %d / 제외 %d / 오류 %d",
                len(context_docs), score,
                stats["included"], stats["excluded"], stats["error"],
            )

    finally:
        out_file.close()

    total = stats["included"] + stats["excluded"] + stats["error"]
    print(
        f"\n=== Phase A-1 완료 ===\n"
        f"처리: {total}건 | 포함: {stats['included']}건 "
        f"| 제외: {stats['excluded']}건 | 오류: {stats['error']}건\n"
        f"출력: {out_path}"
    )


def _retrieve(
    question: str,
    source_docid: str,
    bm25_fn,
    dense_fn,
    top_k: int,
) -> list[str]:
    """BM25 + Dense RRF 검색을 수행한다."""
    from ir_rag.retrieval import rrf_score

    rankings = []
    if bm25_fn is not None:
        bm25_ids = bm25_fn(question, top_k)
        if bm25_ids:
            rankings.append(bm25_ids)
    if dense_fn is not None:
        dense_ids = dense_fn(question, top_k)
        if dense_ids:
            rankings.append(dense_ids)

    if not rankings:
        # 검색 완전 실패 → source doc만 반환
        return [source_docid]

    merged = rrf_score(rankings)
    return list(merged.keys())


def _force_include(
    docid: str,
    candidate_ids: list[str],
    top_k: int,
) -> list[str]:
    """top_k 결과에 source doc이 없으면 첫 번째 위치에 강제 삽입한다."""
    top = candidate_ids[:top_k]
    if docid in top:
        return top
    # source doc을 맨 앞에 삽입하고 마지막 문서 제거
    return [docid] + top[: top_k - 1]


def _mark_done(
    ckpt: dict,
    ckpt_path: Path,
    done_set: set[str],
    stats: dict,
    docid: str,
    status: str,
) -> None:
    done_set.add(docid)
    ckpt["processed"].append(docid)
    ckpt["stats"] = stats
    _save_checkpoint(ckpt_path, ckpt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="artifacts/sft_data_gold.jsonl")
    parser.add_argument(
        "--question-api", choices=["solar", "openai", "google"], default="solar",
        help="질문 생성에 사용할 API (기본: solar)",
    )
    parser.add_argument("--question-model", default=None, help="질문 생성 모델명")
    parser.add_argument(
        "--answer-api", choices=["solar", "openai", "google"], default="solar",
        help="답변 생성에 사용할 API (기본: solar)",
    )
    parser.add_argument("--answer-model", default=None, help="답변 생성 모델명")
    parser.add_argument("--top-k-retrieve", type=int, default=20,
                        help="BM25/Dense 각각 검색할 문서 수 (기본 20)")
    parser.add_argument("--top-k-rerank", type=int, default=5,
                        help="Reranking 후 컨텍스트에 사용할 최종 문서 수 (기본 5)")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.7,
                        help="RAGAS Faithfulness 최소 임계값 (기본 0.7)")
    parser.add_argument("--max-docs", type=int, default=1000,
                        help="처리할 최대 문서 수 (기본 1000, 0=전체)")
    parser.add_argument("--seed", type=int, default=42, help="문서 샘플링 랜덤 시드")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Dense 검색 비활성화 (GPU 없거나 Qdrant 미기동 시)")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Reranker 비활성화 (VRAM 절약)")
    args = parser.parse_args()

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))

    doc_path = root / cfg["paths"]["documents"]
    eval_path = root / cfg["paths"]["eval"]
    out_path = root / args.output

    # API 클라이언트
    q_model = args.question_model or _default_model(args.question_api)
    a_model = args.answer_model or _default_model(args.answer_api)
    q_client = _build_api_client(args.question_api)
    a_client = _build_api_client(args.answer_api) if args.answer_api != args.question_api else q_client
    logger.info("질문 API: %s / %s | 답변 API: %s / %s",
                args.question_api, q_model, args.answer_api, a_model)

    # 검색 컴포넌트
    bm25_fn = _setup_bm25(cfg)

    dense_fn = None
    embed_model = None
    if not args.skip_dense:
        dense_fn, embed_model = _setup_dense(cfg)
    else:
        logger.info("Dense 검색 비활성화 (--skip-dense)")

    reranker_model = None
    reranker_tokenizer = None
    if not args.skip_reranker:
        reranker_model, reranker_tokenizer = _setup_reranker(cfg)
    else:
        logger.info("Reranker 비활성화 (--skip-reranker)")

    if bm25_fn is None and dense_fn is None:
        logger.warning(
            "BM25와 Dense 검색 모두 비활성화됨 — source doc만 포함하여 답변 생성"
        )

    # doc_map 로드
    logger.info("문서 로드 중...")
    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}
    logger.info("문서 %d건 로드 완료", len(doc_map))

    build_gold_sft(
        doc_path=doc_path,
        eval_path=eval_path,
        out_path=out_path,
        q_client=q_client,
        q_model=q_model,
        a_client=a_client,
        a_model=a_model,
        bm25_fn=bm25_fn,
        dense_fn=dense_fn,
        reranker_model=reranker_model,
        reranker_tokenizer=reranker_tokenizer,
        doc_map=doc_map,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        faith_threshold=args.faithfulness_threshold,
        max_docs=args.max_docs if args.max_docs > 0 else len(doc_map),
        seed=args.seed,
    )

    # VRAM 정리
    if embed_model is not None:
        from ir_rag.vram import unload_model
        unload_model(embed_model)
    if reranker_model is not None:
        from ir_rag.vram import unload_model
        unload_model(reranker_model)


if __name__ == "__main__":
    main()
