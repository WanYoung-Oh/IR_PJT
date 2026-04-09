#!/usr/bin/env python3
"""제출 파일 작성. 확장자는 `.csv` 이지만 한 줄당 JSON 한 개 (베이스라인 관례).

필드: eval_id, standalone_query, topk, answer, references — data/sample_submission.csv 와 동일.

모드:
  --placeholder   형식 검증용 더미 값 (리더보드 유효 제출 아님)
  --pipeline      실제 RAG 파이프라인 실행 (ES + Reranker + Generator 필요)

사용 예시:
    # 더미 제출 (형식 검증)
    python scripts/export_submission.py --placeholder --config config/default.yaml

    # 실제 파이프라인 실행
    python scripts/export_submission.py --pipeline --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.generator import format_context, generate_chitchat, generate_with_selfcheck
from ir_rag.io_util import iter_jsonl, write_jsonl
from ir_rag.query_rewrite import build_search_query, is_science_question
from ir_rag.reranker import load_reranker, rerank_with_crossencoder, soft_voting_rerank
from ir_rag.retrieval import es_bm25_doc_ids, rrf_score
from ir_rag.submission import SubmissionRecord, validate_submission_row
from ir_rag.vram import unload_model


def run_pipeline(
    cfg: dict,
    root: Path,
    top_k_retrieve: int = 100,
    top_k_rerank: int = 20,
    top_k_submit: int = 3,
    skip_rewrite: bool = False,
) -> list[dict]:
    """전체 RAG 파이프라인을 실행하여 제출 행 목록을 반환한다.

    24GB VRAM 제약으로 인해 3-Phase 순차 로드 방식으로 동작한다:
      Phase 1. 임베딩 모델 로드 → 전체 쿼리 검색(BM25+Dense RRF) → 언로드
      Phase 2. Reranker 로드 → 전체 리랭킹 → 언로드
      Phase 3. LLM 로드 → 전체 답변 생성 → 언로드
    """
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient

    from ir_rag.embeddings import build_huggingface_embedding
    from ir_rag.retrieval import qdrant_dense_doc_ids

    eval_path = root / cfg["paths"]["eval"]
    doc_path = root / cfg["paths"]["documents"]

    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}
    samples = list(iter_jsonl(eval_path))

    es = Elasticsearch(cfg["elasticsearch"]["url"])
    es_index = cfg["elasticsearch"]["index"]
    qdrant = QdrantClient(url=cfg["qdrant"]["url"])
    qdrant_coll = cfg["qdrant"]["collection"]

    # ── Phase 0: LLM으로 쿼리 재작성 (멀티턴만) ─────────────────────────────
    standalone_queries: list[dict] = []
    if skip_rewrite:
        print("=== Phase 0: 건너뜀 (--skip-rewrite) ===")
        for sample in samples:
            msg = sample["msg"]
            standalone_queries.append({
                "eid": int(sample["eval_id"]),
                "msg": msg,
                "standalone": msg[-1]["content"],
                "is_science": True,
            })
    else:
        print("=== Phase 0: Query Rewriting ===")
        print("LLM 로드 중 (쿼리 재작성용) …")
        llm = _load_llm(cfg)

        for sample in samples:
            eid = int(sample["eval_id"])
            msg = sample["msg"]
            science = is_science_question(msg, llm)
            standalone = build_search_query(msg, llm=llm) if science else msg[-1]["content"]
            standalone_queries.append({
                "eid": eid, "msg": msg, "standalone": standalone, "is_science": science,
            })
            tag = "과학" if science else "치챗"
            print(f"  [{eid}][{tag}] {standalone[:60]}")

        llm = unload_model(llm)
        print("LLM 언로드 완료\n")

    # ── Phase 1: 임베딩 모델로 전체 검색 ────────────────────────────────────
    print("=== Phase 1: Hybrid Retrieval ===")
    print("임베딩 모델 로드 중 …")
    embed_model = build_huggingface_embedding(cfg["embedding"])

    retrieval_results: list[dict] = []
    for sq in standalone_queries:
        eid, msg, standalone, science = sq["eid"], sq["msg"], sq["standalone"], sq["is_science"]
        if not science:
            retrieval_results.append({
                "eid": eid, "msg": msg, "standalone": standalone,
                "rrf": {}, "candidate_ids": [], "is_science": False,
            })
            print(f"  [{eid}] 치챗 — 검색 건너뜀")
            continue
        vec = embed_model.get_query_embedding(standalone)
        bm25_ids = es_bm25_doc_ids(es, es_index, standalone, top_k_retrieve)
        dense_ids = qdrant_dense_doc_ids(qdrant, qdrant_coll, lambda _: vec, standalone, top_k_retrieve)
        rrf = rrf_score([bm25_ids, dense_ids])
        candidate_ids = list(rrf.keys())[:top_k_retrieve]
        retrieval_results.append({
            "eid": eid, "msg": msg, "standalone": standalone,
            "rrf": rrf, "candidate_ids": candidate_ids, "is_science": True,
        })
        print(f"  [{eid}] 검색 완료 — 후보 {len(candidate_ids)}개")

    embed_model = unload_model(embed_model)
    print("임베딩 모델 언로드 완료\n")

    # ── Phase 2: Reranker로 전체 리랭킹 ─────────────────────────────────────
    print("=== Phase 2: Reranking ===")
    reranker_cfg = cfg.get("reranker", {})
    reranker_model_name = reranker_cfg.get("model_name", "Qwen/Qwen3-Reranker-8B")
    reranker_trust_rc = bool(reranker_cfg.get("trust_remote_code", False))
    print(f"Reranker 로드 중 … ({reranker_model_name})")
    reranker, reranker_tok = load_reranker(reranker_model_name, trust_remote_code=reranker_trust_rc)

    rerank_results: list[dict] = []
    for r in retrieval_results:
        if not r["is_science"]:
            rerank_results.append({**r, "combined": {}, "topk_ids": []})
            continue
        reranker_scores = rerank_with_crossencoder(
            r["standalone"], r["candidate_ids"], doc_map, reranker, reranker_tok
        )
        combined = soft_voting_rerank(r["rrf"], reranker_scores)
        topk_ids = list(combined.keys())[:top_k_rerank]
        rerank_results.append({**r, "combined": combined, "topk_ids": topk_ids})
        print(f"  [{r['eid']}] 리랭킹 완료 — top{top_k_rerank}")

    reranker = unload_model(reranker)
    print("Reranker 언로드 완료\n")

    # ── Phase 3: LLM으로 전체 답변 생성 ─────────────────────────────────────
    print("=== Phase 3: LLM 답변 생성 ===")
    print("LLM 로드 중 …")
    llm = _load_llm(cfg)

    rows_out: list[dict] = []
    for r in rerank_results:
        question = r["msg"][-1]["content"]
        if not r["is_science"]:
            answer = generate_chitchat(question, llm)
            refs = []
            topk_final: list[str] = []
        else:
            context = format_context(r["topk_ids"], doc_map, top_k=top_k_submit)
            answer = generate_with_selfcheck(question, context, llm)
            refs = [
                {"score": float(r["combined"].get(did, 0.0)), "content": doc_map.get(did, "")}
                for did in r["topk_ids"][:top_k_submit]
            ]
            topk_final = r["topk_ids"][:top_k_submit]
        rec = SubmissionRecord(
            eval_id=r["eid"],
            standalone_query=r["standalone"],
            topk=topk_final,
            answer=answer,
            references=refs,
        )
        row = rec.to_dict()
        validate_submission_row(row)
        rows_out.append(row)
        print(f"  [{r['eid']}] {r['standalone'][:40]}… → {r['topk_ids'][:top_k_submit]}")

    llm = unload_model(llm)
    return rows_out


def _load_llm(cfg: dict):
    """config에 지정된 LLM을 로드한다.

    vllm_url 이 설정된 경우 vLLM OpenAI 호환 API를 사용하고,
    없으면 LlamaIndex HuggingFaceLLM으로 fallback한다.
    """
    vllm_url = cfg.get("vllm", {}).get("url")
    if vllm_url:
        from llama_index.llms.openai_like import OpenAILike
        model_name = cfg.get("vllm", {}).get("model_name", "science-rag")
        return OpenAILike(
            model=model_name,
            api_base=f"{vllm_url}/v1",
            api_key="dummy",
            is_chat_model=True,
        )

    from llama_index.llms.huggingface import HuggingFaceLLM
    llm_cfg = cfg.get("llm", {})
    return HuggingFaceLLM(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen3.5-4B"),
        tokenizer_name=llm_cfg.get("model_name", "Qwen/Qwen3.5-4B"),
        max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        context_window=llm_cfg.get("context_window", 4096),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "-o", "--output",
        default="artifacts/sample_submission.csv",
    )
    parser.add_argument("--placeholder", action="store_true",
                        help="형식 검증용 더미 값")
    parser.add_argument("--dummy-topk", action="store_true",
                        help="플레이스홀더 시 topk를 무작위 docid 3개로 채움")
    parser.add_argument("--pipeline", action="store_true",
                        help="실제 RAG 파이프라인 실행 (ES + Reranker + LLM 필요)")
    parser.add_argument("--top-k-retrieve", type=int, default=100)
    parser.add_argument("--top-k-rerank", type=int, default=20)
    parser.add_argument("--skip-rewrite", action="store_true",
                        help="Phase 0 건너뜀 — 원본 쿼리로 Phase 1~3만 점검")
    args = parser.parse_args()

    if not args.placeholder and not args.pipeline:
        parser.error("--placeholder 또는 --pipeline 중 하나를 지정하세요.")

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    eval_path = root / cfg["paths"]["eval"]
    top_k = int(cfg.get("submission_top_k", 3))
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.pipeline:
        print("실제 파이프라인 모드 실행 중 …")
        rows_out = run_pipeline(
            cfg, root,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            top_k_submit=top_k,
            skip_rewrite=args.skip_rewrite,
        )
    else:
        rows_out = []
        for sample in iter_jsonl(eval_path):
            eid = int(sample["eval_id"])
            msg = sample["msg"]
            standalone = build_search_query(msg, llm=None)

            if args.dummy_topk:
                topk = [str(uuid.uuid4()) for _ in range(top_k)]
                refs = [{"score": 0.0, "content": "[PLACEHOLDER]"} for _ in topk]
            else:
                topk, refs = [], []

            answer = "[PLACEHOLDER] 파이프라인·LLM 연결 후 실제 답변으로 교체하세요."
            rec = SubmissionRecord(
                eval_id=eid, standalone_query=standalone,
                topk=topk, answer=answer, references=refs,
            )
            row = rec.to_dict()
            validate_submission_row(row)
            rows_out.append(row)

    write_jsonl(out, rows_out)
    print(f"Wrote {len(rows_out)} lines → {out}")


if __name__ == "__main__":
    main()
