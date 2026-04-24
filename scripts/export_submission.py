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

    # Phase 0·3 을 Solar Pro API만 사용 (로컬 LLM 미사용, SOLAR_API_KEY 필요)
    python scripts/export_submission.py --pipeline --config config/default.yaml \\
      --phase0-api solar --phase3-api solar

    # standalone / HyDE / alt 3축 RRF 가중치 (기본은 균등 1,1,1)
    python scripts/export_submission.py --pipeline --config config/default.yaml \\
      --rrf-weights 0.5,0.25,0.25

    # Phase 2.5 Listwise 재정렬 활성화 (Solar API)
    python scripts/export_submission.py --pipeline --config config/default.yaml \\
      --phase0-cache artifacts/phase0_queries.csv --phase3-api solar --listwise

    # Phase 2 캐시 재사용 (Reranker 재실행 없이 Listwise 파라미터 튜닝)
    python scripts/export_submission.py --pipeline --config config/default.yaml \\
      --phase2-cache artifacts/phase2_rerank.jsonl --phase3-api solar \\
      --listwise --listwise-n 10 --listwise-fewshot --skip-generation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.generator import format_context, generate_chitchat, generate_with_selfcheck
from ir_rag.io_util import iter_jsonl, write_jsonl
from ir_rag.listwise_reranker import listwise_rerank
from ir_rag.llm_openai_chat import OpenAIChatCompletionLLM
from ir_rag.query_rewrite import build_search_query, generate_alt_query, is_science_question
from ir_rag.reranker import load_reranker, rerank_with_crossencoder, soft_voting_rerank
from ir_rag.retrieval import (
    build_uuid_to_docid,
    es_bm25_doc_ids,
    generate_hyde_doc,
    qdrant_dense_doc_ids,
    rrf_score,
)
from ir_rag.submission import SubmissionRecord, validate_submission_row
from ir_rag.vram import unload_model

logger = logging.getLogger(__name__)


def _configure_reproducibility(seed: int) -> None:
    """임베딩·리랭커 등 PyTorch 연산 전에 호출한다.

    동일 ``seed``·정렬 타이브레이크(``reranker``/``retrieval``)로 실행 간 차이를 줄인다.
    GPU 커널·드라이버에 따라 비트 단위 완전 일치는 보장되지 않을 수 있다.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except ImportError:
        logger.warning("torch 미설치 — 시드·cuDNN 설정 생략")
        return
    print(
        f"재현성: random/numpy/torch seed={seed}, "
        "cudnn.deterministic=True, cudnn.benchmark=False"
    )


def _dump_phase2_topk(
    rerank_results: list[dict],
    eid: int,
    path: Path,
    *,
    skip_generation: bool,
    listwise_after_gate: bool,
) -> None:
    """Phase 2 직후 특정 eval_id의 topk_ids 앞 5개를 JSON으로 저장한다."""
    for r in rerank_results:
        if r.get("eid") != eid:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        topk = r.get("topk_ids") or []
        payload = {
            "eval_id": eid,
            "topk_ids_first5": topk[:5],
            "meta": {
                "skip_generation": skip_generation,
                "listwise_effective_next": listwise_after_gate,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[dump-phase2] eval_id={eid} → {path}")
        return
    logger.warning("[dump-phase2] eval_id=%s 를 rerank_results에서 찾지 못함", eid)


def _load_reasoning_llm(cfg: dict, backend: str):
    """Phase 0 / 2.5 / 3 공통: 로컬 HF 또는 Solar Pro API."""
    if backend == "solar":
        print("LLM: Solar Pro API — OpenAI 호환 (https://api.upstage.ai/v1)")
        return OpenAIChatCompletionLLM(api="solar", model="solar-pro")
    return _load_llm(cfg)


def run_pipeline(
    cfg: dict,
    root: Path,
    top_k_retrieve: int = 30,
    top_k_rerank: int = 15,
    top_k_submit: int = 3,
    skip_rewrite: bool = False,
    skip_dense: bool = False,
    bm25_weight: float = 0.6,
    dense_weight: float = 0.4,
    phase0_cache: Path | None = None,
    listwise: bool = False,
    listwise_n: int = 10,
    listwise_preserve_top1: bool = True,
    listwise_fewshot: bool = False,
    use_multi_field: bool = False,
    skip_generation: bool = False,
    phase0_backend: str = "hf",
    phase3_backend: str = "hf",
    axis_rrf_weights: tuple[float, float, float] | None = None,
    seed: int = 42,
    dump_phase2_eid: int | None = None,
    dump_phase2_path: Path | None = None,
    phase2_cache: Path | None = None,
) -> list[dict]:
    """전체 RAG 파이프라인을 실행하여 제출 행 목록을 반환한다.

    24GB VRAM 제약으로 인해 순차 로드 방식으로 동작한다:
      Phase 0.   LLM 로드 → 쿼리 재작성 + HyDE doc/alt_query 생성 → 언로드
      Phase 1.   임베딩 모델 로드 → BM25+Dense RRF(k=20) → 언로드
      Phase 2.   Reranker 로드 → 전체 리랭킹 → 언로드 → artifacts/phase2_rerank.jsonl 자동 저장
      Phase 2.5. (listwise=True) Solar API로 listwise 재정렬 (GPU 불필요)
      Phase 3.   LLM 로드 → 전체 답변 생성 → 언로드  ← skip_generation=True 시 생략

    Phase 2 캐시 재사용 (빠른 Phase 2.5 파라미터 튜닝):
      --phase2-cache artifacts/phase2_rerank.jsonl 지정 시 Phase 0-2를 건너뛰고
      Phase 2.5부터 시작한다. Reranker 재실행 없이 listwise_n / preserve_top1 /
      listwise_fewshot 조합을 빠르게 반복 실험할 수 있다.

    Parameters
    ----------
    skip_dense:
        True 이면 Dense(Qdrant) 검색을 생략하고 BM25 단독 검색을 사용한다.
    skip_generation:
        True 이면 Phase 3 LLM 답변 생성을 생략한다.
        topk 문서 선별에만 집중하는 검색 실험 시 사용. 답변은 placeholder로 대체.
    listwise:
        True 이면 Phase 2.5를 활성화한다. Reranker topk 중 상위 listwise_n개를
        Solar API (LLM)가 listwise 재정렬하여 최종 top_k_submit개를 선택한다.
    listwise_n:
        Listwise 재정렬에 넘기는 Reranker 상위 후보 수 (기본 10).
    listwise_preserve_top1:
        True이면 Reranker top-1 문서를 항상 1위로 고정 (안전 모드, 기본 True).
    listwise_fewshot:
        True이면 few-shot 예시 2개를 프롬프트에 포함.
    use_multi_field:
        True 이면 BM25 검색 시 title/keywords/summary/content 멀티필드 쿼리를 사용한다.
    phase0_backend:
        ``"hf"`` (기본) 로컬 HuggingFace LLM, ``"solar"`` Solar Pro API.
    phase3_backend:
        ``"hf"`` (기본) 로컬 LLM, ``"solar"`` Solar Pro API.
        listwise Phase 2.5도 이 백엔드를 사용한다.
    axis_rrf_weights:
        ``(standalone, HyDE, alt_query)`` 순서의 Phase 1 다축 RRF 가중치.
        ``None`` 이면 균등 ``(1.0, 1.0, 1.0)``.
    phase2_cache:
        지정 시 artifacts/phase2_rerank.jsonl 을 로드하여 Phase 0-2를 건너뛴다.
        Reranker 재실행 없이 Phase 2.5 파라미터만 바꿔 빠르게 재실험할 때 사용.
    seed:
        random / NumPy / PyTorch 시드 및 cuDNN 결정적 모드.
    """
    _configure_reproducibility(seed)

    doc_path = root / cfg["paths"]["documents"]
    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}

    # ── Phase 2 캐시 로드 시 Phase 0-2 전체 생략 ─────────────────────────────
    if phase2_cache is not None:
        print(f"=== Phase 2 캐시 로드 (Phase 0-2 생략): {phase2_cache} ===")
        rerank_results: list[dict] = []
        with open(phase2_cache, encoding="utf-8") as _pf:
            for _line in _pf:
                _line = _line.strip()
                if not _line:
                    continue
                _rec = json.loads(_line)
                rerank_results.append({
                    "eid": _rec["eval_id"],
                    "standalone": _rec["standalone"],
                    "is_science": _rec["is_science"],
                    "msg": _rec["msg"],
                    "topk_ids": _rec["topk_ids"],
                    "combined": _rec["combined"],
                })
        print(f"  캐시 로드 완료 ({len(rerank_results)}건)\n")

    else:
        # ── 정상 경로: Phase 0 → 1 → 2 ──────────────────────────────────────
        from elasticsearch import Elasticsearch
        from qdrant_client import QdrantClient

        from ir_rag.embeddings import build_huggingface_embedding

        eval_path = root / cfg["paths"]["eval"]
        uuid_to_docid = build_uuid_to_docid(str(doc_path))
        print(f"UUID→docid 역매핑 {len(uuid_to_docid)}건 로드")
        samples = list(iter_jsonl(eval_path))

        es = Elasticsearch(cfg["elasticsearch"]["url"])
        es_index = cfg["elasticsearch"]["index"]
        qdrant = QdrantClient(url=cfg["qdrant"]["url"])
        qdrant_coll = cfg["qdrant"]["collection"]

        # ── Phase 0: LLM으로 쿼리 재작성 (멀티턴만) ─────────────────────────
        standalone_queries: list[dict] = []
        if phase0_cache is not None:
            print(f"=== Phase 0: 건너뜀 (캐시 파일 사용: {phase0_cache}) ===")
            import csv as _csv
            msg_map: dict[int, list] = {int(s["eval_id"]): s["msg"] for s in samples}
            with open(phase0_cache, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    eid = int(row["eval_id"])
                    is_science = row["is_science"].strip().lower() in ("true", "1", "yes")
                    standalone_queries.append({
                        "eid": eid,
                        "msg": msg_map[eid],
                        "standalone": row["standalone"],
                        "is_science": is_science,
                        "hyde_doc": row.get("hyde_doc", ""),
                        "alt_query": row.get("alt_query", ""),
                    })
            print(f"  캐시 로드 완료 ({len(standalone_queries)}건)\n")
        elif skip_rewrite:
            print("=== Phase 0: 건너뜀 (--skip-rewrite) ===")
            for sample in samples:
                msg = sample["msg"]
                standalone_queries.append({
                    "eid": int(sample["eval_id"]),
                    "msg": msg,
                    "standalone": msg[-1]["content"],
                    "is_science": True,
                    "hyde_doc": "", "alt_query": "",
                })
        else:
            if phase0_backend == "solar":
                print("=== Phase 0: Query Rewriting (Solar Pro API) ===")
            else:
                print("=== Phase 0: Query Rewriting ===")
            print("LLM 로드 중 (쿼리 재작성용) …")
            llm = _load_reasoning_llm(cfg, phase0_backend)

            for sample in samples:
                eid = int(sample["eval_id"])
                msg = sample["msg"]
                science = is_science_question(msg, llm)
                user_msgs = [m for m in msg if m["role"] == "user"]
                if science and len(user_msgs) > 1:
                    standalone = build_search_query(msg, llm=llm)
                else:
                    standalone = user_msgs[-1]["content"] if user_msgs else ""

                hyde_doc = ""
                alt_query = ""
                if science and standalone:
                    raw_hyde = generate_hyde_doc(standalone, llm)
                    import re as _re
                    hyde_doc = _re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "", raw_hyde, flags=_re.DOTALL).strip()
                    try:
                        alt_query = generate_alt_query(standalone, llm)
                    except Exception as e:
                        print(f"  [경고] alt_query 생성 실패 (eval_id={eid}): {e}")

                standalone_queries.append({
                    "eid": eid, "msg": msg, "standalone": standalone,
                    "is_science": science,
                    "hyde_doc": hyde_doc, "alt_query": alt_query,
                })
                tag = "과학" if science else "치챗"
                turn = f"{len(user_msgs)}턴"
                print(f"  [{eid}][{tag}][{turn}] {standalone[:60]}")

            llm = unload_model(llm)
            print("LLM 언로드 완료\n")

        # ── Phase 0 결과 중간 저장 ───────────────────────────────────────────
        import csv
        phase0_out = root / "artifacts" / "phase0_queries.csv"
        if phase0_cache is not None:
            print(f"Phase 0 중간 저장 건너뜀 (캐시 파일 사용 중)\n")
        else:
            phase0_out.parent.mkdir(parents=True, exist_ok=True)
            with phase0_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["eval_id", "is_science", "standalone", "hyde_doc", "alt_query"])
                writer.writeheader()
                for sq in standalone_queries:
                    writer.writerow({
                        "eval_id":    sq["eid"],
                        "is_science": sq["is_science"],
                        "standalone": sq["standalone"],
                        "hyde_doc":   sq.get("hyde_doc", ""),
                        "alt_query":  sq.get("alt_query", ""),
                    })
            print(f"Phase 0 중간 저장 완료 → {phase0_out} ({len(standalone_queries)}건)\n")

        # ── Phase 1: 임베딩 모델로 전체 검색 ────────────────────────────────
        w0, w1, w2 = axis_rrf_weights if axis_rrf_weights is not None else (1.0, 1.0, 1.0)
        print("=== Phase 1: Hybrid Retrieval ===")
        print(f"  다축 RRF 가중치 (standalone / HyDE / alt): {w0:g}, {w1:g}, {w2:g}")
        if skip_dense:
            print("--skip-dense: BM25 단독 검색 (임베딩 모델 로드 생략)")
            embed_model = None

            def _hybrid_search(query: str) -> list[str]:
                bm25_ids = es_bm25_doc_ids(es, es_index, query, top_k_retrieve, use_multi_field=use_multi_field)
                return list(rrf_score([bm25_ids]).keys())
        else:
            print("임베딩 모델 로드 중 …")
            embed_model = build_huggingface_embedding(cfg["embedding"])

            def _hybrid_search(query: str) -> list[str]:
                vec = embed_model.get_query_embedding(query)
                bm25_ids = es_bm25_doc_ids(es, es_index, query, top_k_retrieve, use_multi_field=use_multi_field)
                dense_ids = qdrant_dense_doc_ids(qdrant, qdrant_coll, lambda _: vec, query, top_k_retrieve, uuid_to_docid=uuid_to_docid)
                return list(rrf_score([bm25_ids, dense_ids], weights=[bm25_weight, dense_weight]).keys())

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

            original_ids = _hybrid_search(standalone)

            import re as _re
            hyde_doc = sq.get("hyde_doc", "") or ""
            alt_query = sq.get("alt_query", "") or ""
            if "<think>" in hyde_doc:
                hyde_doc = _re.sub(r"<think>.*?</think>", "", hyde_doc, flags=_re.DOTALL).strip()
            if "<think>" in alt_query:
                alt_query = _re.sub(r"<think>.*?</think>", "", alt_query, flags=_re.DOTALL).strip()
            if hyde_doc and alt_query:
                hyde_ids = _hybrid_search(hyde_doc)
                alt_ids = _hybrid_search(alt_query)
                rrf = rrf_score([original_ids, hyde_ids, alt_ids], weights=[w0, w1, w2])
                axis = "3축(HyDE)"
            elif hyde_doc:
                extra_ids = _hybrid_search(hyde_doc)
                rrf = rrf_score([original_ids, extra_ids], weights=[w0, w1])
                axis = "2축(HyDE)"
            elif alt_query:
                extra_ids = _hybrid_search(alt_query)
                rrf = rrf_score([original_ids, extra_ids], weights=[w0, w2])
                axis = "2축(alt)"
            else:
                rrf = rrf_score([original_ids])
                axis = "1축(원본)"

            candidate_ids = list(rrf.keys())[:top_k_retrieve]
            retrieval_results.append({
                "eid": eid, "msg": msg, "standalone": standalone,
                "rrf": rrf, "candidate_ids": candidate_ids, "is_science": True,
            })
            print(f"  [{eid}] 검색 완료 — {axis} 후보 {len(candidate_ids)}개")

        embed_model = unload_model(embed_model)
        print("임베딩 모델 언로드 완료\n")

        # ── Phase 2: Reranker로 전체 리랭킹 ─────────────────────────────────
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

        # Phase 2 debug dump
        if dump_phase2_eid is not None and dump_phase2_path is not None:
            _dump_phase2_topk(
                rerank_results,
                dump_phase2_eid,
                dump_phase2_path,
                skip_generation=skip_generation,
                listwise_after_gate=listwise,
            )

        # ── Phase 2 결과 자동 저장 (Phase 2.5 파라미터 재실험용) ─────────────
        _p2_out = root / "artifacts" / "phase2_rerank.jsonl"
        _p2_out.parent.mkdir(parents=True, exist_ok=True)
        with open(_p2_out, "w", encoding="utf-8") as _f2:
            for _r in rerank_results:
                json.dump({
                    "eval_id":    _r["eid"],
                    "standalone": _r["standalone"],
                    "is_science": _r["is_science"],
                    "msg":        _r["msg"],
                    "topk_ids":   _r.get("topk_ids", []),
                    "combined":   _r.get("combined", {}),
                }, _f2, ensure_ascii=False)
                _f2.write("\n")
        print(f"Phase 2 캐시 저장 완료 → {_p2_out} ({len(rerank_results)}건)\n")

    # ── Phase 2.5: Listwise 재정렬 (선택) ────────────────────────────────────
    if listwise:
        print("=== Phase 2.5: Listwise Reranking (Solar API) ===")
        print(f"  파라미터: listwise_n={listwise_n}, preserve_top1={listwise_preserve_top1}, fewshot={listwise_fewshot}")
        llm_lw = _load_reasoning_llm(cfg, phase3_backend)
        _lw_applied = 0
        for r in rerank_results:
            if not r["is_science"] or not r.get("topk_ids"):
                r["topk_ids_listwise"] = r.get("topk_ids", [])[:top_k_submit]
                continue
            pool_ids = r["topk_ids"][:listwise_n]
            docs_pool = [{"docid": did, "content": doc_map.get(did, "")} for did in pool_ids]
            reordered = listwise_rerank(
                query=r["standalone"],
                docs=docs_pool,
                llm=llm_lw,
                preserve_top1=listwise_preserve_top1,
                use_fewshot=listwise_fewshot,
            )
            r["topk_ids_listwise"] = [d["docid"] for d in reordered[:top_k_submit]]
            _lw_applied += 1
            print(f"  [{r['eid']}] pool={pool_ids[:3]}… → {r['topk_ids_listwise']}")
        llm_lw = unload_model(llm_lw)
        print(f"LLM(listwise) 언로드 완료 (적용 {_lw_applied}건)\n")
    else:
        for r in rerank_results:
            r["topk_ids_listwise"] = r.get("topk_ids", [])[:top_k_submit]

    # ── Phase 3: LLM으로 전체 답변 생성 ─────────────────────────────────────
    rows_out: list[dict] = []

    if skip_generation:
        print("=== Phase 3: 생략 (--skip-generation) ===")
        print("topk 문서 선별 결과만 출력합니다. 답변은 placeholder로 대체됩니다.")
        for r in rerank_results:
            if not r["is_science"]:
                final_ids = []
                refs = []
            else:
                final_ids = r["topk_ids_listwise"]
                refs = [
                    {"score": float(r["combined"].get(did, 0.0)), "content": doc_map.get(did, "")}
                    for did in final_ids
                ]
            rec = SubmissionRecord(
                eval_id=r["eid"],
                standalone_query=r["standalone"],
                topk=final_ids,
                answer="[RETRIEVAL ONLY]",
                references=refs,
            )
            row = rec.to_dict()
            validate_submission_row(row)
            rows_out.append(row)
            print(f"  [{r['eid']}] → {final_ids}")
        return rows_out

    print("=== Phase 3: LLM 답변 생성 ===")
    print("LLM 로드 중 …")
    llm = _load_reasoning_llm(cfg, phase3_backend)

    for r in rerank_results:
        question = r["msg"][-1]["content"]
        if not r["is_science"]:
            answer = generate_chitchat(question, llm)
            refs = []
            topk_final: list[str] = []
        else:
            final_ids = r["topk_ids_listwise"]
            context = format_context(final_ids, doc_map, top_k=top_k_submit)
            answer = generate_with_selfcheck(question, context, llm)
            refs = [
                {"score": float(r["combined"].get(did, 0.0)), "content": doc_map.get(did, "")}
                for did in final_ids
            ]
            topk_final = final_ids
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
        print(f"  [{r['eid']}] {r['standalone'][:40]}… → 제출 topk: {topk_final}")

    llm = unload_model(llm)
    return rows_out


def _load_llm(cfg: dict):
    """config에 지정된 LLM을 로드한다.

    우선순위:
    1. vllm.url 설정 시 → vLLM OpenAI 호환 API
    2. llm.checkpoint 설정 시 →
       - ``<checkpoint>/merged`` 에 ``train_sft`` 병합 저장이 있으면 PEFT 없이 직접 로드
       - 또는 checkpoint 폴더가 어댑터가 아닌 통합 모델이면 직접 로드
       - 그 외 PEFT LoRA 어댑터 병합 후 로드
    3. 기본 → HuggingFace hub에서 llm.model_name 직접 로드
    """
    vllm_url = cfg.get("vllm", {}).get("url")
    if vllm_url:
        from llama_index.llms.openai_like import OpenAILike
        model_name = cfg.get("vllm", {}).get("model_name", "science-rag")
        print(f"LLM: vLLM 서버 사용 ({vllm_url}, model={model_name})")
        return OpenAILike(
            model=model_name,
            api_base=f"{vllm_url}/v1",
            api_key="dummy",
            is_chat_model=True,
        )

    from llama_index.llms.huggingface import HuggingFaceLLM
    llm_cfg = cfg.get("llm", {})
    base_model = llm_cfg.get("model_name", "Qwen/Qwen3.5-9B")
    checkpoint = llm_cfg.get("checkpoint") or None
    max_new_tokens = llm_cfg.get("max_new_tokens", 512)
    context_window = llm_cfg.get("context_window", 4096)

    if checkpoint:
        import os
        from pathlib import Path
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            from ir_rag.config import repo_root_from
            ckpt_path = repo_root_from(Path.cwd()) / ckpt_path

        trust_remote = llm_cfg.get("trust_remote_code", True)
        print(f"LLM: 체크포인트 로드 — base={base_model}, ckpt={ckpt_path}")

        def _hf_llm_from_dir(model_dir: Path, label: str):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"  {label} → {model_dir}")
            tok = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=trust_remote,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=trust_remote,
            )
            return HuggingFaceLLM(
                model=mdl,
                tokenizer=tok,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )

        merged_subdir = ckpt_path / "merged"
        if merged_subdir.is_dir() and (merged_subdir / "config.json").is_file():
            return _hf_llm_from_dir(merged_subdir, "병합 모델 직접 로드 (PEFT 스킵)")

        if (ckpt_path / "config.json").is_file() and not (ckpt_path / "adapter_config.json").is_file():
            return _hf_llm_from_dir(ckpt_path, "통합 체크포인트 직접 로드")

        try:
            from peft import PeftModel, PeftConfig
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            peft_cfg = PeftConfig.from_pretrained(str(ckpt_path))
            actual_base = peft_cfg.base_model_name_or_path or base_model
            tokenizer = AutoTokenizer.from_pretrained(
                actual_base,
                trust_remote_code=trust_remote,
            )
            model = AutoModelForCausalLM.from_pretrained(
                actual_base,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=trust_remote,
            )
            model = PeftModel.from_pretrained(model, str(ckpt_path))
            model = model.merge_and_unload()
            print(f"  PEFT 어댑터 병합 완료 (base: {actual_base})")
            return HuggingFaceLLM(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
        except Exception as e:
            print(f"  [경고] PEFT 로드 실패 ({e}), base 모델로 fallback: {ckpt_path}")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                str(ckpt_path),
                local_files_only=True,
                trust_remote_code=trust_remote,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=trust_remote,
            )
            return HuggingFaceLLM(
                model=_model,
                tokenizer=_tokenizer,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )

    print(f"LLM: HuggingFace 모델 로드 — {base_model}")
    return HuggingFaceLLM(
        model_name=base_model,
        tokenizer_name=base_model,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
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
    parser.add_argument("--top-k-retrieve", type=int, default=20)
    parser.add_argument("--top-k-rerank", type=int, default=10)
    parser.add_argument("--skip-dense", action="store_true",
                        help="Dense 검색 생략 — BM25 단독 검색 (임베딩 모델 불필요)")
    parser.add_argument("--bm25-weight", type=float, default=0.7,
                        help="Hybrid RRF에서 BM25 가중치 (기본 0.7)")
    parser.add_argument("--dense-weight", type=float, default=0.3,
                        help="Hybrid RRF에서 Dense 가중치 (기본 0.3)")
    parser.add_argument("--skip-rewrite", action="store_true",
                        help="Phase 0 건너뜀 — 원본 쿼리로 Phase 1~3만 점검")
    parser.add_argument("--phase0-cache", metavar="CSV",
                        help="Phase 0 캐시 CSV 경로 지정 시 Phase 0을 건너뛰고 해당 파일로 Phase 1~3 실행")
    parser.add_argument("--phase2-cache", metavar="JSONL",
                        help="Phase 2 캐시 JSONL 경로 지정 시 Phase 0-2를 건너뛰고 Phase 2.5부터 실행. "
                             "Reranker 재실행 없이 listwise 파라미터 튜닝에 사용.")
    parser.add_argument("--listwise", action="store_true",
                        help="Phase 2.5 활성화: Reranker topk 중 상위 --listwise-n개를 "
                             "Solar API(LLM)가 listwise 재정렬하여 최종 top-k 선택")
    parser.add_argument("--listwise-n", type=int, default=10,
                        help="Listwise 재정렬에 넘기는 Reranker 상위 후보 수 (기본 10)")
    parser.add_argument("--no-preserve-top1", action="store_false", dest="listwise_preserve_top1",
                        help="Listwise preserve_top1 비활성 — Reranker top-1을 고정하지 않음 "
                             "(기본은 top-1 고정 안전 모드)")
    parser.add_argument("--listwise-fewshot", action="store_true",
                        help="Listwise 프롬프트에 few-shot 예시 2개 포함")
    parser.add_argument("--multi-field", action="store_true",
                        help="BM25 검색 시 title/keywords/summary/content 멀티필드 쿼리 사용 (F-2 색인 후)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Phase 3 LLM 답변 생성 생략. topk 문서 선별만 수행하여 검색 실험 속도 향상."
                             " 답변은 placeholder로 대체됨. MAP(문서 일치율) 측정용.")
    parser.add_argument(
        "--phase0-api",
        choices=("hf", "solar"),
        default="hf",
        help="Phase 0: standalone·HyDE·alt_query·과학판별 — hf=로컬 LLM, solar=Solar Pro API",
    )
    parser.add_argument(
        "--phase3-api",
        choices=("hf", "solar"),
        default="hf",
        help="Phase 3: answer 생성 — hf=로컬 LLM, solar=Solar Pro API. "
             "--listwise 시 Phase 2.5도 동일 백엔드 사용.",
    )
    parser.add_argument(
        "--rrf-weights",
        default=None,
        metavar="W,W,W",
        help="Phase 1에서 standalone / HyDE / alt_query 다축 RRF 가중치 (쉼표로 3개). "
             "예: 0.5,0.25,0.25. 기본은 균등 1,1,1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="--pipeline 시 random/NumPy/torch 시드 및 cuDNN 결정적 모드 (기본 42).",
    )
    parser.add_argument(
        "--dump-phase2-eid",
        type=int,
        default=None,
        metavar="EID",
        help="Phase 2 직후 해당 eval_id의 topk_ids 상위 5개를 --dump-phase2-file에 JSON 저장",
    )
    parser.add_argument(
        "--dump-phase2-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="--dump-phase2-eid 와 함께 사용 (미지정 시 artifacts/phase2_dump_eid{EID}.json)",
    )
    parser.set_defaults(listwise_preserve_top1=True)
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
        phase0_cache_path = Path(args.phase0_cache) if args.phase0_cache else None
        phase2_cache_path = Path(args.phase2_cache) if args.phase2_cache else None

        if phase2_cache_path is not None:
            print(f"실제 파이프라인 모드 실행 중 … (Phase 2 캐시: {phase2_cache_path})")
        elif phase0_cache_path is not None:
            print(f"실제 파이프라인 모드 실행 중 … (Phase 0 캐시: {phase0_cache_path})")
        else:
            print("실제 파이프라인 모드 실행 중 …")

        axis_rrf_weights: tuple[float, float, float] | None = None
        if args.rrf_weights:
            parts = [p.strip() for p in args.rrf_weights.split(",") if p.strip()]
            if len(parts) != 3:
                parser.error(
                    "--rrf-weights 는 세 개의 숫자를 쉼표로 구분해 주세요. 예: 0.5,0.25,0.25"
                )
            try:
                axis_rrf_weights = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError as e:
                parser.error(f"--rrf-weights 파싱 실패: {e}")

        dump_eid = args.dump_phase2_eid
        dump_path = args.dump_phase2_file
        if dump_eid is not None and dump_path is None:
            dump_path = root / "artifacts" / f"phase2_dump_eid{dump_eid}.json"
        if dump_path is not None and dump_eid is None:
            parser.error("--dump-phase2-file 은 --dump-phase2-eid 와 함께 지정하세요.")

        rows_out = run_pipeline(
            cfg, root,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            top_k_submit=top_k,
            skip_rewrite=args.skip_rewrite,
            skip_dense=args.skip_dense,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            phase0_cache=phase0_cache_path,
            listwise=args.listwise,
            listwise_n=args.listwise_n,
            listwise_preserve_top1=args.listwise_preserve_top1,
            listwise_fewshot=args.listwise_fewshot,
            use_multi_field=args.multi_field,
            skip_generation=args.skip_generation,
            phase0_backend=args.phase0_api,
            phase3_backend=args.phase3_api,
            axis_rrf_weights=axis_rrf_weights,
            seed=args.seed,
            dump_phase2_eid=dump_eid,
            dump_phase2_path=dump_path,
            phase2_cache=phase2_cache_path,
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
