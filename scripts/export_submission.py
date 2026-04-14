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
from ir_rag.llm_openai_chat import OpenAIChatCompletionLLM
from ir_rag.query_rewrite import build_search_query, generate_alt_query, is_science_question
from ir_rag.reranker import load_reranker, rerank_with_crossencoder, soft_voting_rerank
from ir_rag.retrieval import es_bm25_doc_ids, generate_hyde_doc, qdrant_dense_doc_ids, rrf_score
from ir_rag.submission import SubmissionRecord, validate_submission_row
from ir_rag.vram import unload_model


def _llm_select_docs(
    query: str,
    candidate_ids: list[str],
    doc_map: dict[str, str],
    llm: any,
    top_k: int = 3,
) -> list[str]:
    """Phase 2.5: LLM 프롬프트로 top-k 문서를 선별한다.

    Reranker 점수가 놓치는 의미적 관련성을 LLM이 포착.
    파싱 실패 또는 선별 수 부족 시 candidate_ids[:top_k] 폴백.

    Parameters
    ----------
    candidate_ids:
        Reranker top-k 결과 docid 목록.
    doc_map:
        docid → 본문 매핑.
    llm:
        ``llm.complete(prompt)`` 인터페이스를 가진 LLM 객체.
    """
    import re

    doc_lines = []
    valid_ids = [d for d in candidate_ids if d in doc_map]
    for i, did in enumerate(valid_ids, 1):
        snippet = doc_map[did][:250].replace("\n", " ")
        doc_lines.append(f"[문서 {i}] {snippet}")
    doc_list = "\n".join(doc_lines)

    prompt = (
        f"질문: {query}\n\n"
        f"아래 문서 중 질문에 가장 정확하게 답할 수 있는 문서 {top_k}개를 선택하세요.\n"
        f"문서 번호만 콤마로 구분하여 출력하세요. 예: 1, 3, 5\n\n"
        f"{doc_list}\n\n"
        f"선택한 문서 번호 ({top_k}개):"
    )
    try:
        raw = llm.complete(prompt).text
        # <think> 태그 제거
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # 숫자 파싱
        nums = [int(x.strip()) for x in re.split(r"[,\s]+", raw) if x.strip().isdigit()]
        # 유효 범위 필터 (1-indexed)
        selected = [valid_ids[n - 1] for n in nums if 1 <= n <= len(valid_ids)]
        # 중복 제거, top_k개 확보
        seen: set[str] = set()
        result = []
        for did in selected:
            if did not in seen:
                seen.add(did)
                result.append(did)
            if len(result) >= top_k:
                break
        if len(result) >= top_k:
            return result
    except Exception:
        pass
    # 폴백: Reranker 점수 순 상위 top_k
    return valid_ids[:top_k]


def _load_reasoning_llm(cfg: dict, backend: str):
    """Phase 0 / 2.5 / 3 공통: 로컬 HF 또는 Solar Pro API."""
    if backend == "solar":
        print("LLM: Solar Pro API — OpenAI 호환 (https://api.upstage.ai/v1)")
        return OpenAIChatCompletionLLM(api="solar", model="solar-pro")
    return _load_llm(cfg)


def run_pipeline(
    cfg: dict,
    root: Path,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 10,
    top_k_submit: int = 3,
    skip_rewrite: bool = False,
    skip_dense: bool = False,
    bm25_weight: float = 0.7,
    dense_weight: float = 0.3,
    phase0_cache: Path | None = None,
    llm_select: bool = False,
    use_multi_field: bool = False,
    skip_generation: bool = False,
    phase0_backend: str = "hf",
    phase3_backend: str = "hf",
    axis_rrf_weights: tuple[float, float, float] | None = None,
) -> list[dict]:
    """전체 RAG 파이프라인을 실행하여 제출 행 목록을 반환한다.

    24GB VRAM 제약으로 인해 순차 로드 방식으로 동작한다:
      Phase 0.   LLM 로드 → 쿼리 재작성 + HyDE doc/alt_query 생성 → 언로드
      Phase 1.   임베딩 모델 로드 → BM25+Dense RRF(k=20) → 언로드
      Phase 2.   Reranker 로드 → 전체 리랭킹 → 언로드
      Phase 2.5. (llm_select=True) LLM 로드 → top-k 문서 선별 → 언로드
      Phase 3.   LLM 로드 → 전체 답변 생성 → 언로드  ← skip_generation=True 시 생략

    Parameters
    ----------
    skip_dense:
        True 이면 Dense(Qdrant) 검색을 생략하고 BM25 단독 검색을 사용한다.
        Dense 인덱스 오염 확인 후 기본값으로 사용 권장.
    skip_generation:
        True 이면 Phase 3 LLM 답변 생성을 생략한다.
        topk 문서 선별에만 집중하는 검색 실험 시 사용. 답변은 placeholder로 대체.
        Phase 0도 --phase0-cache와 함께 사용하면 LLM 로드 없이 검색+재순위만 실행.
    llm_select:
        True 이면 Phase 2.5를 활성화한다. Reranker 이후 LLM이 최종 문서를 선별한다.
        skip_generation=True 시 자동으로 비활성화된다.
    use_multi_field:
        True 이면 BM25 검색 시 title/keywords/summary/content 멀티필드 쿼리를 사용한다.
        F-2 메타 색인 완료 후 활성화한다.
    phase0_backend:
        ``"hf"`` (기본) 로컬 HuggingFace LLM, ``"solar"`` Solar Pro API
        (standalone / HyDE / alt_query / 과학 판별).
    phase3_backend:
        ``"hf"`` (기본) 로컬 LLM, ``"solar"`` Solar Pro API (answer 생성 및
        ``--llm-select`` 시 문서 선별).
    axis_rrf_weights:
        ``(standalone, HyDE, alt_query)`` 순서의 Phase 1 다축 RRF 가중치.
        ``None`` 이면 균등 ``(1.0, 1.0, 1.0)``. 2축일 때는 (standalone, 존재하는 축)에 매핑.
    """
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient

    from ir_rag.embeddings import build_huggingface_embedding
    from ir_rag.retrieval import build_uuid_to_docid, qdrant_dense_doc_ids

    eval_path = root / cfg["paths"]["eval"]
    doc_path = root / cfg["paths"]["documents"]

    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}
    uuid_to_docid = build_uuid_to_docid(str(doc_path))
    print(f"UUID→docid 역매핑 {len(uuid_to_docid)}건 로드")
    samples = list(iter_jsonl(eval_path))

    es = Elasticsearch(cfg["elasticsearch"]["url"])
    es_index = cfg["elasticsearch"]["index"]
    qdrant = QdrantClient(url=cfg["qdrant"]["url"])
    qdrant_coll = cfg["qdrant"]["collection"]

    # ── Phase 0: LLM으로 쿼리 재작성 (멀티턴만) ─────────────────────────────
    standalone_queries: list[dict] = []
    if phase0_cache is not None:
        print(f"=== Phase 0: 건너뜀 (캐시 파일 사용: {phase0_cache}) ===")
        import csv as _csv
        # eval_id → msg 매핑 (msg는 CSV에 없으므로 eval.jsonl에서 로드)
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
                "hyde_doc": "", "alt_query": "",   # skip_rewrite 시 HyDE 없이 1축 동작
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
            # 멀티턴만 standalone 재작성, 단일턴은 원본 그대로 사용
            user_msgs = [m for m in msg if m["role"] == "user"]
            if science and len(user_msgs) > 1:
                standalone = build_search_query(msg, llm=llm)
            else:
                standalone = user_msgs[-1]["content"] if user_msgs else ""

            # 과학 쿼리에 HyDE 문서 + alt_query 생성 (3축 BM25 RRF 활성화)
            # <think> 태그는 strip 후 사용 (버리지 않음)
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

    # ── Phase 0 결과 중간 저장 (캐시 파일 지정 시 건너뜀) ───────────────────
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

    # ── Phase 1: 임베딩 모델로 전체 검색 ────────────────────────────────────
    w0, w1, w2 = axis_rrf_weights if axis_rrf_weights is not None else (1.0, 1.0, 1.0)
    print("=== Phase 1: Hybrid Retrieval ===")
    print(f"  다축 RRF 가중치 (standalone / HyDE / alt): {w0:g}, {w1:g}, {w2:g}")
    if skip_dense:
        print("--skip-dense: BM25 단독 검색 (임베딩 모델 로드 생략)")
        embed_model = None

        def _hybrid_search(query: str) -> list[str]:
            """BM25 단독 검색."""
            bm25_ids = es_bm25_doc_ids(es, es_index, query, top_k_retrieve, use_multi_field=use_multi_field)
            return list(rrf_score([bm25_ids]).keys())
    else:
        print("임베딩 모델 로드 중 …")
        embed_model = build_huggingface_embedding(cfg["embedding"])

        def _hybrid_search(query: str) -> list[str]:
            """BM25 + Dense → Weighted RRF 결과 docid 목록."""
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
        # <think> 태그 오염 → 버리지 않고 strip 후 사용
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

    # ── Phase 2.5: LLM 최종 문서 선별 (선택) ────────────────────────────────
    if skip_generation:
        llm_select = False  # 생성 생략 시 선별도 비활성화
    if llm_select:
        print("=== Phase 2.5: LLM 문서 선별 ===")
        print("LLM 로드 중 (문서 선별용) …")
        llm_sel = _load_reasoning_llm(cfg, phase3_backend)
        for r in rerank_results:
            if not r["is_science"] or not r["topk_ids"]:
                continue
            question = r["msg"][-1]["content"]
            selected = _llm_select_docs(
                query=question,
                candidate_ids=r["topk_ids"],
                doc_map=doc_map,
                llm=llm_sel,
                top_k=top_k_submit,
            )
            r["topk_ids_selected"] = selected
            print(f"  [{r['eid']}] {r['topk_ids'][:3]} → {selected}")
        llm_sel = unload_model(llm_sel)
        print("LLM(선별) 언로드 완료\n")
    else:
        for r in rerank_results:
            r["topk_ids_selected"] = r["topk_ids"][:top_k_submit]

    # ── Phase 3: LLM으로 전체 답변 생성 ─────────────────────────────────────
    rows_out: list[dict] = []

    if skip_generation:
        # 검색 실험 모드: LLM 없이 topk만 확정, 답변은 placeholder
        print("=== Phase 3: 생략 (--skip-generation) ===")
        print("topk 문서 선별 결과만 출력합니다. 답변은 placeholder로 대체됩니다.")
        for r in rerank_results:
            if not r["is_science"]:
                final_ids = []
                refs = []
            else:
                final_ids = r["topk_ids_selected"]
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
            final_ids = r["topk_ids_selected"]
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
        print(f"  [{r['eid']}] {r['standalone'][:40]}… → {r['topk_ids'][:top_k_submit]}")

    llm = unload_model(llm)
    return rows_out


def _load_llm(cfg: dict):
    """config에 지정된 LLM을 로드한다.

    우선순위:
    1. vllm.url 설정 시 → vLLM OpenAI 호환 API
    2. llm.checkpoint 설정 시 → 로컬 fine-tuned 체크포인트 (PEFT LoRA 포함)
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
    checkpoint = llm_cfg.get("checkpoint") or None  # null/빈 문자열 → None 처리
    max_new_tokens = llm_cfg.get("max_new_tokens", 512)
    context_window = llm_cfg.get("context_window", 4096)

    # checkpoint 지정 시: PEFT LoRA 어댑터 병합 후 로드
    if checkpoint:
        import os
        from pathlib import Path
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            # config 기준 상대경로 → repo root 기준으로 해석
            from ir_rag.config import repo_root_from
            ckpt_path = repo_root_from(Path.cwd()) / ckpt_path
        print(f"LLM: 체크포인트 로드 — base={base_model}, ckpt={ckpt_path}")
        try:
            from peft import PeftModel, PeftConfig
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            peft_cfg = PeftConfig.from_pretrained(str(ckpt_path))
            # base_model이 config와 다를 수 있으므로 PEFT config 우선
            actual_base = peft_cfg.base_model_name_or_path or base_model
            tokenizer = AutoTokenizer.from_pretrained(actual_base)
            model = AutoModelForCausalLM.from_pretrained(
                actual_base,
                torch_dtype=torch.bfloat16,
                device_map="auto",
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
            # fallback: checkpoint를 standalone 모델로 시도 (절대경로는 직접 로드)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), local_files_only=True)
            _model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
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
    parser.add_argument("--llm-select", action="store_true",
                        help="Phase 2.5 활성화: Reranker 이후 LLM이 최종 top-k 문서를 선별")
    parser.add_argument("--multi-field", action="store_true",
                        help="BM25 검색 시 title/keywords/summary/content 멀티필드 쿼리 사용 (F-2 색인 후)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Phase 3 LLM 답변 생성 생략. topk 문서 선별만 수행하여 검색 실험 속도 향상."
                             " 답변은 placeholder로 대체됨. MAP(문서 일치율) 측정용.")
    parser.add_argument(
        "--phase0-api",
        choices=("hf", "solar"),
        default="hf",
        help="Phase 0: standalone·HyDE·alt_query·과학판별 — hf=로컬 LLM, solar=Solar Pro API (SOLAR_API_KEY)",
    )
    parser.add_argument(
        "--phase3-api",
        choices=("hf", "solar"),
        default="hf",
        help="Phase 3: answer 생성 — hf=로컬 LLM, solar=Solar Pro API. --llm-select 시 동일 백엔드 사용.",
    )
    parser.add_argument(
        "--rrf-weights",
        default=None,
        metavar="W,W,W",
        help="Phase 1에서 standalone / HyDE / alt_query 다축 RRF 가중치 (쉼표로 3개). "
             "예: 0.5,0.25,0.25. 기본은 균등 1,1,1. 2축일 때는 standalone과 존재하는 축에만 적용.",
    )
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
        if phase0_cache_path is not None:
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
            llm_select=args.llm_select,
            use_multi_field=args.multi_field,
            skip_generation=args.skip_generation,
            phase0_backend=args.phase0_api,
            phase3_backend=args.phase3_api,
            axis_rrf_weights=axis_rrf_weights,
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
