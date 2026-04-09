#!/usr/bin/env python3
"""RAGAS 기반 RAG 품질 평가 (Faithfulness / Answer Relevancy / Context Recall).

설계 문서 §⑥ 평가 「RAGAS + LangSmith」 참조.

사용 예시:
    # 기본 평가 (LangSmith 추적 없음)
    python scripts/ragas_eval.py \
        --submission artifacts/sample_submission.csv \
        --eval data/eval.jsonl \
        --documents data/documents.jsonl \
        --config config/default.yaml

    # LangSmith 추적 활성화
    LANGCHAIN_API_KEY=lsv2_... LANGCHAIN_TRACING_V2=true \\
    python scripts/ragas_eval.py ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl, write_jsonl


def build_ragas_dataset(
    submission_rows: list[dict],
    eval_rows: list[dict],
    doc_map: dict[str, str],
    top_k_context: int = 3,
) -> "Dataset":  # noqa: F821
    """제출 파일과 문서 코퍼스에서 RAGAS 입력 Dataset을 구성한다."""
    from datasets import Dataset

    eval_by_id = {int(r["eval_id"]): r for r in eval_rows}
    questions, answers, contexts = [], [], []

    for row in submission_rows:
        eid = int(row["eval_id"])
        sample = eval_by_id.get(eid)
        if sample is None:
            continue
        question = sample["msg"][-1]["content"]
        answer = row.get("answer", "")
        topk = row.get("topk") or []
        ctx = [doc_map[did] for did in topk[:top_k_context] if did in doc_map]

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="sample_submission.csv 경로")
    parser.add_argument("--eval", default=None, help="eval.jsonl 경로 (미지정 시 config 사용)")
    parser.add_argument("--documents", default=None, help="documents.jsonl 경로 (미지정 시 config 사용)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="결과 JSONL 출력 경로 (미지정 시 stdout)")
    parser.add_argument("--top-k-context", type=int, default=3, help="컨텍스트에 포함할 topk 문서 수")
    args = parser.parse_args()

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError as e:
        raise SystemExit(
            f"RAGAS 미설치: {e}\n"
            "pip install ragas datasets 후 재실행하세요."
        )

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))

    sub_path = Path(args.submission)
    if not sub_path.is_absolute():
        sub_path = root / sub_path
    eval_path = Path(args.eval) if args.eval else root / cfg["paths"]["eval"]
    doc_path = Path(args.documents) if args.documents else root / cfg["paths"]["documents"]

    print(f"제출 파일: {sub_path}")
    print(f"eval.jsonl: {eval_path}")
    print(f"documents: {doc_path}")

    submission_rows = list(iter_jsonl(sub_path))
    eval_rows = list(iter_jsonl(eval_path))
    doc_map: dict[str, str] = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}

    dataset = build_ragas_dataset(
        submission_rows, eval_rows, doc_map,
        top_k_context=args.top_k_context,
    )

    # LangSmith 추적 설정 (환경변수 LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2 참조)
    if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith 추적 활성화")

    print(f"평가 중 … ({len(dataset)}건)")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        mean_scores = df[["faithfulness", "answer_relevancy", "context_recall"]].mean()
        print("\n── RAGAS 평가 결과 ──")
        for metric, score in mean_scores.items():
            print(f"  {metric}: {score:.4f}")

        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = root / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(out_path, df.to_dict(orient="records"))
            print(f"\n결과 저장 → {out_path}")
    else:
        print(result)


if __name__ == "__main__":
    main()
