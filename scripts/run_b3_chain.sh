#!/usr/bin/env bash
# B-3 체인: 트리플렛 생성 → Reranker Fine-tuning
set -e
cd /data/ephemeral/home/IR

LOG="artifacts/b3_chain.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "[$(date '+%H:%M:%S')] B-3 체인 시작"
echo "=========================================="

# ── B-3a: 트리플렛 생성 ────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] B-3a: build_reranker_triplets.py 시작"
echo "  (F-2 멀티필드 색인 적용 --use-multi-field)"
python3 scripts/build_reranker_triplets.py \
    --config config/default.yaml \
    --sft-data artifacts/sft_doc_qa.jsonl \
    --use-multi-field \
    --output artifacts/reranker_triplets.jsonl

TRIPLET_COUNT=$(wc -l < artifacts/reranker_triplets.jsonl)
echo "[$(date '+%H:%M:%S')] B-3a 완료 — ${TRIPLET_COUNT}건 트리플렛 생성"
echo ""

# ── B-3b: Reranker Fine-tuning ─────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] B-3b: train_reranker.py 시작 (QLoRA 4-bit)"
python3 scripts/train_reranker.py \
    --config config/default.yaml \
    --data artifacts/reranker_triplets.jsonl \
    --model Qwen/Qwen3-Reranker-8B \
    --output-dir artifacts/qwen3-reranker-8b-science \
    --epochs 3

echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] B-3 전체 완료."
echo ""
echo "적용 방법 (config/default.yaml 수정):"
echo "  reranker:"
echo "    model_name: \"artifacts/qwen3-reranker-8b-science\""
echo ""
echo "검증 명령:"
echo "  python scripts/export_submission.py \\"
echo "    --pipeline --config config/default.yaml \\"
echo "    --skip-dense --multi-field \\"
echo "    --phase0-cache artifacts/phase0_queries.csv \\"
echo "    --skip-generation \\"
echo "    --output artifacts/sample_submission_b3_retrieval.csv"
echo "=========================================="
