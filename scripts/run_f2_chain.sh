#!/usr/bin/env bash
# F-1 완료 후 F-2 전체 자동 실행 체인
# F-2a: build_synonyms.py
# F-2b: index_es.py (LMJelinekMercer + 동의어 + 메타 멀티필드)

set -e
cd /data/ephemeral/home/IR

LOG="artifacts/f2_chain.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "[$(date '+%H:%M:%S')] F-2 체인 대기 시작 (F-1 PID=1652535)"
echo "=========================================="

# F-1 완료 대기 (PID 폴링)
while kill -0 1652535 2>/dev/null; do
    COUNT=$(wc -l < artifacts/doc_metadata.jsonl 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] F-1 진행 중 ... ${COUNT}/4272건"
    sleep 60
done

COUNT=$(wc -l < artifacts/doc_metadata.jsonl 2>/dev/null || echo 0)
echo ""
echo "[$(date '+%H:%M:%S')] F-1 완료 확인 (${COUNT}건) — F-2 시작"
echo ""

# ── F-2a: 동의어 사전 빌드 ──────────────────────────────────────────────────
echo "=========================================="
echo "[$(date '+%H:%M:%S')] F-2a: build_synonyms.py 시작"
echo "=========================================="
python3 scripts/build_synonyms.py \
    --metadata artifacts/doc_metadata.jsonl \
    --api solar \
    --top-keywords 300 \
    --output artifacts/science_synonyms.txt
echo "[$(date '+%H:%M:%S')] F-2a 완료 → artifacts/science_synonyms.txt"
echo ""

# ── F-2b: ES 재인덱싱 ───────────────────────────────────────────────────────
echo "=========================================="
echo "[$(date '+%H:%M:%S')] F-2b: index_es.py 시작 (LMJelinekMercer + 동의어 + 멀티필드)"
echo "=========================================="
python3 scripts/index_es.py \
    --config config/default.yaml \
    --metadata artifacts/doc_metadata.jsonl \
    --synonyms artifacts/science_synonyms.txt \
    --lm-jelinek-mercer \
    --recreate
echo "[$(date '+%H:%M:%S')] F-2b 완료 — ES 재인덱싱 완료"
echo ""

echo "=========================================="
echo "[$(date '+%H:%M:%S')] F-2 전체 완료."
echo "다음 실험 명령:"
echo ""
echo "  python scripts/export_submission.py \\"
echo "    --pipeline --config config/default.yaml \\"
echo "    --skip-dense --multi-field \\"
echo "    --phase0-cache artifacts/phase0_queries.csv \\"
echo "    --skip-generation \\"
echo "    --output artifacts/sample_submission_f2_retrieval.csv"
echo "=========================================="
