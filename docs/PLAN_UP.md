# SFT 데이터 품질 개선 및 학습 계획

> **최우선 목표**: 제대로 된 SFT 학습이 선행되어야 RAG 파이프라인 전체가 정상 동작한다.  
> 학습 → 검색 품질 검증 → 서빙 UI 순으로 진행한다.

---

## 현황 진단

### 현재 SFT의 근본 문제

| 항목 | 현재 상태 | 문제 |
|------|----------|------|
| 학습 데이터 출처 | `eval.jsonl` 기반 (200건) | 문서가 아닌 질문에서 출발 → 문서-질문 불일치 |
| 검색 방식 | BM25 단독 | 표층 키워드 매칭으로 노이즈 문서 혼입 |
| Faithfulness | 0.19 ~ 0.37 수준 | 로컬 LLM이 문서 밖 지식으로 답변 생성 |
| 학습 패턴 | "문서 무관하게 알고 있는 지식으로 답하는 패턴" | RAG 행동 패턴 미학습 |
| 데이터 양 | 200건 상한 | eval.jsonl 샘플 수에 의존 |

### 핵심 원인

```
현재 흐름: 질문 → 검색(노이즈 문서) → 답변 생성 → 학습
                         ↑
           BM25 표층 매칭 → 명왕성이 나무 분류 문서에 혼입

모델이 배우는 것: "문서와 무관하게 알고 있는 지식으로 답하는 패턴"
결과: Faithfulness 0.19~0.37, Self-check 재생성 후에도 0.7 미달
```

**결론**: 문서가 아닌 질문에서 출발하기 때문에 문서-질문 정렬이 깨진다.  
`documents.jsonl`을 출발점으로 삼아야 문서-질문-답변 삼각 일관성이 보장된다.

---

## 전체 로드맵 (학습 최우선)

```
[Phase A] documents.jsonl 기반 Gold SFT 데이터 구축  ← 최우선
    A-1  문서 → 질문 역방향 생성 (핵심)
    A-2  문서 요약 학습 데이터 생성 (보조)
    A-3  Faithfulness 게이트로 품질 보증

[Phase B] eval.jsonl 기반 기존 데이터 정제           ← A 이후 혼합용
    B-1  Hybrid 검색 + Reranker 필터로 재구축
    B-2  정제된 데이터를 Phase A 데이터와 혼합

[Phase C] 학습 실행 및 검증
    C-1  SFT 학습 (Unsloth)
    C-2  Faithfulness 재측정으로 학습 효과 검증

[Phase D] 서빙 UI                                   ← 학습 완료 후
```

---

## Phase A — documents.jsonl 기반 Gold SFT 데이터 구축 ✅ 구현 완료

> **최우선 실행** | 문서-질문-답변 삼각 일관성이 설계 단계에서 보장됨

### 테스트 결과 (10건 실측, 2026-04-12)

| 항목 | 수치 |
|------|------|
| 문서당 평균 소요 | **16.9초** |
| Faithfulness 통과율 | 70% (7/10) |
| 재생성 발생률 | 40% |
| 최종 제외율 | 30% |

**소요 시간 예측** (BM25-only 기준, Solar API 병목)

| 대상 | 예상 시간 | 예상 수득 레코드 |
|------|-----------|----------------|
| 1,000건 (1차) | ~4.7시간 | ~700건 |
| 4,272건 (전체) | ~20시간 | ~2,990건 |

> 체크포인트 자동 저장 (`artifacts/sft_data_gold.jsonl.ckpt.json`) — 중단 시 동일 명령어로 이어받기 가능.  
> **야간 배치 실행 권장.**

---

### A-1. 문서 → 질문 역방향 생성 (핵심) ✅

**왜 역방향인가?**
- 문서에서 질문을 생성하면 답의 근거가 해당 문서임이 구조적으로 보장됨
- 현재 방식(질문 → 검색)과 달리 검색 노이즈 문제 원천 차단
- `documents.jsonl` 4,272건 활용으로 200건 상한 탈피

**처리 흐름**
```
documents.jsonl의 문서 d 선택
  → Solar API로 d를 근거로 답할 수 있는 과학 질문 q 생성  (eval.jsonl 스타일 few-shot 8개 참조)
  → q로 Hybrid 검색(BM25 + Dense) → Reranker top-K
  → 검색 결과에 d 포함 여부 확인
      포함 O → 검색 결과 문서 세트로 답변 생성
      포함 X → d 강제 포함 (position 0 삽입)하여 답변 생성
  → RAGAS Faithfulness ≥ 0.7 → 최종 포함
  → Faithfulness < 0.7 → 재생성 1회 → 그래도 미달 시 제외
```

**스크립트**: `scripts/build_sft_from_docs.py`

```bash
# 24GB GPU — BM25-only (야간 배치 권장, ~4.7시간/1000건)
python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar \
  --answer-api solar \
  --skip-dense \
  --skip-reranker \
  --max-docs 1000 \
  --output artifacts/sft_data_gold.jsonl

# 24GB GPU — BM25 + Dense (임베딩 모델 ~18GB, Reranker 제외)
python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar \
  --answer-api solar \
  --skip-reranker \
  --top-k-retrieve 20 \
  --top-k-rerank 5 \
  --max-docs 1000 \
  --output artifacts/sft_data_gold.jsonl

# 40GB+ GPU — 풀 파이프라인 (BM25 + Dense + Reranker)
python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar \
  --answer-api solar \
  --top-k-retrieve 20 \
  --top-k-rerank 5 \
  --faithfulness-threshold 0.7 \
  --max-docs 1000 \
  --output artifacts/sft_data_gold.jsonl

# 이어받기 (중단 후 재시작) — 동일 명령어 재실행으로 자동 처리
```

**VRAM 가이드**

| 플래그 | VRAM | 비고 |
|--------|------|------|
| `--skip-dense --skip-reranker` | GPU 불필요 | BM25 + Solar API만 사용 |
| `--skip-reranker` | ~18GB | 임베딩 모델 로딩 |
| 기본값 | ~40GB | Dense + Reranker 순차 로드 |

**한계 및 주의**
- 문서 1건당 Solar API 호출 2~4회 (질문 + 답변 + 재생성 + Faithfulness 평가)
- BM25-only 시 source doc 강제 삽입 비율이 높아 Faithfulness 통과율 저하 가능 (실측 70%)
  → Dense + Reranker 사용 시 통과율 개선 기대

---

### A-2. 문서 요약 학습 데이터 생성 (보조) ✅

**목적**: 모델이 문서 핵심을 추출하는 능력을 키워 RAG 답변 품질을 간접 강화

**방법**
```
문서 d → Solar API로 3~5문장 요약 생성
→ (문서, 요약) 쌍을 SFT 데이터로 추가
→ 시스템 프롬프트: "제공된 문서를 핵심만 간결하게 요약하세요"
```

**주의**: 4B 소형 모델은 task 충돌 가능성 있음.  
QA 데이터(A-1)와 요약 데이터 비율을 **8:2** 수준으로 유지 권장.

**스크립트**: `scripts/build_sft_summary.py`

```bash
python scripts/build_sft_summary.py \
  --config config/default.yaml \
  --api solar \
  --max-docs 500 \
  --output artifacts/sft_summary_data.jsonl

# 이어받기 — 동일 명령어 재실행
```

---

### A-3. Faithfulness 게이트 ✅ (A-1에 통합 구현)

A-1에서 문서별로 자동 적용됨:

```
각 row에 대해:
  RAGAS Faithfulness 점수 산출 (Solar API)
  ≥ 0.7  → 학습 데이터 포함
  < 0.7  → 재생성 1회 (retry 프롬프트로 강화)
           그래도 미달 → 제외
```

> Solar API 타임아웃 방지: `nest_asyncio.apply()` 적용 완료 (`build_sft_from_docs.py` 내장)

---

## Phase B — eval.jsonl 기반 기존 데이터 정제

> Phase A 완료 후 혼합 데이터로 활용. 단독으로는 품질 한계 있음.

### B-1. Hybrid 검색 + Reranker 필터로 재구축

현재 BM25 단독 → **BM25 + Dense RRF → Reranker 필터**로 업그레이드

```
query (phase0_csv의 standalone 활용)
  → BM25 top-20 + Dense top-20 → RRF(k=60) → 후보 top-20
  → Reranker score 산출
  → score < 0.3 문서 제거
  → Solar API 답변 생성
  → Faithfulness 게이트 통과 시 포함
```

**스크립트**: `build_sft_data.py` 확장 (신규 인자 `--hybrid`, `--reranker-threshold`)

### B-2. Phase A + B 데이터 혼합

```
최종 학습 데이터 = Phase A (Gold) + Phase B (정제)
권장 비율: Gold 70% + 정제 30%
출력: artifacts/sft_data_final.jsonl
```

---

## Phase C — 학습 실행 및 검증

### C-1. SFT 학습

```bash
python scripts/train_sft.py \
  --data artifacts/sft_data_final.jsonl \
  --model Qwen/Qwen3.5-4B \
  --output artifacts/qwen35-4b-science-rag
```

### C-2. 학습 효과 검증

학습 전/후 Faithfulness를 동일 eval 샘플로 비교:

```bash
# 학습 후 파이프라인 재실행 (phase0 캐시 활용)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --phase0-cache artifacts/phase0_queries.csv \
  --top-k-retrieve 20 --top-k-rerank 10

# RAGAS 평가
python scripts/ragas_eval.py \
  --submission artifacts/sample_submission.csv \
  --eval data/eval.jsonl \
  --documents data/documents.jsonl
```

**목표 지표**

| 지표 | 현재 | 목표 |
|------|------|------|
| Faithfulness | 0.19 ~ 0.37 | ≥ 0.70 |
| Self-check 통과율 | 낮음 | ≥ 80% |

---

## Phase D — 서빙 UI

> **학습 검증 완료 후 진행**. 학습이 제대로 되지 않은 상태에서 UI를 붙여도 품질 보장 불가.

### 기술 스택

| 구분 | 선택 | 이유 |
|------|------|------|
| Frontend | Vanilla HTML/CSS/JS (단일 파일) | 프레임워크 불필요, 즉시 배포 가능 |
| 실시간 통신 | SSE (Server-Sent Events) | 단방향 스트림에 최적, 구현 단순 |
| Backend | FastAPI + `uvicorn` | 기존 Python 환경, async 스트리밍 지원 |
| LLM 연결 | vLLM OpenAI 호환 API (`localhost:8000`) | `serve_vllm.py` 재사용, `stream=True` 지원 |
| 검색 | 기존 ES + Qdrant + Reranker 모듈 재사용 | 추가 인프라 불필요 |

### 디렉터리 구조

```
scripts/serve_app.py   # FastAPI 앱 (신규)
static/index.html      # 단일 HTML (CSS·JS 내장, 신규)
```

### 서비스 플로우

```
[사용자 입력] → POST /api/chat
                    │
                    ├─ 1. Query Rewrite (vLLM)
                    ├─ 2. Hybrid Search (ES BM25 + Qdrant Dense → RRF)
                    ├─ 3. Reranker → top-3 doc_id + content[:50]
                    └─ 4. LLM 스트리밍 (vLLM stream=True)

SSE events:
  token  → 프론트에서 글자 단위 렌더링
  docs   → 검색 완료 시 top-3 문서 카드 표시
  done   → 완료 신호
  error  → 오류 메시지
```

### 화면 구성

```
┌─────────────────────────────────────────────────┐
│  ◉  Science RAG                    [새 질문] 버튼 │
├─────────────────────────────────────────────────┤
│  [사용자] 나무 분류 방법은?                        │
│  [AI] 나무의 분류는 형태적 특징과 분자 수준...▌    │
│  ┌──────── 참고 문서 ─────────────────────────┐  │
│  │ 📄 doc042  한 학생이 다양한 종류의 나무를…  │  │
│  │ 📄 doc017  생물학에서 일부 생물체의 분류…   │  │
│  │ 📄 doc089  같은 속에 속한 나무들은 종류…   │  │
│  └────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  [질문을 입력하세요...]               [전송 ▶]    │
└─────────────────────────────────────────────────┘
```

### 구현 체크리스트

- [ ] `scripts/serve_app.py` — FastAPI 앱, `/api/chat` SSE 엔드포인트
- [ ] `static/index.html` — 채팅 UI (CSS·JS 내장)
- [ ] Query Rewrite → Hybrid Search → Rerank 파이프라인 연결
- [ ] vLLM 스트리밍 연결 및 SSE 이벤트 전달
- [ ] `새 질문` 버튼: 히스토리 초기화
- [ ] 문서 카드: doc_id + content 50자 truncation
- [ ] 오류 처리 (vLLM 미기동, ES/Qdrant 미응답 등)

---

## 전체 실행 순서 요약

| 순서 | Phase | 스크립트 | 상태 | 선행 조건 |
|------|-------|---------|------|---------|
| 1 | A-1 | `build_sft_from_docs.py` | ✅ 구현 완료 (야간 배치 예정) | Solar API |
| 2 | A-2 | `build_sft_summary.py` | ✅ 구현 완료 | Solar API |
| 3 | B-1 | `build_sft_data.py` 확장 | ⬜ 미시작 | Qdrant + Reranker GPU |
| 4 | B-2 | 데이터 혼합 스크립트 | ⬜ 미시작 | A-1, B-1 완료 |
| 5 | C-1 | `train_sft.py` | ⬜ 미시작 | sft_data_final.jsonl |
| 6 | C-2 | `ragas_eval.py` | ⬜ 미시작 | 학습 완료 |
| 7 | D   | `serve_app.py`, `index.html` | ⬜ 미시작 | C-2 검증 완료 |

> **진행 원칙**: 학습 검증(C-2) 완료 전까지 Phase D는 시작하지 않는다.

---

## 야간 배치 실행 명령어 (현행)

```bash
# A-1 (1000건, ~4.7시간) — 야간 nohup 실행
nohup python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar --answer-api solar \
  --skip-dense --skip-reranker \
  --max-docs 1000 \
  --output artifacts/sft_data_gold.jsonl \
  > artifacts/sft_from_docs.log 2>&1 &

# A-2 (500건, ~1.5시간) — A-1 완료 후 또는 병렬 실행
nohup python scripts/build_sft_summary.py \
  --config config/default.yaml \
  --api solar \
  --max-docs 500 \
  --output artifacts/sft_summary_data.jsonl \
  > artifacts/sft_summary.log 2>&1 &

# 진행 상황 확인
tail -f artifacts/sft_from_docs.log
tail -f artifacts/sft_summary.log

# 생성된 레코드 수 확인
wc -l artifacts/sft_data_gold.jsonl
wc -l artifacts/sft_summary_data.jsonl
```
