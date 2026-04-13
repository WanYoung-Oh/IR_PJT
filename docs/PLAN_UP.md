# SFT 데이터 품질 개선 및 학습 계획

> **최우선 목표**: 제대로 된 SFT 학습이 선행되어야 RAG 파이프라인 전체가 정상 동작한다.  
> 학습 → 검색 품질 검증 → 서빙 UI 순으로 진행한다.

---

## 진행 현황 (2026-04-13 기준)

| Phase    | 항목                                               | 상태       | 결과                                                               |
| -------- | -------------------------------------------------- | ---------- | ------------------------------------------------------------------ |
| 인프라   | Qdrant 인덱싱                                      | ✅ 완료    | 4,272건 인덱싱 (UUID5 직접 upsert 방식으로 LlamaIndex 버그 우회)   |
| B-1      | Hybrid+Reranker+Faithfulness 데이터 재구축         | ✅ 완료    | 포함 158건 / 제외 62건 (통과율 72%)                                |
| A-1      | documents.jsonl 기반 QA 생성 (2000건)              | ✅ 완료    | 포함 1,747건 / 통과율 87.4% → `artifacts/sft_doc_qa.jsonl` (8.9MB) |
| B-2      | A-1 + B-1 데이터 혼합                              | ✅ 완료    | 1,905건 → `artifacts/sft_data_final.jsonl` (9.2MB)                 |
| C-1      | SFT 학습 (Qwen3.5-4B QLoRA 4-bit)                  | ✅ 완료    | `artifacts/qwen35-4b-science-rag` 저장 완료                        |
| C-2      | export_submission.py 재실행 (fine-tuned 모델 적용) | ✅ 완료    | `artifacts/sample_submission_clean.csv` 생성                       |
| C-3      | 리더보드 제출 및 평가                              | ✅ 완료    | **MAP=0.3364, MRR=0.3379** (베이스라인 0.6682 대비 대폭 하락)      |
| D-1      | MAP 하락 원인 분석 및 파이프라인 수정              | ✅ 완료    | Dense 노이즈·alt_query 오염 확인, BM25 단독 + Phase 0 단순화 적용  |

### 주요 이슈 해결 기록

- **Qdrant 0건 버그**: LlamaIndex `upload_points()`가 비-UUID docid를 HTTP 400으로 거부하나 예외를 삼킴 → `index_qdrant.py`를 `uuid.uuid5` 직접 upsert 방식으로 전면 교체
- **B-1 Dense 검색 정상화**: Qdrant 수정 후 `[283] RRF 후보 20건 (BM25 0 + Dense 20)` 확인 — BM25 히트 없는 쿼리도 Dense로 커버
- **A-1 데이터 폐기**: 기존 `sft_data.jsonl` (200건, BM25-only, Faithfulness 미검증)은 품질 미달로 학습에 사용하지 않음

---

## MAP 하락 원인 분석 (2026-04-13)

### 리더보드 결과

| 실험 | MAP | MRR | 비고 |
|------|-----|-----|------|
| 베이스라인 (ES BM25 단독) | **0.6682** | 0.6682 | 대회 제공 sample_submission 기준 |
| Hybrid+Reranker+fine-tuned LLM | **0.3364** | 0.3379 | 대폭 하락 |

### 확인된 원인 2가지

**원인 1: Dense 검색(Qdrant)이 BM25보다 부정확**
- topk 평균 일치율: sample 대비 **12%** (138/200건 완전 불일치)
- Qwen3-Embedding-8B가 의미적으로 유사하지만 덜 직접적인 문서를 검색
- RRF 3축 융합 과정에서 정답 BM25 문서가 Dense 노이즈에 밀려남

```
예시 (eval_id=18 "기체의 부피나 형태가 왜 일정하지 않을까?"):
  BM25 정답: "기체는 일정한 부피와 형태를 가지고 있지 않습니다..."
  Dense 결과: 프로판, 고체 설명 등 간접 관련 문서
```

**원인 2: alt_query `<think>` 태그 오염**
- 200건 과학 쿼리 중 **151건(75%)** 의 alt_query에 `<think>` 태그 미제거
- 오염된 alt_query가 RRF 3축 검색에 투입되어 완전히 엉뚱한 문서 유입
- Qwen3 모델의 thinking 토큰이 `<think>` 태그 채로 출력된 버그

### 적용한 수정사항

| 항목 | 수정 내용 |
|------|----------|
| `--skip-dense` 옵션 추가 | `export_submission.py` — BM25 단독 검색 가능 |
| Phase 0 단순화 | 단일턴은 원본 쿼리 그대로, HyDE·alt_query 생성 제거 |
| 멀티턴만 LLM 재작성 | 20건만 해당 (`build_search_query` 활용) |
| `<think>` 필터링 | 검색 전 alt_query/hyde_doc에서 `<think>` 포함 시 무시 |
| `ragas_eval.py` 수정 | `load_dotenv` 추가, `context_recall`·`answer_relevancy` 제거 (Solar 미지원) |

### 다음 실험 계획

```bash
# BM25 단독 + Phase 0 단순화 적용 → MAP 회복 확인
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense \
  --output artifacts/sample_submission_bm25only.csv
```

예상 결과: MAP ~0.6682 수준 회복 → 이후 Dense 재투입 여부 결정

---

## 현황 진단

### 현재 SFT의 근본 문제

| 항목             | 현재 상태                                      | 문제                                         |
| ---------------- | ---------------------------------------------- | -------------------------------------------- |
| 학습 데이터 출처 | `eval.jsonl` 기반 (200건)                      | 문서가 아닌 질문에서 출발 → 문서-질문 불일치 |
| 검색 방식        | BM25 단독                                      | 표층 키워드 매칭으로 노이즈 문서 혼입        |
| Faithfulness     | 0.19 ~ 0.37 수준                               | 로컬 LLM이 문서 밖 지식으로 답변 생성        |
| 학습 패턴        | "문서 무관하게 알고 있는 지식으로 답하는 패턴" | RAG 행동 패턴 미학습                         |
| 데이터 양        | 200건 상한                                     | eval.jsonl 샘플 수에 의존                    |

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

## Phase A — documents.jsonl 기반 Gold SFT 데이터 구축 ✅ 완료

> **최우선 실행** | 문서-질문-답변 삼각 일관성이 설계 단계에서 보장됨

### 완료 결과 (2026-04-12)

| 항목              | 수치                                               |
| ----------------- | -------------------------------------------------- |
| 처리 목표 문서 수 | 2,000건                                            |
| Faithfulness 통과 | **1,747건** (통과율 87.4%)                         |
| 출력 파일         | `artifacts/sft_doc_qa.jsonl` (8.9MB)               |
| 검색 방식         | BM25+Dense (--skip-reranker), source doc 강제 포함 |

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
# 현재 실행 중 (24GB GPU — BM25 + Dense, Reranker 제외, 2000건 목표)
nohup python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar --answer-api solar \
  --skip-reranker \
  --max-docs 2000 \
  --output artifacts/sft_doc_qa.jsonl \
  > artifacts/build_sft_doc_qa.log 2>&1 &

# 이어받기 (중단 후 재시작) — 동일 명령어 재실행으로 자동 처리
```

**VRAM 가이드**

| 플래그                         | VRAM       | 비고                       |
| ------------------------------ | ---------- | -------------------------- |
| `--skip-dense --skip-reranker` | GPU 불필요 | BM25 + Solar API만 사용    |
| `--skip-reranker`              | ~18GB      | 임베딩 모델 로딩           |
| 기본값                         | ~40GB      | Dense + Reranker 순차 로드 |

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

### B-1. Hybrid 검색 + Reranker 필터로 재구축 ✅ 완료 (2026-04-12)

**결과**: 포함 158건 / 제외 62건 / 오류 0건 (통과율 72%)  
**출력**: `artifacts/sft_b1_data.jsonl` (298KB)

```
query (phase0_csv의 standalone 활용)
  → BM25 top-20 + Dense top-20 → RRF(k=20) → 후보 top-20
  → Qwen3-Reranker-8B score 산출 (threshold 0.3)
  → Solar API 답변 생성
  → RAGAS Faithfulness ≥ 0.7 통과 시 포함 (재생성 1회 허용)
```

**실행 명령**:

```bash
nohup python scripts/build_sft_data.py \
  --config config/default.yaml \
  --hybrid --reranker-threshold 0.3 \
  --faithfulness-gate --faithfulness-threshold 0.7 \
  --answer-api solar \
  --output artifacts/sft_b1_data.jsonl \
  > artifacts/build_sft_b1.log 2>&1 &
```

### B-2. Phase A + B 데이터 혼합 ✅ 완료 (2026-04-12)

> ⚠️ 기존 `sft_data.jsonl` (A-1 이전 생성, BM25-only, Faithfulness 미검증)은 **사용하지 않음**

```
최종 학습 데이터 = Phase A (sft_doc_qa.jsonl) + Phase B-1 (sft_b1_data.jsonl)
실제 규모: 1,747건 + 158건 = 1,905건 (9.2MB)
출력: artifacts/sft_data_final.jsonl ✅
```

```bash
cat artifacts/sft_doc_qa.jsonl artifacts/sft_b1_data.jsonl > artifacts/sft_data_final.jsonl
```

---

## Phase B-3 — Reranker Fine-tuning 검토 (A-1 완료 후)

> **선택적 실행** | A-1 데이터로 트리플렛 자동 구성 가능 여부 확인 후 결정

### 아이디어 개요

Qwen3-Reranker-8B를 한국어 과학 도메인에 맞게 fine-tuning하여  
일반 도메인 reranker의 한계(동의어·전문 용어 매칭 부족)를 보완한다.

### 학습 데이터 구성 (A-1 부산물 활용)

```
A-1 처리 흐름에서 자동 생성 가능한 트리플렛:

  (query=생성된 질문, positive=source_docid, negatives=BM25+Dense 검색 결과 중 source 제외)

예상 규모: ~1,680건 트리플렛 (A-1 통과 건수 기준)
```

| 데이터 소스     | positive                          | negative                   | 품질 |
| --------------- | --------------------------------- | -------------------------- | ---- |
| A-1 source doc  | 질문 생성 원본 문서 (구조적 보장) | 동일 쿼리 검색 결과 나머지 | 양호 |
| B-1 통과 데이터 | Faithfulness 통과 doc_ids         | RRF 후보 중 미사용 문서    | 중간 |

### 학습 방식

```
모델: Qwen/Qwen3-Reranker-8B
방식: 4-bit QLoRA (bitsandbytes + Unsloth, 24GB VRAM 내 가능)
손실: Binary Cross-Entropy (query, doc, label=0/1)
      또는 Pairwise Ranking Loss (positive > negative margin)
배치: (query, pos, neg) 트리플렛 → in-batch negatives 활용
```

### 검증 방법

```bash
# fine-tuning 전/후 reranker 교체 후 MAP 비교
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --phase0-cache artifacts/phase0_queries.csv

python scripts/run_competition_map.py \
  --submission artifacts/sample_submission.csv
```

### 판단 기준

- A-1 완료 후 트리플렛 ~1,000건 이상 확보 가능하면 → **진행**
- MAP 개선이 +0.02 이상이면 채택, 미만이면 기본 Qwen3-Reranker-8B 유지

> ⚠️ 트리플렛 생성 스크립트(`build_reranker_triplets.py`)는 A-1 완료 시점에 구현 예정

---

## Phase C — 학습 실행 및 검증

### C-1. SFT 학습 ✅ 완료 (2026-04-12)

| 항목      | 값                                                    |
| --------- | ----------------------------------------------------- | --- |
| 모델      | Qwen/Qwen3.5-4B                                       |
| 방식      | QLoRA 4-bit (Unsloth)                                 |
| 데이터    | `sft_data_final.jsonl` (1,905건)                      |
| Epochs    | 3 / Steps                                             | 360 |
| 배치      | per_device=1, grad_accum=16 → 유효 배치 16            |
| LoRA      | r=16, alpha=16 / trainable params 0.47% (21M / 4.56B) |
| 저장 위치 | `artifacts/qwen35-4b-science-rag` ✅                  |

**비고**: Flash Attention 2 broken → Xformers fallback 자동 적용 (성능 영향 없음)

```bash
nohup python scripts/train_sft.py \
  --data artifacts/sft_data_final.jsonl \
  --model Qwen/Qwen3.5-4B \
  --output-dir artifacts/qwen35-4b-science-rag \
  --epochs 3 --max-seq-len 2048 \
  > artifacts/train_sft.log 2>&1 &
```

### C-2. 파이프라인 재실행 (fine-tuned 모델 적용) 🔄 진행 중 (2026-04-12)

`artifacts/qwen35-4b-science-rag` 모델을 적용하여 `export_submission.py` 전체 파이프라인 재실행 중.  
phase0 캐시(`artifacts/phase0_queries.csv`) 활용으로 쿼리 재작성 단계는 스킵.

```bash
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --phase0-cache artifacts/phase0_queries.csv \
  --top-k-retrieve 20 --top-k-rerank 10
```

### C-3. 학습 효과 검증

파이프라인 재실행 완료 후 Faithfulness를 동일 eval 샘플로 비교:

```bash
# RAGAS 평가
python scripts/ragas_eval.py \
  --submission artifacts/sample_submission.csv \
  --eval data/eval.jsonl \
  --documents data/documents.jsonl
```

**목표 지표**

| 지표              | 현재        | 목표   |
| ----------------- | ----------- | ------ |
| Faithfulness      | 0.19 ~ 0.37 | ≥ 0.70 |
| Self-check 통과율 | 낮음        | ≥ 80%  |

---

## Phase D — 서빙 UI

> **학습 검증 완료 후 진행**. 학습이 제대로 되지 않은 상태에서 UI를 붙여도 품질 보장 불가.

### 기술 스택

| 구분        | 선택                                    | 이유                                       |
| ----------- | --------------------------------------- | ------------------------------------------ |
| Frontend    | Vanilla HTML/CSS/JS (단일 파일)         | 프레임워크 불필요, 즉시 배포 가능          |
| 실시간 통신 | SSE (Server-Sent Events)                | 단방향 스트림에 최적, 구현 단순            |
| Backend     | FastAPI + `uvicorn`                     | 기존 Python 환경, async 스트리밍 지원      |
| LLM 연결    | vLLM OpenAI 호환 API (`localhost:8000`) | `serve_vllm.py` 재사용, `stream=True` 지원 |
| 검색        | 기존 ES + Qdrant + Reranker 모듈 재사용 | 추가 인프라 불필요                         |

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

| 순서 | Phase  | 스크립트                     | 상태         | 결과                                        |
| ---- | ------ | ---------------------------- | ------------ | ------------------------------------------- |
| 0    | 인프라 | `index_qdrant.py`            | ✅ 완료      | 4,272건 Qdrant 인덱싱                       |
| 1    | A-1    | `build_sft_from_docs.py`     | ✅ 완료      | 1,747건 / 통과율 87.4%                      |
| 2    | A-2    | `build_sft_summary.py`       | ⬜ 선택적    | 필요 시 추후 실행                           |
| 3    | B-1    | `build_sft_data.py`          | ✅ 완료      | 158건 / 통과율 72%                          |
| 4    | B-2    | 데이터 혼합                  | ✅ 완료      | 1,905건 → `sft_data_final.jsonl`            |
| 5    | B-3    | `build_reranker_triplets.py` | 🔍 검토 예정 | C-1 완료 후 MAP 확인 후 결정                |
| 6    | C-1    | `train_sft.py`               | ✅ 완료      | `artifacts/qwen35-4b-science-rag` 저장 완료 |
| 7    | C-2    | `export_submission.py`       | 🔄 진행 중   | fine-tuned 모델 적용, 결과 대기 중          |
| 8    | C-3    | `ragas_eval.py`              | ⬜ 미시작    | C-2 완료 후                                 |
| 9    | D      | `serve_app.py`, `index.html` | ⬜ 미시작    | C-3 검증 완료 후                            |

> **진행 원칙**: 학습 검증(C-2) 완료 전까지 Phase D는 시작하지 않는다.

---

## 다음 단계 (C-2 완료 후)

```bash
# 1. RAGAS 품질 평가 (C-3) — fine-tuned 모델 적용 결과 검증
python scripts/ragas_eval.py \
  --submission artifacts/sample_submission.csv \
  --eval data/eval.jsonl \
  --documents data/documents.jsonl

# 2. (선택) MAP 평가
python scripts/run_competition_map.py \
  --submission artifacts/sample_submission.csv
```

## 쿼리 확장 개선 내역 (2026-04-12)

`export_submission.py` Phase 0의 `alt_query` 생성 프롬프트를 강화했다.

**변경 전**: "다음 과학 질문을 다른 표현으로 재작성하세요 (1개만)."

**변경 후**: 동의어·유사어·상위어·하위어를 명시적으로 사용하도록 지시 + few-shot 예시 2개 추가

```
예시: '세제 거품 원리' → '비누 계면활성제 기포 생성 원리'
예시: '광합성 명반응' → '엽록체 빛 에너지 ATP 합성 반응'
```

→ BM25 recall 개선 기대 (동의어로 표현된 문서도 검색 커버리지에 포함)
